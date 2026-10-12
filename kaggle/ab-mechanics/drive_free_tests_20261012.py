"""Adopt compact v2, collect proof, then launch forced mechanics once on free quota.

Never submits, retries gameplay, uses Runpod, or relaunches itself. All cloud
reads and failures are retained. Launch recovery is the executor's decision.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from free_tests_20261012 import RECEIPTS, CONTROL_OUTPUT

LOG = Path('/Users/morkeeth/CODE/fleet-ops/reviews/ARC-DAILY-LOG.md')

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')

def run(command):
    result = subprocess.run(command, text=True, capture_output=True, timeout=180)
    return {'command':command, 'returncode':result.returncode, 'stdout':result.stdout, 'stderr':result.stderr}

def notify_fable(message):
    receipt=run(['cmux','agent','message','surface:1000020000','--from','Codex ARC KAGGLE',message])
    with (RECEIPTS/'fable-messages.jsonl').open('a') as file:
        file.write(json.dumps({'read_at':now(),'receipt':receipt})+'\n')

def status(api, ref):
    return json.loads(str(api.kernels_status(ref)))

def snapshot(api):
    record = {'read_at':now(), 'competitions':{}, 'kernels':[]}
    for competition in ('arc-prize-2026-arc-agi-3', 'arc-prize-2026-arc-agi-2'):
        receipt = run(['kaggle','competitions','submissions','-c',competition,'--csv'])
        record['competitions'][competition] = {'receipt':receipt,
            'rows':list(csv.DictReader(io.StringIO(receipt['stdout']))) if receipt['returncode']==0 else []}
    quota = api.quota_view()
    # SDK JSON timedelta rendering omits whole days; retain explicit seconds.
    record['quota'] = json.loads(str(quota))
    record['quota_seconds'] = {name: {'used':q.time_used.total_seconds(),
        'allowed':q.total_time_allowed.total_seconds(), 'reserved':q.time_reserved.total_seconds(),
        'remaining':(q.total_time_allowed-q.time_used).total_seconds()}
        for name,q in (('GPU',quota.gpu_quota),('TPU',quota.tpu_quota)) if q is not None}
    listing = run(['kaggle','kernels','list','--mine','--page-size','100','--csv'])
    record['kernel_listing'] = listing
    refs = {row['ref'] for row in csv.DictReader(io.StringIO(listing['stdout'])) if '/arc' in row['ref']}
    refs.update(('oscarmorke/arc3-intel-compact-20261012', 'oscarmorke/arc3-free-mount-probe-20261012'))
    def read(ref):
        try:return {'ref':ref, **status(api,ref)}
        except Exception as exc:return {'ref':ref, 'read_error':str(exc)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        record['kernels'] = list(pool.map(read, sorted(refs)))
    record['outstanding_kernels'] = [r for r in record['kernels'] if r.get('status') in ('RUNNING','QUEUED')]
    path = RECEIPTS / ('live-state-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
    save(path, record)
    print(json.dumps({'snapshot':str(path),'outstanding':record['outstanding_kernels'],'quota':record['quota']}),flush=True)
    return record

def download(api, ref, dest):
    dest.mkdir(exist_ok=True)
    # Exclude movies; retain gameplay, request traces, adoption and failure logs.
    page = None
    all_files = []
    while True:
        files, page = api.kernels_output(ref, str(dest),
            file_pattern=r'(\.json|\.jsonl|\.log|\.py|\.parquet)$', force=True,
            quiet=True, page_token=page, page_size=200)
        all_files.extend(files)
        if not page:break
    return all_files

def compare(arm, dest, terminal, freeze):
    result = {'arm':arm,'read_at':now(),'kernel_status':terminal,'valid_pairs':0,
              'promote':False,'verdict':'retain-control','output_dir':str(dest),
              'control':freeze['control_summary'], 'private_gameplay_only':True}
    if not (dest/'benchmark.json').is_file() or not (dest/'experiment-complete.json').is_file():
        result['reason']='Missing completed benchmark/output; no gameplay score comparison'
        save(RECEIPTS/(arm+'-verdict.json'),result)
        return result
    benchmark=json.loads((dest/'benchmark.json').read_text())
    rows=benchmark.get('game_runs',[])
    control=json.loads(CONTROL_OUTPUT.read_text())['game_runs']
    ids=freeze['games']
    valid=(terminal.get('status')=='COMPLETE' and len(rows)==25 and
        len({r['game_id'] for r in rows})==25 and {r['game_id'] for r in rows}==set(ids) and
        all(r.get('state') in ('won','gave_up') and isinstance(r.get('final_score'),(int,float))
            and math.isfinite(r['final_score'])
            and isinstance(r.get('levels_completed'),int) for r in rows))
    result['benchmark_sha256']=sha(dest/'benchmark.json')
    result['valid']=valid
    if not valid:
        result['reason']='Kernel or terminal-game validity gate failed'
        save(RECEIPTS/(arm+'-verdict.json'),result)
        return result
    by_id={r['game_id']:r for r in rows}
    pairs=[]
    for c in control:
        t=by_id[c['game_id']]
        pairs.append({'game_id':c['game_id'],'control_score':c['final_score'],'treatment_score':t['final_score'],
                      'score_delta':t['final_score']-c['final_score'],
                      'control_levels':c['levels_completed'],'treatment_levels':t['levels_completed'],
                      'control_won':c['state']=='won','treatment_won':t['state']=='won',
                      'control_actions':len(c['history']),'treatment_actions':len(t['history'])})
    result.update(valid_pairs=25, pairs=pairs,
        treatment={'games':25,'wins':sum(r['state']=='won' for r in rows),
                   'levels':sum(r['levels_completed'] for r in rows),
                   'mean_score':sum(r['final_score'] for r in rows)/25,
                   'actions':sum(len(r['history']) for r in rows)},
        paired_mean_score_delta=sum(p['score_delta'] for p in pairs)/25)
    if arm=='compact':
        events=[json.loads(line) for line in (dest/'compact-events.jsonl').read_text().splitlines() if line.strip()] if (dest/'compact-events.jsonl').is_file() else []
        drops=sum(e.get('dropped_messages',0) for e in events)
        result['context_drop_events']=sum(e.get('dropped_messages',0)>0 for e in events)
        result['dropped_messages']=drops
        result['promote']=(drops>0 and result['treatment']['wins']>=freeze['control_summary']['wins'] and
            (result['treatment']['levels']>freeze['control_summary']['levels'] or result['paired_mean_score_delta']>0))
    else:
        events=[json.loads(line) for line in (dest/'mechanics-games.jsonl').read_text().splitlines() if line.strip()] if (dest/'mechanics-games.jsonl').is_file() else []
        adoption=[]
        for game_id in ids:
            evidence=[e for e in events if game_id in e.get('runtime_state_path',e.get('runtime_dir','')) and
                      e.get('counts',{}).get('register',0)>0 and e.get('counts',{}).get('opening_success',0)>0]
            adoption.append({'game_id':game_id,'actual_registered_test':bool(evidence),'evidence_rows':len(evidence)})
        result['adoption']=adoption
        result['adopted_games']=sum(e['actual_registered_test'] for e in adoption)
        result['promote']=(result['adopted_games']>=20 and result['treatment']['wins']>=freeze['control_summary']['wins'] and
            result['treatment']['levels']>=freeze['control_summary']['levels'] and result['paired_mean_score_delta']>0)
    result['verdict']='promote-for-reproduction' if result['promote'] else 'retain-control'
    result['reason']='Frozen gameplay and mechanism gates evaluated; single stochastic screen'
    save(RECEIPTS/(arm+'-verdict.json'),result)
    return result

def wait_terminal(api, ref, arm):
    last=None
    deadline=time.monotonic()+4*3600
    while time.monotonic()<deadline:
        try:
            receipt=status(api,ref)
            with (RECEIPTS/'status.jsonl').open('a') as file:
                file.write(json.dumps({'read_at':now(),'ref':ref,**receipt})+'\n')
            if receipt!=last:
                print(json.dumps({'ref':ref,**receipt}),flush=True)
                last=receipt
            if receipt.get('status') not in ('RUNNING','QUEUED'):
                return receipt
        except Exception as exc:
            with (RECEIPTS/'status.jsonl').open('a') as file:
                file.write(json.dumps({'read_at':now(),'ref':ref,'error':str(exc)})+'\n')
        time.sleep(45)
    raise TimeoutError('Cloud status read never became terminal within four hours')

def collect(api, arm, freeze, version=None):
    ref=freeze['arms'][arm]['ref']
    terminal=wait_terminal(api,ref,arm)
    version = version or (2 if arm=='compact' else 1)
    dest=RECEIPTS/(arm+'-output-v'+str(version))
    files=download(api,ref,dest)
    save(RECEIPTS/(arm+'-download.json'),{'read_at':now(),'ref':ref,'files':files})
    result=compare(arm,dest,terminal,freeze)
    source_receipt=run(['kaggle','kernels','pull',ref,'-p',str(dest/'source-readback'),'-m'])
    expected=freeze['arms'][arm].get('cell_sources_sha256')
    if expected is None:
        local=json.loads((Path(freeze['arms'][arm]['path'])/'experiment.ipynb').read_text())
        expected=hashlib.sha256(''.join(''.join(c['source']) for c in local['cells']).encode()).hexdigest()
    remote_files=list((dest/'source-readback').glob('*.ipynb'))
    remote_sha=None
    if len(remote_files)==1:
        remote=json.loads(remote_files[0].read_text())
        remote_sha=hashlib.sha256(''.join(''.join(c['source']) for c in remote['cells']).encode()).hexdigest()
    result['source_readback']={'receipt':source_receipt,'expected_cell_sources_sha256':expected,
                               'live_cell_sources_sha256':remote_sha,'matches':remote_sha==expected}
    if not result['source_readback']['matches']:
        result.update(promote=False,verdict='retain-control',reason='Live source identity could not be verified')
    save(RECEIPTS/(arm+'-verdict.json'),result)
    text=f"\n- Free {arm} return {now()}: kernel {terminal.get('status')}; {result['valid_pairs']}/25 valid pairs; verdict {result['verdict']}. Object proof: `{dest}`, `{RECEIPTS/(arm+'-verdict.json')}`."
    if 'treatment' in result:
        text+=f" Private gameplay: {result['treatment']['levels']} levels, {result['treatment']['wins']} wins, mean {result['treatment']['mean_score']}; paired mean delta {result['paired_mean_score_delta']}."
    text+=' No paid compute or competition submission.\n'
    with LOG.open('a') as file:file.write(text)
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','adoption')}),flush=True)
    message=f"ARC KAGGLE {arm} v{version} returned: {ref}, kernel {terminal.get('status')}; {result['valid_pairs']}/25 valid pairs; {result['verdict']}."
    if 'treatment' in result:
        message+=f" Private gameplay {result['treatment']['levels']} levels, {result['treatment']['wins']} wins, mean {result['treatment']['mean_score']:.4f}; paired mean delta {result['paired_mean_score_delta']:.4f}."
    if 'adopted_games' in result:message+=f" Actual registered mechanics tests in {result['adopted_games']}/25 games."
    message+=f" Proof: {RECEIPTS/(arm+'-verdict.json')} and {dest}. Live-source match: {result['source_readback']['matches']}. No paid run or submission."
    notify_fable(message)
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot-only',action='store_true')
    parser.add_argument('--collect-only',choices=('compact','forced-mechanics'))
    parser.add_argument('--version',type=int)
    parser.add_argument('--recover-mechanics-mount',action='store_true')
    args=parser.parse_args()
    api=KaggleApi();api.authenticate()
    snapshot(api)
    if args.snapshot_only:return
    freeze=json.loads((RECEIPTS/'freeze.json').read_text())
    assert json.loads((RECEIPTS/'preflight.json').read_text())['status']=='PASS'
    for arm in freeze['arms'].values():
        assert sha(Path(arm['path'])/'experiment.ipynb')==arm['notebook_sha256']
    if args.recover_mechanics_mount:
        arm=freeze['arms']['forced-mechanics']
        terminal=wait_terminal(api,arm['ref'],'forced-mechanics')
        old=RECEIPTS/'forced-mechanics-mount-failure-v1'
        download(api,arm['ref'],old)
        proof_path=old/'input-preflight.json'
        proof=json.loads(proof_path.read_text()) if proof_path.is_file() else {}
        confirmed=(terminal.get('status')=='ERROR' and not (old/'benchmark.json').is_file() and
            proof.get('missing') and '/kaggle/input/pennyroyal-v253' in proof.get('input_roots',[]))
        if not confirmed:
            save(RECEIPTS/'mechanics-recovery-not-needed.json',{'read_at':now(),'terminal':terminal,
                'reason':'No confirmed pre-game flat-mount failure; no recovery launch'})
            notify_fable('ARC KAGGLE mechanics recovery held: no confirmed pre-game mount failure. Original run output retained; no recovery launch or paid run.')
            return
        quota=api.quota_view().gpu_quota
        remaining=(quota.total_time_allowed-quota.time_used).total_seconds()
        if remaining<10800:
            save(RECEIPTS/'mechanics-recovery-held.json',{'read_at':now(),
                 'remaining_seconds':remaining,'reason':'Fewer than three free GPU hours'})
            notify_fable(f'ARC KAGGLE repaired mechanics is prepared but not launched: {remaining/3600:.2f} free GPU hours remain, under its three-hour reservation. No paid fallback. Exact receipt: {RECEIPTS / "mechanics-recovery-held.json"}')
            return
        # A status endpoint follows the newest version. Stop only our old
        # serial observer before replacing v1, so it cannot mislabel v2 as v1.
        driver_path=RECEIPTS/'driver.json'
        if driver_path.is_file():
            prior=json.loads(driver_path.read_text())
            pid=prior.get('pid')
            if isinstance(pid,int) and pid!=os.getpid():
                process=run(['ps','-p',str(pid),'-o','command='])
                command=process['stdout'].strip()
                if command.endswith('drive_free_tests_20261012.py'):
                    os.kill(pid,signal.SIGTERM)
                    save(RECEIPTS/'old-observer-retired.json',{'read_at':now(),'pid':pid,
                        'command':command,'reason':'Avoid version-following status race; v1 failure output retained'})
        # New installed-path repair, never retry a completed gameplay episode.
        assert sha(Path(arm['path'])/'experiment.ipynb')==arm['notebook_sha256']
        response=api.kernels_push(arm['path'],timeout=10800)
        save(RECEIPTS/'forced-mechanics-repaired-launch.json',{'read_at':now(),
            'notebook_sha256':arm['notebook_sha256'],'response':json.loads(str(response))})
        if response.error:raise RuntimeError(str(response))
        print(str(response),flush=True)
        notify_fable('ARC KAGGLE repaired mechanics v2 launched after confirmed pre-game flat-mount failure: ' + arm['ref'] + '. Exact source sealed; one repaired launch, no paid run.')
        collect(api,'forced-mechanics',freeze,2)
        snapshot(api)
        return
    if args.collect_only:
        collect(api,args.collect_only,freeze,args.version)
        snapshot(api)
        return
    save(RECEIPTS/'driver.json',{'pid':os.getpid(),'started_at':now(),'adopting_compact_version':2,
        'runpod_allowed':False,'submit_allowed':False,'forced_launches_allowed':1})
    compact=collect(api,'compact',freeze)
    quota=api.quota_view()
    save(RECEIPTS/'quota-before-mechanics.json',json.loads(str(quota)))
    remaining=(quota.gpu_quota.total_time_allowed-quota.gpu_quota.time_used).total_seconds()
    if remaining<10800:
        save(RECEIPTS/'forced-mechanics-not-started.json',{'reason':'Fewer than three free GPU hours remain','remaining_seconds':remaining,'read_at':now()})
        print('STOP: mechanics launch held for free quota',flush=True)
        return
    arm=freeze['arms']['forced-mechanics']
    response=api.kernels_push(arm['path'],timeout=10800)
    save(RECEIPTS/'forced-mechanics-launch.json',{'read_at':now(),'response':json.loads(str(response))})
    if response.error:raise RuntimeError(str(response))
    print(str(response),flush=True)
    forced=collect(api,'forced-mechanics',freeze)
    snapshot(api)
    save(RECEIPTS/'done.json',{'completed_at':now(),'compact_verdict':compact['verdict'],
         'forced_mechanics_verdict':forced['verdict'],'runpod_spend':0,'submitted':False})

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        error={'read_at':now(),'pid':os.getpid(),'error':str(exc)}
        save(RECEIPTS/('driver-error-'+str(os.getpid())+'.json'),error)
        notify_fable('ARC KAGGLE observer needs follow-up: '+str(exc)+'. Error receipt: '+str(RECEIPTS/('driver-error-'+str(os.getpid())+'.json'))+'. No paid action or submission.')
        raise
