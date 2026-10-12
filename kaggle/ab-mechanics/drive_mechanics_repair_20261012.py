"""Prove identified calls on two model-played games, then run one free 25 pair.

The immutable stage seal is separate from the earlier mutable build manifest.
No Runpod or competition submit capability is used by this driver.
"""
import hashlib
import json
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import drive_free_tests_20261012 as driver
from free_tests_20261012 import RECEIPTS

ROOT=RECEIPTS/'mechanics-repair'

def source_match(ref, record, output):
    receipt=driver.run(['kaggle','kernels','pull',ref,'-p',str(output/'source-readback'),'-m'])
    files=list((output/'source-readback').glob('*.ipynb'))
    actual=None
    if len(files)==1:
        n=json.loads(files[0].read_text())
        actual=hashlib.sha256(''.join(''.join(c['source']) for c in n['cells']).encode()).hexdigest()
    proof={'read_at':driver.now(),'ref':ref,'expected':record['cell_sources_sha256'],
           'actual':actual,'matches':actual==record['cell_sources_sha256'],'receipt':receipt}
    driver.save(output/'source-identity.json',proof)
    return proof

def main():
    driver.RECEIPTS=ROOT
    api=KaggleApi();api.authenticate()
    freeze=json.loads((ROOT/'freeze.json').read_text())
    for record in (freeze['canary'],freeze['arms']['forced-mechanics']):
        assert driver.sha(Path(record['path'])/'experiment.ipynb')==record['notebook_sha256']
    driver.save(ROOT/'driver.json',{'pid':os.getpid(),'started_at':driver.now(),
        'canary_version':2,'full_launches_allowed':1,'paid_allowed':False,'submit_allowed':False})
    canary=freeze['canary']
    terminal=driver.wait_terminal(api,canary['ref'],'two-game-canary')
    output=ROOT/'canary-v2-output'
    driver.download(api,canary['ref'],output)
    identity=source_match(canary['ref'],canary,output)
    path=output/'two-game-mechanics-proof.json'
    proof=json.loads(path.read_text()) if path.is_file() else {}
    pass_gate=(terminal.get('status')=='COMPLETE' and identity['matches'] and
        proof.get('identified_test_games')==2 and
        {r['game_id'] for r in proof.get('games',[])}==set(canary['games']) and
        all(r['registered_calls']>0 and r['real_test_calls']>0 for r in proof.get('games',[])))
    driver.save(ROOT/'canary-verdict.json',{'read_at':driver.now(),'passed':pass_gate,
        'kernel':terminal,'source_identity':identity,'proof':proof})
    if not pass_gate:
        driver.notify_fable(f'ARC KAGGLE two-game canary did not pass; full 25-pair rerun held. Exact proof: {ROOT / "canary-verdict.json"}. No paid run or submission.')
        return
    driver.notify_fable(f'ARC KAGGLE two-game canary PASS: registered predictors and real mechanics tests attributed to both ls20 and sb26; live-source match. Proof: {ROOT / "canary-verdict.json"}. Preparing the authorized free 25-pair rerun, no paid run or competition submission.')
    quota=api.quota_view().gpu_quota
    remaining=(quota.total_time_allowed-quota.time_used).total_seconds()
    # The frozen Intel control took about 77 min including setup. Keep 2 h
    # free headroom; do not change the paired game's 7200 s/250/100k limits.
    driver.save(ROOT/'quota-before-full.json',{'read_at':driver.now(),
        'remaining_seconds':remaining,'reserved_seconds':quota.time_reserved.total_seconds(),
        'minimum_headroom_seconds':7200,'platform_cap_seconds':10800})
    if remaining<7200:
        driver.notify_fable(f'ARC KAGGLE two-game call proof passed, but full free rerun held: {remaining/3600:.2f} GPU hours remain, below 2 h headroom. No paid fallback. Proof: {ROOT / "quota-before-full.json"}.')
        return
    full=freeze['arms']['forced-mechanics']
    response=api.kernels_push(full['path'],timeout=10800)
    launch=json.loads(str(response))
    driver.save(ROOT/'full-launch.json',{'read_at':driver.now(),'notebook_sha256':full['notebook_sha256'],
        'cell_sources_sha256':full['cell_sources_sha256'],'response':launch})
    if response.error:raise RuntimeError(str(response))
    version=launch['versionNumber']
    driver.notify_fable(f'ARC KAGGLE repaired free 25-pair mechanics run launched as v{version}: {full["ref"]}. Immutable source sealed; actual per-game call receipts required in >=20 games. No paid run or submission.')
    result=driver.collect(api,'forced-mechanics',freeze,version)
    driver.snapshot(api)
    driver.save(ROOT/'returned.json',{'read_at':driver.now(),'version':version,
        'valid_pairs':result['valid_pairs'],'adopted_games':result.get('adopted_games'),
        'verdict':result['verdict'],'source_matches':result['source_readback']['matches'],
        'paid_spend':0,'submitted':False})

if __name__=='__main__':
    try:main()
    except Exception as exc:
        driver.save(ROOT/'driver-error.json',{'read_at':driver.now(),'error':str(exc)})
        driver.notify_fable(f'ARC KAGGLE repaired-run driver stopped: {exc}. Exact error: {ROOT / "driver-error.json"}. No paid action or competition submission.')
        raise
