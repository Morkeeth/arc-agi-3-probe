"""Collect the sealed free rescue; report both baselines, never submit or spend."""
import copy
import json
import os
import hashlib
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import drive_free_tests_20261012 as d
from drive_mechanics_repair_20261012 import source_match
from free_tests_20261012 import RECEIPTS, CONTROL_OUTPUT, HERE

ROOT = RECEIPTS / 'reasoning-retention'


def prepare_returned_daily(api, intel):
    folder=HERE/'daily-reasoning-20261013'
    preparation_path=folder/'preparation.json'
    preparation=json.loads(preparation_path.read_text())
    preparation['promotion_passed']=intel['promote']
    preparation['private_25_pairs_complete']=intel['valid_pairs']==25
    quota=api.quota_view().gpu_quota
    remaining=(quota.total_time_allowed-quota.time_used).total_seconds()
    preparation['free_quota_seconds_before_gateway']=remaining
    if not intel['promote'] or remaining<1800:
        preparation['selection']='Held: original Intel promotion gate failed' if not intel['promote'] else 'Held: fewer than 1800 free GPU seconds remain for gateway preflight'
        d.save(preparation_path,preparation)
        d.notify_fable(f"Tomorrow ARC KAGGLE candidate prepared but not ready to submit: {preparation['selection']}. ARC-AGI-2 fallback candidate also built, output replay passes, full GPU validation pending. Exact preparation {preparation_path}. No paid fallback or submission.")
        return
    path=folder/'experiment.ipynb'
    assert d.sha(path)==preparation['candidate_sha256']
    notebook=json.loads(path.read_text())
    record={'cell_sources_sha256':hashlib.sha256(''.join(''.join(c['source']) for c in notebook['cells']).encode()).hexdigest()}
    ref=json.loads((folder/'kernel-metadata.json').read_text())['id']
    response=api.kernels_push(str(folder),timeout=1800)
    launch=json.loads(str(response));d.save(ROOT/'daily-gateway-launch.json',launch)
    if response.error:raise RuntimeError(str(response))
    d.notify_fable(f'ARC KAGGLE reasoning passed original Intel gate. Free exact daily-candidate gateway preflight launched: {ref} v{launch["versionNumber"]}, 1800-second cap. No competition submission.')
    terminal=d.wait_terminal(api,ref,'daily-gateway')
    output=ROOT/'daily-gateway-output';d.download(api,ref,output)
    identity=source_match(ref,record,output)
    proof_path=output/'forced-rerun-proof.json'
    proof=json.loads(proof_path.read_text()) if proof_path.is_file() else {}
    parquet=output/'submission.parquet'
    schema=d.run(['/Users/morkeeth/CODE/arc-agi-3-probe/.venv/bin/python','-c',
        "import pandas as pd,sys; p=pd.read_parquet(sys.argv[1]); assert list(p.columns)==['row_id','game_id','end_of_game','score']; assert len(p)>0; print(p.to_json(orient='records'))",str(parquet)]) if parquet.is_file() else {'returncode':1,'stderr':'missing parquet'}
    passed=(terminal.get('status')=='COMPLETE' and identity['matches'] and proof.get('actions',0)>0 and schema['returncode']==0)
    preparation.update(ready_to_submit=passed,ref=ref,version=launch['versionNumber'],
        actual_gateway_proof=proof,source_readback=identity,parquet_readback=schema,
        selection='Ready for tomorrow review; no submission made' if passed else 'Held: daily gateway completion/action/output/source gate failed',
        remaining_gates=[] if passed else ['Daily gateway preflight gate'])
    d.save(preparation_path,preparation);d.save(ROOT/'daily-gateway-verdict.json',preparation)
    d.notify_fable(f'Tomorrow ARC KAGGLE ready to submit: {passed}; {ref} v{launch["versionNumber"]}. Actual private gateway actions {proof.get("actions")}; kernel {terminal.get("status")}; source matched {identity["matches"]}; Parquet read-back {schema["returncode"]==0}. Object proof {output}. ARC-AGI-2 remains prepared, full GPU validation pending. No submission or paid run.')


def main():
    d.RECEIPTS = ROOT
    seal = json.loads((ROOT / 'freeze.json').read_text())
    record = seal['arms']['compact']
    assert d.sha(Path(record['path']) / 'experiment.ipynb') == record['notebook_sha256']
    assert d.sha(Path(seal['comparison_benchmark'])) == seal['comparison_benchmark_sha256']
    api = KaggleApi(); api.authenticate()
    d.save(ROOT / 'driver.json', {'pid': os.getpid(), 'started_at': d.now(),
        'version': 4, 'paid_allowed': False, 'submit_allowed': False})
    terminal = d.wait_terminal(api, record['ref'], 'reasoning-retention')
    output = ROOT / 'output-v4'
    d.download(api, record['ref'], output)
    identity = source_match(record['ref'], record, output)
    d.CONTROL_OUTPUT = Path(seal['comparison_benchmark'])
    rescue = d.compare('compact', output, terminal, seal)
    rescue['source_readback'] = identity
    rescue['rescue_passed'] = bool(rescue['promote'] and identity['matches'])
    rescue['promote'] = False
    rescue['verdict'] = 'rescue-supported' if rescue['rescue_passed'] else 'rescue-not-supported'
    d.save(ROOT / 'against-compact-v3.json', rescue)
    original = copy.deepcopy(seal)
    rows = json.loads(CONTROL_OUTPUT.read_text())['game_runs']
    original['control_summary'] = {'games': 25, 'wins': sum(r['state'] == 'won' for r in rows),
        'levels': sum(r['levels_completed'] for r in rows),
        'mean_score': sum(r['final_score'] for r in rows) / 25,
        'actions': sum(len(r['history']) for r in rows)}
    d.CONTROL_OUTPUT = CONTROL_OUTPUT
    intel = d.compare('compact', output, terminal, original)
    intel['source_readback'] = identity
    intel['promote'] = bool(intel['promote'] and identity['matches'])
    intel['verdict'] = 'promote-for-reproduction' if intel['promote'] else 'retain-intel-control'
    d.save(ROOT / 'against-intel-control.json', intel)
    # Do not leave the generic helper's intermediate file as an ambiguous verdict.
    d.save(ROOT / 'compact-verdict.json', {'use': 'see both explicit comparisons',
        'rescue': str(ROOT / 'against-compact-v3.json'),
        'promotion': str(ROOT / 'against-intel-control.json')})
    result = {'returned_at': d.now(), 'kernel': terminal, 'source_matches': identity['matches'],
        'valid_pairs': intel['valid_pairs'], 'rescue_passed': rescue['rescue_passed'],
        'promotion_passed': intel['promote'], 'paid_spend': 0, 'submitted': False}
    d.save(ROOT / 'returned.json', result)
    message = f"ARC KAGGLE reasoning retention v4 returned {terminal.get('status')}: {intel['valid_pairs']}/25 valid pairs; source match {identity['matches']}."
    if 'treatment' in intel:
        t = intel['treatment']
        message += f" {t['levels']} levels, {t['wins']} wins, mean {t['mean_score']:.4f}; paired delta vs compact v3 {rescue['paired_mean_score_delta']:+.4f}, vs Intel {intel['paired_mean_score_delta']:+.4f}. Rescue {rescue['rescue_passed']}; promotion {intel['promote']}."
    message += f" Proof: {ROOT / 'against-compact-v3.json'}, {ROOT / 'against-intel-control.json'}, {output}. No paid run or submission."
    d.notify_fable(message)
    with d.LOG.open('a') as f:
        f.write('\n- ' + d.now() + ': ' + message + '\n')
    d.snapshot(api)
    prepare_returned_daily(api,intel)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        d.save(ROOT / 'driver-error.json', {'read_at': d.now(), 'error': str(exc)})
        d.notify_fable(f'ARC KAGGLE reasoning collector stopped: {exc}. Proof {ROOT / "driver-error.json"}; no paid action or submit.')
        raise
