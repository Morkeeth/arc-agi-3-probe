"""Retain trace evidence and correct the v1 attribution/source report."""
import copy
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from free_tests_20261012 import HERE, RECEIPTS

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    root=RECEIPTS/'mechanics-repair'
    path=RECEIPTS/'forced-mechanics-verdict.json'
    old=json.loads(path.read_text())
    (root/'original-v1-report.json').write_text(json.dumps(old,indent=2)+'\n')
    original=subprocess.run(['git','show','33d664a:kaggle/ab-mechanics/free-forced-mechanics-20261012/experiment.ipynb'],
        cwd=HERE.parents[1],capture_output=True,text=True,check=True)
    notebook=json.loads(original.stdout)
    sealed=hashlib.sha256(''.join(''.join(c['source']) for c in notebook['cells']).encode()).hexdigest()
    actual=old['source_readback']['live_cell_sources_sha256']
    assert actual==sealed
    call_path=RECEIPTS/'forced-mechanics-output-v1/mechanics-calls.jsonl'
    calls=[json.loads(line) for line in call_path.read_text().splitlines() if line.strip()]
    report=copy.deepcopy(old)
    report['original_checker_comparison']=report['source_readback']
    report['source_readback']={**report['source_readback'],'expected_cell_sources_sha256':sealed,
        'matches':True,'seal_object':'git 33d664a, launched v1; not the later repaired build'}
    report['call_aggregate']={'methods':dict(Counter(c['method'] for c in calls)),
        'real_test_calls':sum(c['method']=='test' and any(a.get('executed') for a in c.get('real_action_results',[])) for c in calls),
        'object':str(call_path),'sha256':sha(call_path)}
    for row in report['adoption']:
        row['actual_registered_test']=None
        row['reason']='Game identity absent from shared-directory call receipts; unknown, not zero'
    report['reason']='Source v1 identity verified. Distinct-game adoption unknown; private wins/levels/score screen does not pass the promotion rule.'
    report['report_correction']='The previous source checker compared live v1 to the later repair. Actual recorded calls exist, but were not attributable to games.'
    path.write_text(json.dumps(report,indent=2)+'\n')
    evidence={'compact_source':str(RECEIPTS/'compact-output-v3/source-readback/arc3-intel-compact-20261012.ipynb'),
        'mechanics_v1_source_match':actual==sealed,'mechanics_call_aggregate':report['call_aggregate'],
        'removed_fields':{'user':'Earlier image/frame prompts skipped from digest',
            'assistant':'Reads content only; reasoning_content is absent from digest',
            'tool_calls':'At most 3 code fragments, 200 characters each',
            'tool_results':'First 700 characters per response',
            'rolling_record':'Last 4000 characters; reset on frame level change'},
        'sb26_trace':{},'benchmarks':{}}
    traces=RECEIPTS/'compact-output-v3/sb26-7fbdac44_p0_requests.jsonl'
    rows=[json.loads(line) for line in traces.read_text().splitlines() if line.strip()]
    evidence['sb26_trace']['object']=str(traces)
    evidence['sb26_trace']['sha256']=sha(traces)
    evidence['sb26_trace']['line_9_reasoning']=[m.get('reasoning_content') for m in rows[8]['messages']
        if m.get('role')=='assistant' and m.get('reasoning_content') and 'selection' in m['reasoning_content']]
    evidence['sb26_trace']['line_25_reasoning']=[m.get('reasoning_content') for m in rows[24]['messages']
        if m.get('role')=='assistant' and m.get('reasoning_content')][-2:]
    for label,path in (('control',RECEIPTS.parent/'private-validation-output/benchmark.json'),
                       ('compact',RECEIPTS/'compact-output-v3/benchmark.json')):
        benchmark=json.loads(path.read_text())
        evidence['benchmarks'][label]={'object':str(path),'sha256':sha(path),
            'levels':sum(g['levels_completed'] for g in benchmark['game_runs']),
            'wins':sum(g['state']=='won' for g in benchmark['game_runs']),
            'actions':sum(len(g['history']) for g in benchmark['game_runs']),
            'action_associated_generated_tokens':sum(sum(a.get('generated_tokens',0) for a in g['history']) for g in benchmark['game_runs']),
            'sb26':[{k:g[k] for k in ('game_id','levels_completed','state','final_score','solver_note')}
                     for g in benchmark['game_runs'] if g['game_id']=='sb26-7fbdac44']}
    (root/'trace-audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'mechanics_calls':report['call_aggregate'],'source_matches_launched_v1':True,
        'compact_trace_evidence':str(root/'trace-audit.json')},indent=2))

if __name__=='__main__':main()
