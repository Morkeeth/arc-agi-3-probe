"""Seal an immutable two-game model canary and repaired 25-game treatment."""
import copy
import hashlib
import json
from pathlib import Path
from free_tests_20261012 import HERE, RECEIPTS, sources

ROOT=RECEIPTS/'mechanics-repair'
GAMES=['ls20-9607627b','sb26-7fbdac44']

def write(dest, notebook, ref, title):
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/'experiment.ipynb'
    path.write_text(json.dumps(notebook,indent=1)+'\n')
    metadata=json.loads((HERE/'free-forced-mechanics-20261012/kernel-metadata.json').read_text())
    metadata.update(id=ref,title=title)
    (dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return {'path':str(dest),'ref':ref,'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'cell_sources_sha256':hashlib.sha256(''.join(sources(notebook)).encode()).hexdigest()}

def main():
    ROOT.mkdir(exist_ok=True)
    frozen=json.loads((RECEIPTS/'freeze.json').read_text())
    full=json.loads((HERE/'free-forced-mechanics-20261012/experiment.ipynb').read_text())
    full_record=write(HERE/'free-mechanics-repaired-20261012',full,
        'oscarmorke/arc3-intel-forced-mechanics-20261012','ARC3 Intel forced-mechanics 20261012')
    frozen['arms']={'forced-mechanics':full_record}
    frozen['launch_policy']='Free two-game canary must return 2/2 identified register+real test records and live-source match before one 25-game launch. No paid run or competition submission.'
    canary=copy.deepcopy(full)
    config=''.join(canary['cells'][8]['source'])
    config=config.replace('max_runtime_s_per_game=7200','max_runtime_s_per_game=600')
    config=config.replace('analyzer_timeout=900.0','analyzer_timeout=300.0')
    config=config.replace('concurrency=25','concurrency=2')
    config=config.replace('max_actions_per_game=250','max_actions_per_game=5')
    config=config.replace('max_generated_tokens_per_game=100000','max_generated_tokens_per_game=30000')
    config=config.replace('target.max_runtime_s=10200','target.max_runtime_s=2400')
    canary['cells'][8]['source']=config.splitlines(True)
    play=''.join(canary['cells'][10]['source'])
    original='    expected='+repr(frozen['games'])
    assert play.count(original)==1
    play=play.replace(original,"    bm.games=[g for g in bm.games if g.env_name in "+repr(GAMES)+"]\n    expected="+repr(GAMES))
    proof='''
calls=[json.loads(line) for line in Path('/kaggle/working/mechanics-calls.jsonl').read_text().splitlines() if line.strip()]
games=CANARY_GAMES
rows=[]
for game_id in games:
    registered=[r for r in calls if game_id in str(r.get('runtime_state_path','')) and r['method']=='register']
    tested=[r for r in calls if game_id in str(r.get('runtime_state_path','')) and r['method']=='test'
            and any(a.get('executed') for a in r.get('real_action_results',[]))]
    rows.append({'game_id':game_id,'registered_calls':len(registered),'real_test_calls':len(tested),
                 'test_evidence':tested})
proof={'games':rows,'identified_test_games':sum(r['registered_calls']>0 and r['real_test_calls']>0 for r in rows),
       'benchmark_games':[{'game_id':r.game_id,'state':r.state,'actions':len(r.history)} for r in bm.game_runs],
       'scope':'Two-game model/Sdk call-attribution canary, not a score-improvement test'}
Path('/kaggle/working/two-game-mechanics-proof.json').write_text(json.dumps(proof,indent=2))
assert proof['identified_test_games']==2, proof
assert len(bm.game_runs)==2 and all(r.state in ('won','gave_up') and len(r.history)>0 for r in bm.game_runs)
print('TWO_GAME_MECHANICS_PASS='+json.dumps(proof))
'''.replace('CANARY_GAMES',repr(GAMES))
    canary['cells'][10]['source']=(play+proof).splitlines(True)
    canary_record=write(HERE/'free-mechanics-canary-20261012',canary,
        'oscarmorke/arc3-intel-forced-mechanics-20261012','ARC3 Intel forced-mechanics 20261012')
    canary_record.update(games=GAMES,platform_timeout_seconds=3000)
    frozen['canary']=canary_record
    frozen['sealed_before_launch']=True
    (ROOT/'freeze.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps({'canary':canary_record,'full':full_record},indent=2))

if __name__=='__main__':main()
