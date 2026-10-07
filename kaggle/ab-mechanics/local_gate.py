"""Exercise the actual SDK and installed sandbox; no model API or game source reads."""
import ast,builtins,json,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
from arc_agi import Arcade,OperationMode
from arcengine import GameAction
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'upstream/ARC3-Inference'))
from inference.agent.python_tool_sandbox import run_sandboxed_python
arc=Arcade(operation_mode=OperationMode.OFFLINE,environments_dir='/Users/morkeeth/CODE/arc-agi-3-probe-data/environment_files')
ids=sorted([e.game_id for e in arc.get_environments()],key=lambda s:hashlib.sha256(('mechanics-v1:'+s).encode()).hexdigest())
split={'method':'sha256(mechanics-v1:game_id), frozen before policy evaluation; all public games; not hidden competition evidence','development':ids[:10],'untuned':ids[10:]}
(HERE/'game-split.json').write_text(json.dumps(split,indent=2))
g=arc.make(ids[0],seed=0)
def frame():
    o=g.observation_space
    return {'grid':o.frame[-1].tolist(),'level':o.levels_completed,'step':len(events),'shape':[64,64],'ascii':''}
events=[]
# Development-only calibration, never claim learning/generalization from this lookup fixture.
training=[]
for a in [1,2,3,4]*3:
    if a not in g.observation_space.available_actions: continue
    before=frame()['grid']; out=g.step(getattr(GameAction,"ACTION"+str(a))); after=frame()['grid']
    training.append([before,'ACTION'+str(a),after])
    if out.levels_completed or str(out.state).endswith('GAME_OVER'): break
g=arc.make(ids[0],seed=0)
encoded = [[''.join('0123456789abcdef'[v] for row in b for v in row),a,''.join('0123456789abcdef'[v] for row in c for v in row)] for b,a,c in training]
source='def predict(grid, action):\n    rows = '+repr(encoded)+'\n    key = "".join("0123456789abcdef"[v] for row in grid for v in row)\n    for before, act, after in rows:\n        if key == before and action == act:\n            return [[int(v,16) for v in after[i:i+64]] for i in range(0,4096,64)]\n    return None\n'

def handler(actions,**kwargs):
    assert len(actions)==1
    a=actions[0]['action'];before=frame()
    o=g.step(getattr(GameAction,a))
    events.append({'before':before,'action':a,'after':frame()})
    return {'action_result':{'actions_executed':1},'state':{'current_frame':frame(),'history':[],'valid_actions':[]}}
code='m=mechanics.register('+repr(source)+')\nreports=[]\n'
for row in training[:4]: code+='reports.append(mechanics.test('+repr(row[1])+'))\n'
code+='plan=mechanics.plan(m,'+repr(training[5][2])+',["ACTION1","ACTION2","ACTION3","ACTION4"])\n'
code+='executed=mechanics.execute(m,plan["actions"])\nresult={"reports":reports,"status":mechanics.status(),"plan":plan,"executed":executed}\n'
r=run_sandboxed_python(code=code,timeout_seconds=20,initial_state={'current_frame':frame(),'history':[],'valid_actions':[]},action_handler=handler)
assert not r.get('error'),r
assert len(events)>4 and r["result"]["plan"]["status"]=="found" and r["result"]["executed"],r.get("result")
assert all(x['status']=='observed' for x in r['result']['reports']),r
assert all(x['models']['m0']=='match' for x in r['result']['reports']),r['result']
# Pure negative gates use the SAME policy class, isolated from game score claims.
ns={'ast':ast,'builtins':builtins,'json':json};exec((HERE/'mechanics_runtime.py').read_text(),ns)
f=SimpleNamespace(level=0,_grid=[[1,0,0,0,0]])
calls=[]
def advance(a):
    calls.append(a); row=f._grid[0];i=row.index(1);row[i]=0;row[min(i+1,4)]=1
m=ns['Mechanics'](None,lambda:f,advance)
model=m.register('def predict(grid, action):\n    row=grid[0]\n    i=row.index(1)\n    row[i]=0\n    row[min(i+1,4)]=1\n    return grid')
assert m.plan(model,[[0,0,1,0,0]],['RIGHT'])['status']=='untrusted'
m.test('RIGHT');m.test('RIGHT');assert m.status()[model]['trusted']
plan=m.plan(model,[[0,0,0,0,1]],['RIGHT']);assert len(plan['actions'])==2
assert len(m.execute(model,plan['actions']))==2
f.level=1;assert m.status()=={}
# A confidently wrong model loses trust on first mismatch, without a second action.
f.level=2;f._grid=[[1,0,0,0,0]];m._scope();model=m.register('def predict(grid, action):\n    return [[0,0,0,0,1]]')
m.state['models'][model]['trusted']=True
n=len(calls);m.execute(model,['RIGHT','RIGHT']);assert len(calls)==n+1 and not m.status()[model]['trusted']
report={'status':'PASS','game_id':ids[0],'actual_sdk_transitions':len(events),'scope':'Deterministic replay of development-game transitions validates plumbing and prospective comparison, NOT learned mechanics/generalization or score gain. Synthetic policy gates exercise planner, trust and mismatch stop.','sandbox_result':r,'actual_events':events,'negative_gates':['untrusted planning refused','bounded BFS found target','stepwise plan execution','level invalidates trust','mismatch stops before next action']}
(HERE/'local-gate.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('sandbox_result','actual_events')}))
