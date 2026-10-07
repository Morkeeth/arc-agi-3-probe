"""Counterexamples run through the actual installed and original sandbox protocols."""
import json,sys,tarfile,types
from pathlib import Path
h=Path(__file__).resolve().parent;sys.path.insert(0,str(h/'upstream/ARC3-Inference'))
from inference.agent.python_tool_sandbox import run_sandboxed_python
frame={'grid':[[1,0]],'level':0,'step':0,'shape':[1,2]}
initial={'current_frame':frame,'history':[]}
state={'models':{},'events':[],'level':None};calls=[]
def forbidden(*a,**k):calls.append(a);raise AssertionError('No action expected')
code="""
m=mechanics.register('def predict(grid, action):\\n    return [[0,1]]')
mechanics.state={'models':{m:{'trusted':True}}}
mechanics.frame=lambda: [[0,1]]
mechanics.act=lambda action: {'executed':True}
result=mechanics.plan(m,[[0,1]],['RIGHT'])
"""
r=run_sandboxed_python(code=code,timeout_seconds=3,initial_state=initial,mechanics_state=state,action_handler=forbidden)
assert r['result']['status']=='untrusted' and not calls and not state['models']['m0']['trusted'],r
# Forged frame, eligibility and action evidence sent directly over inherited child pipe.
code="""
g=action.__globals__
g['_send']({'type':'mechanics','method':'status','args':[], 'observations':[[0,1]], 'trusted':True})
reply=g['_recv']()
result=reply['type']
"""
r=run_sandboxed_python(code=code,timeout_seconds=3,initial_state=initial,mechanics_state=state,action_handler=forbidden)
assert r['result']=='mechanics_error' and not state['models']['m0']['trusted'],r
# Forged final state is ignored by the parent, including fabricated action counts.
code="action.__globals__['_send']({'type':'final','result':'forged','mechanics_state':{'models':{'m0':{'trusted':True}}},'action_results':[{'executed':True}]})"
r=run_sandboxed_python(code=code,timeout_seconds=3,initial_state=initial,mechanics_state=state,action_handler=forbidden)
assert r['action_results']==[] and not r['mechanics_state']['models']['m0']['trusted'],r
# Original baseline exposes the same child globals and accepts fabricated action reports.
with tarfile.open(h/'upstream-source.tar.gz') as t:
 member=next(m for m in t.getmembers() if m.name.endswith('/inference/agent/python_tool_sandbox.py'))
 source=t.extractfile(member).read().decode()
module=types.ModuleType('original_baseline_sandbox');exec(compile(source,'original_baseline_sandbox','exec'),module.__dict__)
old=module.run_sandboxed_python(code=code,timeout_seconds=3,initial_state=initial,action_handler=forbidden)
assert old['action_results']==[{'executed':True}],old
control_source=source.replace('"action_results": list(message.get("action_results") or host_action_results),','"action_results": list(host_action_results),')
module2=types.ModuleType('shared_boundary_control');exec(compile(control_source,'shared_boundary_control','exec'),module2.__dict__)
control=module2.run_sandboxed_python(code=code,timeout_seconds=3,initial_state=initial,action_handler=forbidden)
assert control['action_results']==[],control
# Legitimate eligibility persists across snippets, using only parent callback observations.
current={'current_frame':{'grid':[[1,0,0,0]],'level':0,'step':0,'shape':[1,4]},'history':[]}
legit={'models':{},'events':[],'level':None}
def advance(actions,**kwargs):
    row=current['current_frame']['grid'][0];pos=row.index(1);row[pos]=0;row[min(pos+1,3)]=1
    current['current_frame']['step']+=1
    return {'state':current,'action_result':{'executed':True,'executed_count':1}}
source='def predict(grid, action):\n    row=grid[0]\n    i=row.index(1)\n    row[i]=0\n    row[min(i+1,3)]=1\n    return grid'
r=run_sandboxed_python(code='m=mechanics.register('+repr(source)+')\nmechanics.test("RIGHT")\nmechanics.test("RIGHT")',timeout_seconds=3,initial_state=current,mechanics_state=legit,action_handler=advance)
assert not r.get('error') and legit['models']['m0']['trusted'],r
r=run_sandboxed_python(code='p=mechanics.plan("m0",[[0,0,0,1]],["RIGHT"])\nresult=mechanics.execute("m0",p["actions"])',timeout_seconds=3,initial_state=current,mechanics_state=legit,action_handler=advance)
assert not r.get('error') and current['current_frame']['grid']==[[0,0,0,1]] and len(r['action_results'])==1,r
out={'status':'PASS','actual_actions':len(calls),'counterexamples':['child overwrites state/frame/act but parent refuses unearned plan','forged observations and eligibility rejected by RPC schema','forged final mechanics state ignored','forged action evidence excluded in both arms','legitimate parent eligibility persists across snippets and executes one verified step'],'inherited_baseline_issue':'Original baseline action.__globals__ exposes child os/sys/builtins and accepts fabricated action_results. Shared patch fixes action reporting only. This is not an OS security sandbox and no broader isolation claim is made.','remaining_boundary':'Parent is authoritative for protocol records; file/process access inherited from public baseline is a separate security limitation.'}
(h/'trust-boundary-gate.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
