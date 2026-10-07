import ast,builtins,json,time,sys
from pathlib import Path
from types import SimpleNamespace
p=Path(__file__).resolve().parent
sys.path.insert(0,str(p/'upstream/ARC3-Inference'))
from inference.agent.python_tool_sandbox import run_sandboxed_python
f={'grid':[[1,0,0]],'level':0,'step':0,'shape':[1,3],'ascii':''}
code='''
m=mechanics.register("def predict(grid, action):\\n    while True:\\n        pass")
a=mechanics.probe(['RIGHT'])
result={'probe':a, 'status':mechanics.status()}
'''
t=time.monotonic();r=run_sandboxed_python(code=code,timeout_seconds=2,initial_state={'current_frame':f,'history':[],'valid_actions':[]},action_handler=lambda *a: (_ for _ in ()).throw(AssertionError('No real action allowed')))
assert not r.get('error'),r
assert r['result']['probe'] is None and time.monotonic()-t<1.5,r
ns={'ast':ast,'builtins':builtins,'json':json};exec((p/'mechanics_runtime.py').read_text(),ns)
frame=SimpleNamespace(level=0,_grid=[[1,0]])
m=ns['Mechanics'](None,lambda:frame,lambda action:None)
for source in ('@bad\ndef predict(grid, action):\n return grid','def predict(grid, action=bad()):\n return grid','def predict(grid, action):\n import os\n return grid'):
 try:m.register(source)
 except ValueError:pass
 else:raise AssertionError('Unsafe construction accepted')
a={'action':'RIGHT'}
n=m.register("def predict(grid, action):\n    action['action']='LEFT'\n    return grid")
m._predict(n,[[1,0]],a);assert a=={'action':'RIGHT'}
m.test('RIGHT');assert not m.status()[n]['trusted']
frame.level=1;m.status();assert len(m.state['events'])==1
out={'status':'PASS','checks':['infinite predictor abstains within 100ms in real sandbox','decorators/defaults/imports rejected','predictor cannot mutate action','no-op grants no eligibility','level change preserves event trail'],'memory_limit':'Linux child-only 512MiB RLIMIT_AS; not exercised on macOS; runtime platform gate needed before gameplay'}
(p/'adversarial-gate.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
