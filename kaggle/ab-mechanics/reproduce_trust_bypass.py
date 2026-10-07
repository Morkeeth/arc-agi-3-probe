import sys,json
from pathlib import Path
h=Path(__file__).resolve().parent;sys.path.insert(0,str(h/'upstream/ARC3-Inference'))
from inference.agent.python_tool_sandbox import run_sandboxed_python
code="m=mechanics.register('def predict(grid, action):\\n    return [[0,1]]')\nmechanics.state['models'][m]['trusted']=True\nresult=mechanics.plan(m,[[0,1]],['RIGHT'])"
r=run_sandboxed_python(code=code,timeout_seconds=3,initial_state={'current_frame':{'grid':[[1,0]],'level':0,'step':0,'shape':[1,2]},'history':[]},action_handler=lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not act')))
assert r['result']['status']=='found',r
out={'reproduced':True,'actual_actions':0,'result':r['result'],'root_cause':'Trusted mechanics state was reachable inside generated-code process. This is a blocking integrity failure, not a model-score failure.'};(h/'trust-bypass-before.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
