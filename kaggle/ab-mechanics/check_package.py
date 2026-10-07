import ast,hashlib,json
from pathlib import Path
h=Path(__file__).resolve().parent
checks=[]
for arm in ('control','treatment'):
 n=json.loads((h/arm/'experiment.ipynb').read_text());m=json.loads((h/arm/'kernel-metadata.json').read_text())
 assert m['is_private'] and not m['enable_internet']
 assert all(x.endswith('/1') for x in m['dataset_sources']+m['model_sources'])
 for i,c in enumerate(n['cells']):
  src=''.join(c['source'])
  if src.startswith('%%writefile'):
   assert src.splitlines()[0]=='%%writefile /kaggle/harness-changes.patch';continue
  src='\n'.join('pass # IPython shell command' if s.startswith('!') else s for s in src.splitlines())
  compile(src,f'{arm}-cell{i}','exec',flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
 assert 'submission.parquet' not in ''.join(n['cells'][-1]['source'])
 checks.append({'arm':arm,'code_cells':len(n['cells']),'sha256':hashlib.sha256((h/arm/'experiment.ipynb').read_bytes()).hexdigest()})
a=json.loads((h/'control/experiment.ipynb').read_text());b=json.loads((h/'treatment/experiment.ipynb').read_text())
different=[i for i,(x,y) in enumerate(zip(a['cells'],b['cells'])) if x!=y]
assert different==[0,2,10],different
out={'status':'PASS','notebooks':checks,'different_cells':different,'difference_scope':'arm receipt, treatment installation+prompt only, completion receipt. Budgets and runtime mounts identical.','linux_cpu_gate':'Docker python3.12-slim --network none --memory768m: memory allocation refusal, infinite-loop timeout, valid predictor all PASS','local_sdk_gate':'six actual ls20 development transitions including two-step plan; lookup fixture, no generalization claim'}
(h/'package-gate.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
