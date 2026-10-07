import ast,hashlib,json
from pathlib import Path
h=Path(__file__).resolve().parent
checks=[]
embedded_files=('mechanics_runtime.py','install_policy.py')
manifest=json.loads((h/'pair-package.json').read_text())
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
 setup='\n'.join('pass # IPython shell command' if line.startswith('!') else line for line in ''.join(n['cells'][2]['source']).splitlines())
 embedded={}
 for node in ast.walk(ast.parse(setup)):
  if not (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text' and len(node.args)==1):continue
  path=node.func.value
  if not (isinstance(path,ast.BinOp) and isinstance(path.op,ast.Div) and isinstance(path.left,ast.Name) and path.left.id=='policy_dir' and isinstance(path.right,ast.Constant)):continue
  name=path.right.value
  if name in embedded_files:
   assert name not in embedded, f'{arm}: duplicate embedded {name}'
   embedded[name]=ast.literal_eval(node.args[0])
 assert set(embedded)==set(embedded_files), f'{arm}: missing embedded policy file'
 for name in embedded_files:
  assert embedded[name].encode()==(h/name).read_bytes(), f'{arm}: embedded {name} differs from current source'
 assert hashlib.sha256((h/arm/'experiment.ipynb').read_bytes()).hexdigest()==manifest['arms'][arm]['notebook_sha256'], f'{arm}: manifest hash mismatch'
 checks.append({'arm':arm,'code_cells':len(n['cells']),'sha256':hashlib.sha256((h/arm/'experiment.ipynb').read_bytes()).hexdigest()})
a=json.loads((h/'control/experiment.ipynb').read_text());b=json.loads((h/'treatment/experiment.ipynb').read_text())
different=[i for i,(x,y) in enumerate(zip(a['cells'],b['cells'])) if x!=y]
assert different==[0,2,10],different
metadata=[json.loads((h/arm/'kernel-metadata.json').read_text()) for arm in ('control','treatment')]
assert {k:v for k,v in metadata[0].items() if k not in ('id','title')}=={k:v for k,v in metadata[1].items() if k not in ('id','title')}, 'Runtime inputs or platform differ'
local=json.loads((h/'local-gate.json').read_text())
assert local['status']=='PASS' and local.get('live_sdk_frame_reads',0)>0, 'Live SDK callback integration missing'
for name in (*embedded_files,'local_gate.py'):
 assert local.get('source_sha256',{}).get(name)==hashlib.sha256((h/name).read_bytes()).hexdigest(), f'Local SDK result is stale for {name}'
out={'status':'PASS','notebooks':checks,'embedded_source_sha256':{name:hashlib.sha256((h/name).read_bytes()).hexdigest() for name in embedded_files},'different_cells':different,'difference_scope':'arm receipt, installer mode (both arms parent-only action reporting; treatment adds parent-owned mechanics and prompt), completion receipt. Budgets and mounts identical.','linux_cpu_gate':'Prior recorded Docker CPU gate; not rerun during package reconciliation.','local_sdk_gate':{'actual_sdk_transitions':local['actual_sdk_transitions'],'live_sdk_frame_reads':local['live_sdk_frame_reads'],'observation_path':local['observation_path'],'scope':local['scope']}}
(h/'package-gate.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
