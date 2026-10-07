"""Build a private, offline-only pair. This script never launches a kernel."""
import copy,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
BASE=Path('/Users/morkeeth/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian LIFE/01 Projects/Hackathons/research/arc-competitive-20261007/milestone2')
raw=(BASE/'arc-agi-3-milestone-2-solution.ipynb').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='88a379f17c3a4ba8c98715fc23155346ccd6eb289bdafebadfc61a3178f9d9bd'
original=json.loads(raw);metadata=json.loads((BASE/'kernel-metadata.json').read_text())
inputs=json.loads((H/'runtime-inputs.json').read_text());split=json.loads((H/'game-split.json').read_text())
files={name:(H/name).read_text() for name in ('mechanics_runtime.py','install_policy.py')}
def cell(source):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':source.splitlines(True)}
manifest={'base_notebook_sha256':hashlib.sha256(raw).hexdigest(),'platform_timeout_seconds':10800,'kernels':2,'maximum_free_gpu_hours':6,'launch_authorized_by_root_only':True,'retries':0,'submission':False,'arms':{},'evaluation':{'games':split,'passes':1,'per_game_wall_seconds':7200,'per_game_actions':250,'per_game_generated_tokens':100000,'token_cap_semantics':'baseline stops at next solver boundary; one response may exceed threshold by max_output12288; actual usage must be reported','soft_end_seconds_from_notebook_start':9600,'interpretation':'whole-policy screen, public untuned games; not hidden score, not isolated memory or planner causality','kill_bar':'No eligible plans or no heldout paired level/action-efficiency benefit at matched resource use kills current policy. Regressions, unequal usage, startup failure or truncation invalidate a superiority claim. One seed is exploratory, not significance.'}}
for arm in ('control','treatment'):
 n=copy.deepcopy(original)
 # Retain executable baseline only; no markdown generated and no diagnostics after run.
 n['cells']=[copy.deepcopy(original['cells'][i]) for i in (2,4,6,8,10,12,14,16,18,20)]
 for c in n['cells']:c['outputs']=[];c['execution_count']=None
 # Prevent any competition rerun path; all resource timing starts before installations.
 pre="import os,time,json,hashlib,sys,resource\nfrom pathlib import Path\nassert not os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').lower() in ('1','true'), 'Private offline experiment only'\nEXPERIMENT_START=time.time()\n"
 # Snapshot structural mounts before consuming them. Dataset versions are pinned in metadata.
 pre+="EXPECTED_INPUTS="+repr({k:v['files'] for k,v in inputs['datasets'].items()})+"\n"
 pre+="for ref,rows in EXPECTED_INPUTS.items():\n    root=Path('/kaggle/input/datasets')/ref\n    for row in rows:\n        path=root/row['name']\n        assert path.is_file() and path.stat().st_size==row.get('totalBytes',0), str(path)\n"
 pre+="BUNDLE_HASHES="+repr(json.loads((H/'bundle-file-hashes.json').read_text()))+"\nfor rel,digest in BUNDLE_HASHES.items():\n    assert hashlib.sha256((Path('/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy')/rel).read_bytes()).hexdigest()==digest, rel\n"
 pre+="Path('/kaggle/working/experiment-start.json').write_text(json.dumps({'arm':"+repr(arm)+",'epoch':EXPERIMENT_START,'private_offline':True}))\n"
 n['cells'].insert(0,cell(pre))
 setup=''.join(n['cells'][2]['source'])
 setup=setup.replace('NOTEBOOK_START_TIME = time.time()','NOTEBOOK_START_TIME = EXPERIMENT_START').replace('NOTEBOOK_START_EPOCH = time.time()','NOTEBOOK_START_EPOCH = EXPERIMENT_START')
 # Always verify pinned source bytes after the public patch, before inference imports.
 setup+="\nassert not TRUE_SUBMISSION\n"
 setup+="BASE_HASHES="+repr(json.loads((H/'baseline-source-hashes.json').read_text()))+"\nfor rel,digest in BASE_HASHES.items():\n    assert hashlib.sha256((BUNDLE_DIR/'src/ARC3-Inference/inference'/rel).read_bytes()).hexdigest()==digest, rel\n"
 gate=(H/'mechanics_runtime.py').read_text()+"\nimport ast,builtins,json\nfrom types import SimpleNamespace\nf=SimpleNamespace(level=0,_grid=[[1,0]])\nm=Mechanics(None,lambda:f,lambda a:None)\nn=m.register('def predict(grid, action):\\n    return [0]*1000000000')\nassert m._predict(n,[[1,0]],'RIGHT') is None\nn=m.register('def predict(grid, action):\\n    while True: pass')\nassert m._predict(n,[[1,0]],'RIGHT') is None\nn=m.register('def predict(grid, action):\\n    return grid')\nassert m._predict(n,[[1,0]],'RIGHT') == [[1,0]]\nprint('MEMORY_TIME_GATE_PASS')\n"
 (H/"linux_gate.py").write_text(gate)
 setup+="subprocess.run([sys.executable,'-I','-S','-c',"+repr(gate)+"],check=True,timeout=5)\n"
 if arm=='treatment':
  setup+="policy_dir=Path('/kaggle/mechanics-policy');policy_dir.mkdir(exist_ok=True)\n"
  for name,content in files.items():setup+=f"(policy_dir/{name!r}).write_text({content!r})\n"
  setup+="subprocess.run([sys.executable,str(policy_dir/'install_policy.py'),str(BUNDLE_DIR/'src/ARC3-Inference')],check=True,timeout=30)\n"
 n['cells'][2]['source']=setup.splitlines(True)
 # Fail startup before spending gameplay budget if serving is still unavailable.
 startup=''.join(n['cells'][6]['source'])
 startup+="\nwith urlopen(f'http://127.0.0.1:{SERVED_MODEL_PORT}/health',timeout=10) as response:\n    assert response.status==200, 'Startup did not complete; no gameplay'\n"
 n['cells'][6]['source']=startup.splitlines(True)
 cfg="assert not TRUE_SUBMISSION\nassert hasattr(bm.solver,'max_generated_tokens_per_game') or 'max_generated_tokens_per_game' in getattr(type(bm.solver),'__dataclass_fields__',{})\nbm.solver.max_runtime_s_per_game=7200\nbm.solver.analyzer_timeout=900.0\nbm.solver.concurrency=25\nbm.solver.max_actions_per_game=250\nbm.solver.max_generated_tokens_per_game=100000\nbm.solver.save_request_logs=True\nbm.n_passes=1\nbm.game_weights=None\ndemo_excluded_games=[]\ntarget.max_runtime_s=10200\n"
 n['cells'][8]['source']=cfg.splitlines(True)
 run=''.join(n['cells'][10]['source'])
 run=run[:run.index('if not TRUE_SUBMISSION:\n    # An offline run isn')]
 run=run.replace('bm.games = _offline_games(competition_env_files)',"bm.games = _offline_games(competition_env_files)\n    expected="+repr(split['development']+split['untuned'])+"\n    assert sorted(g.env_name for g in bm.games)==sorted(expected), 'Game identity mismatch'\n    bm.games.sort(key=lambda g:expected.index(g.env_name))")
 run+="\nPath('/kaggle/working/experiment-complete.json').write_text(json.dumps({'arm':"+repr(arm)+",'elapsed':time.time()-EXPERIMENT_START,'competition_submission':False}))\n"
 n['cells'][10]['source']=run.splitlines(True)
 d=H/arm;d.mkdir(exist_ok=True);file=d/'experiment.ipynb';file.write_text(json.dumps(n,indent=1))
 m=copy.deepcopy(metadata);m.update(id='oscarmorke/arc-mechanics-'+arm+'-v1',title='ARC mechanics '+arm+' v1',code_file='experiment.ipynb',is_private=True)
 m.pop('id_no',None);m['dataset_sources']=[ref+'/1' for ref in inputs['datasets']]
 (d/'kernel-metadata.json').write_text(json.dumps(m,indent=2))
 manifest['arms'][arm]={'path':str(d),'notebook_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'root_launch_command':'kaggle kernels push -p '+str(d)+' -t 10800'}
(H/'pair-package.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest['arms'],indent=2))
