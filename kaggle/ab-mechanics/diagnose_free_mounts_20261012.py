"""Build a CPU-only mount read, preserving the experiment's exact attachments."""
import json
from pathlib import Path
from free_tests_20261012 import BASE, HERE

dest = HERE / 'free-mount-probe-20261012'
dest.mkdir(exist_ok=True)
code = '''from pathlib import Path
import hashlib,json
rows=[]
for path in Path('/kaggle/input').rglob('dependencies.lock'):
    rows.append({'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
roots=[str(p) for p in Path('/kaggle/input').glob('*')]
bundle_files=[str(p) for p in Path('/kaggle/input').rglob('taaf-kaggle-bundle.json')]
proof={'input_roots':roots,'locks':rows,'bundle_files':bundle_files}
Path('/kaggle/working/mount-proof.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof))
'''
n={'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':code}],
   'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
(dest/'experiment.ipynb').write_text(json.dumps(n,indent=1))
m=json.loads((BASE/'kernel-metadata.json').read_text())
m.update(id='oscarmorke/arc3-free-mount-probe-20261012',title='ARC3 free mount probe 20261012',enable_gpu=False)
m.pop('machine_shape',None)
(dest/'kernel-metadata.json').write_text(json.dumps(m,indent=2))
print(dest)
