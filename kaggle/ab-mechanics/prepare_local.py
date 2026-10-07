"""Rehydrate the pinned public source and install the policy for CPU checks."""
from pathlib import Path
import hashlib,json,tarfile
from install_policy import install
h=Path(__file__).resolve().parent
archive=h/'upstream-source.tar.gz'
pin=json.loads((h/'source-pin.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest()==pin['archive_sha256']
with tarfile.open(archive) as t:
 for member in t.getmembers():
  if '/ARC3-Inference/' not in member.name or not member.isfile():continue
  rel=Path('ARC3-Inference')/member.name.split('/ARC3-Inference/',1)[1]
  if '..' in rel.parts:raise ValueError('Invalid archived path')
  if rel.suffix not in ('.py','.json','.toml','.lock'):continue
  path=h/'upstream'/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(t.extractfile(member).read())
print(json.dumps(install(h/'upstream/ARC3-Inference'),indent=2))
