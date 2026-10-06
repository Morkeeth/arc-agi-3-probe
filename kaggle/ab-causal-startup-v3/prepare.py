"""Prepare, never launch, a pair with bounded cold-start headroom.
Original v2 source and notebooks remain unchanged.
"""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'ab-causal-pilot'
PATCHES={'SERVER_READY_TIMEOUT = 1_500':'SERVER_READY_TIMEOUT = 2_400','SETUP_TOTAL_TIMEOUT = 1_800':'SETUP_TOTAL_TIMEOUT = 2_700'}

def patch_source(source,expected_sha):
    if hashlib.sha256(source.encode()).hexdigest()!=expected_sha:
        raise ValueError('setup source differs from pinned bytes')
    for before,after in PATCHES.items():
        if source.count(before)!=1:raise ValueError('ambiguous timeout source')
        source=source.replace(before,after)
    return source

def prepare(source_path):
    source=source_path.read_text(); expected=hashlib.sha256(source.encode()).hexdigest()
    patched=patch_source(source,expected)
    wrapper='''import hashlib,json
from pathlib import Path
original=Path(__import__('os').environ['TAAF_KAGGLE_BUNDLE_DIR'])/'serving_setup.py'
source=original.read_text()
assert hashlib.sha256(source.encode()).hexdigest()==EXPECTED, 'source drift'
for before,after in PATCHES.items():
    assert source.count(before)==1, 'ambiguous patch'
    source=source.replace(before,after)
Path('/kaggle/working/startup_patch_receipt.json').write_text(json.dumps({'original_sha256':EXPECTED,'executed_sha256':hashlib.sha256(source.encode()).hexdigest(),'server_ready_timeout':2400,'setup_total_timeout':2700,'session_timeout_required':8400},sort_keys=True))
exec(compile(source,str(original),'exec'),{'__name__':'__main__','__file__':str(original)})
'''.replace('EXPECTED',repr(expected)).replace('PATCHES',repr(PATCHES))
    records={}
    for arm in ['control','treatment']:
        meta=json.loads((ORIGINAL/arm/'kernel-metadata.json').read_text())
        n=json.loads((ORIGINAL/arm/meta['code_file']).read_text())
        needle='    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)'
        setup=''.join(n['cells'][9]['source']);assert setup.count(needle)==1
        replacement=f'''    if command == '\"$PYTHON\" \"$TAAF_KAGGLE_BUNDLE_DIR/serving_setup.py\"':
        subprocess.run([sys.executable, '-c', {wrapper!r}], check=True, cwd=WORKING_DIR, env=env)
    else:
{needle.replace('    subprocess','        subprocess')}'''
        n['cells'][9]['source']=setup.replace(needle,replacement).splitlines(keepends=True)
        slug=f'arc-causal-{arm}-startup-v3';meta.update(id='oscarmorke/'+slug,title=slug,code_file=slug+'.ipynb')
        dest=HERE/arm;dest.mkdir(exist_ok=True)
        (dest/meta['code_file']).write_text(json.dumps(n,indent=1)+'\n')
        (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        records[arm]={'notebook_sha256':hashlib.sha256((dest/meta['code_file']).read_bytes()).hexdigest(),'changed_cells':[9],'base':str(ORIGINAL/arm),'launch_authorized':False}
    record={'source_sha256':expected,'patched_source_sha256':hashlib.sha256(patched.encode()).hexdigest(),'patches':PATCHES,'arms':records,'session_timeout_required':8400,'meaning':'Cold-start headroom candidate. Not validated on GPU. Original failure unchanged.'}
    (HERE/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    return record

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    print(json.dumps(prepare(p.parse_args().source),indent=2))
