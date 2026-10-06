import ast,hashlib,importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'kaggle/ab-causal-startup-v3'
spec=importlib.util.spec_from_file_location('startup',P/'prepare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_source_drift_refused():
    with pytest.raises(ValueError,match='differs'):m.patch_source('changed','0'*64)

def test_only_two_bounded_constants_change():
    source='SERVER_READY_TIMEOUT = 1_500\nSETUP_TOTAL_TIMEOUT = 1_800\nSOMETHING_ELSE = 123\n'
    result=m.patch_source(source,hashlib.sha256(source.encode()).hexdigest())
    scope={};exec(result,scope)
    assert (scope['SERVER_READY_TIMEOUT'],scope['SETUP_TOTAL_TIMEOUT'],scope['SOMETHING_ELSE'])==(2400,2700,123)

@pytest.mark.parametrize('arm',['control','treatment'])
def test_candidate_preserves_policy_and_gameplay_cells(arm):
    orig=ROOT/'kaggle/ab-causal-pilot'/arm
    oldmeta=json.loads((orig/'kernel-metadata.json').read_text());old=json.loads((orig/oldmeta['code_file']).read_text())
    newmeta=json.loads((P/arm/'kernel-metadata.json').read_text());new=json.loads((P/arm/newmeta['code_file']).read_text())
    assert len(old['cells'])==len(new['cells'])
    assert [i for i,(a,b) in enumerate(zip(old['cells'],new['cells'])) if a!=b]==[9]
    src=''.join(new['cells'][9]['source']);tree=ast.parse(src)
    wrappers=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and "startup_patch_receipt.json" in n.value]
    assert len(wrappers)==1
    ast.parse(wrappers[0])
    assert 'source drift' in wrappers[0]
    for k in oldmeta:
        if k not in ['id','title','code_file']:assert newmeta[k]==oldmeta[k]
