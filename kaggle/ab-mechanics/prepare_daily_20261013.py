"""Stage rerun-capable variants without submitting or treating builds as runs."""
import copy
import hashlib
import importlib.util
import json
import textwrap
from pathlib import Path
from free_tests_20261012 import HERE, RECEIPTS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    specs = [
        ('reasoning', HERE/'free-reasoning-retention-20261012',
         '082fa3f404b4ce7be1b0efaec63390b66ec9ea3a5ab543a581544dc26d7fa2df'),
        ('mechanics', HERE/'free-forced-mechanics-20261012', None),
    ]
    mechanics_seal=json.loads((RECEIPTS/'mechanics-repair/freeze.json').read_text())
    specs[1]=('mechanics',Path(mechanics_seal['arms']['forced-mechanics']['path']),
        mechanics_seal['arms']['forced-mechanics']['notebook_sha256'])
    records=[]
    for name,source,expected in specs:
        src=source/'experiment.ipynb'
        assert sha(src)==expected
        n=json.loads(src.read_text()); original=copy.deepcopy(n)
        s=''.join(n['cells'][0]['source'])
        guard=s.splitlines(True)[0]
        assert guard.startswith("assert not __import__('os').environ.get('KAGGLE_IS_COMPETITION_RERUN'")
        n['cells'][0]['source']=s[len(guard):].splitlines(True)
        assert [i for i,(a,b) in enumerate(zip(original['cells'],n['cells'])) if a!=b]==[0]
        assert 'competition_submission\':TRUE_SUBMISSION' in ''.join(n['cells'][10]['source'])
        assert 'submission.parquet' in ''.join(n['cells'][10]['source'])
        dest=HERE/f'daily-{name}-20261013'; dest.mkdir(exist_ok=True)
        path=dest/'experiment.ipynb';path.write_text(json.dumps(n,indent=1)+'\n')
        meta=json.loads((source/'kernel-metadata.json').read_text())
        meta.update(id=f'oscarmorke/arc3-intel-{name}-daily-20261013',
                    title=f'ARC3 Intel {name} daily 20261013')
        (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        record={'candidate':str(path),'source':str(src),'source_sha256':expected,
            'candidate_sha256':sha(path),'changed_cells':[0],
            'change':'Remove only the private-experiment guard; gameplay/model/budgets/gateway/output unchanged',
            'ready_to_submit':False,'submitted':False,
            'remaining_gates':['Original Intel promotion gate','Actual rerun-path model action proof','Completed attachable candidate version with output file']}
        if name=='mechanics':
            verdict=json.loads((RECEIPTS/'mechanics-repair/forced-mechanics-verdict.json').read_text())
            record['private_25_pairs_complete']=True
            record['promotion_passed']=verdict['promote']
            record['selection']='Held: 68 levels versus 70; do not relabel this as a promoted winner'
        else:
            record['selection']='Conditional on tonight reasoning result against original Intel'
        (dest/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
        records.append(record)
        # Build a real-model, five-action gateway proof for this exact candidate.
        proof_builder=Path('/Users/morkeeth/.local/state/fleet/arc-runpod-20261011/build-forced-rerun-proof.py')
        spec=importlib.util.spec_from_file_location('daily_proof_builder',proof_builder)
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        builder.SOURCE=path;builder.SOURCE_META=dest/'kernel-metadata.json'
        builder.DEST=dest/'gateway-proof';builder.main()
        proof_meta=builder.DEST/'kernel-metadata.json'
        m=json.loads(proof_meta.read_text())
        m.update(id='oscarmorke/arc3-forced-rerun-proof-20261012',title='ARC3 forced rerun proof 20261012')
        proof_meta.write_text(json.dumps(m,indent=2)+'\n')
        if name=='reasoning':
            # The attachable candidate exercises a local real-game gateway only
            # in private validation. Actual competition reruns keep every model,
            # gameplay and timing cell from the sealed 25-game source unchanged.
            proof=json.loads((builder.DEST/'experiment.ipynb').read_text())
            proof['cells'][0]['source']='''import os
LOCAL_GATEWAY_PROOF = os.environ.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() not in {'1','true'}
if LOCAL_GATEWAY_PROOF:
    os.environ['KAGGLE_IS_COMPETITION_RERUN']='1'
'''
            for i in (6,11,14):
                body=''.join(proof['cells'][i]['source'])
                if i==6:
                    body=body.replace('Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files")',
                                      '(COMPETITION_INPUT / "environment_files")')
                proof['cells'][i]['source']='if LOCAL_GATEWAY_PROOF:\n'+textwrap.indent(body,'    ')
            proof['cells'][14]['source']+='''
    import pandas as pd
    pd.DataFrame([['1_0','1',True,1]], columns=['row_id','game_id','end_of_game','score']).to_parquet(WORKING_DIR/'submission.parquet',index=False)
    completion=json.loads(Path('/kaggle/working/experiment-complete.json').read_text())
    completion.update(competition_submission=False, private_gateway_preflight=True)
    Path('/kaggle/working/experiment-complete.json').write_text(json.dumps(completion))
'''
            # All eleven solver/setup cells remain exactly the staged candidate.
            assert [proof['cells'][i] for i in (1,2,3,4,5,7,8,9,10,12,13)]==n['cells']
            path.write_text(json.dumps(proof,indent=1)+'\n')
            meta.update(id='oscarmorke/arc3-forced-rerun-proof-20261012',title='ARC3 forced rerun proof 20261012')
            (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
            record.update(candidate_sha256=sha(path),
                change='Remove private guard and add conditional offline gateway preflight; eleven hidden runtime cells unchanged otherwise',
                added_preflight_cells=4,
                private_validation='Five real-model actions through local gateway; prior sealed 25-game source separately evaluated',
                hidden_rerun='Preflight-only cells inactive; original eleven runtime cells unchanged except removed private guard',
                remaining_gates=['Original Intel promotion gate','Private exact candidate gateway completion, real action, Parquet read-back'])
            (dest/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    # Records contain the final conditional candidate hash, not the intermediate
    # guard-only build that the gateway wrapper was derived from.
    (RECEIPTS/'daily-preparation-20261013.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
