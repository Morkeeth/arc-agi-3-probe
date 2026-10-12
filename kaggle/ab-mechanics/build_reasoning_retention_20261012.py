"""One-variable reasoning-retention rescue against immutable compact v3."""
import copy
import hashlib
import json
from pathlib import Path
from free_tests_20261012 import HERE, RECEIPTS, COMPACT_PATCH, sources

ROOT=RECEIPTS/'reasoning-retention'

def main():
    ROOT.mkdir(exist_ok=True)
    base_path=RECEIPTS/'compact-output-v3/source-readback/arc3-intel-compact-20261012.ipynb'
    base=json.loads(base_path.read_text())
    assert hashlib.sha256(''.join(sources(base)).encode()).hexdigest()=='ab7d5b8bdea7ebe5784e744cad7cb3712bb5edc201a473a59954afc10774cf0a'
    old="            text = _visible_text(message)\n            if text:\n                parts.append('Earlier model claim/plan (unverified): ' + text[:400])"
    new="            reasoning = message.get('reasoning_content') or message.get('reasoning')\n            text = ' '.join(str(reasoning).split())[-400:] if reasoning else _visible_text(message)[:400]\n            if text:\n                parts.append('Earlier model claim/plan (unverified): ' + text)"
    assert COMPACT_PATCH.count(old)==1
    patch=COMPACT_PATCH.replace(old,new,1)
    compile(patch,'reasoning_retention.py','exec')
    n=copy.deepcopy(base)
    setup=''.join(n['cells'][2]['source'])
    assert setup.count(repr(COMPACT_PATCH))==1
    n['cells'][2]['source']=setup.replace(repr(COMPACT_PATCH),repr(patch),1).splitlines(True)
    for cell in n['cells']:
        if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
    assert [i for i,(a,b) in enumerate(zip(sources(base),sources(n))) if a!=b]==[2]
    dest=HERE/'free-reasoning-retention-20261012'
    dest.mkdir(exist_ok=True)
    notebook=dest/'experiment.ipynb'
    notebook.write_text(json.dumps(n,indent=1)+'\n')
    metadata=json.loads((HERE/'free-compact-20261012/kernel-metadata.json').read_text())
    # Existing private notebook avoids the observed new-slug SaveKernel errors.
    metadata.update(id='oscarmorke/arc3-intel-compact-20261012',title='ARC3 Intel compact 20261012')
    (dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    bm_path=RECEIPTS/'compact-output-v3/benchmark.json'
    rows=json.loads(bm_path.read_text())['game_runs']
    control={'games':25,'wins':sum(r['state']=='won' for r in rows),'levels':sum(r['levels_completed'] for r in rows),
        'mean_score':sum(r['final_score'] for r in rows)/25,'actions':sum(len(r['history']) for r in rows)}
    seal={'sealed_before_launch':True,'source_base':str(base_path),'comparison_benchmark':str(bm_path),
        'comparison_benchmark_sha256':hashlib.sha256(bm_path.read_bytes()).hexdigest(),
        'control_summary':control,'games':[r['game_id'] for r in rows],
        'arms':{'compact':{'path':str(dest),'ref':metadata['id'],
            'notebook_sha256':hashlib.sha256(notebook.read_bytes()).hexdigest(),
            'cell_sources_sha256':hashlib.sha256(''.join(sources(n)).encode()).hexdigest()}},
        'question':'Does retaining the final 400 chars of dropped assistant reasoning_content rescue progress under the same 4000-char digest and two raw turn blocks?',
        'changed_cells':[2],'unchanged':'Model/server/game order/context window/2 recent blocks/4000 digest/200 code/700 tool/per-game 7200 seconds/250 actions/100000 tokens/concurrency 25',
        'rescue_gate':'25 valid pairs; wins not down and levels or paired score up against compact v3',
        'promotion_gate':'Also evaluate original Intel control; do not promote merely from beating compact v3',
        'platform_timeout_seconds':10800,'paid_allowed':False,'submission_allowed':False}
    (ROOT/'freeze.json').write_text(json.dumps(seal,indent=2)+'\n')
    print(json.dumps(seal,indent=2))

if __name__=='__main__':main()
