import json,datetime
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
p=Path(__file__).resolve().parent
a=KaggleApi();a.authenticate();out={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'datasets':{},'models':{}}
for ref in ('dfranzen/pennyroyal-v253','dfranzen/taaf-kaggle-source-bundle-copy'):
 rows=[];token=None
 while True:
  r=a.dataset_list_files(ref,page_token=token,page_size=1000).to_dict();rows+=r.get('datasetFiles',r.get('files',[]));token=r.get('nextPageToken')
  if not token:break
 out['datasets'][ref]={'files':rows,'metadata':a.dataset_list(search=ref.split('/')[-1],user=ref.split('/')[0])[0].to_dict()}
for ref in ('dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default','dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default'):
 out['models'][ref]=a.model_instance_versions_list(ref).to_dict()
(p/'runtime-inputs.json').write_text(json.dumps(out,indent=2,default=str))
print({k:len(v['files']) for k,v in out['datasets'].items()})
