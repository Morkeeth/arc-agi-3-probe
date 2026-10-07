import subprocess,json,time,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
pending={a:'oscarmorke/arc-mechanics-'+a+'-v2' for a in ('control','treatment')}
start=time.monotonic()
while pending and time.monotonic()-start<28800:
 for arm,ref in list(pending.items()):
  r=subprocess.run(['kaggle','kernels','status',ref],capture_output=True,text=True,timeout=45)
  row={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arm':arm,'ref':ref,'returncode':r.returncode,'output':r.stdout+r.stderr}
  (ROOT/(arm+'-current.json')).write_text(json.dumps(row,indent=2))
  with (ROOT/'status.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if r.returncode==0 and any('KernelWorkerStatus.'+v in r.stdout for v in ['COMPLETE','ERROR','CANCELLED']):
   out=ROOT/(arm+'-output');out.mkdir(exist_ok=True)
   pulled=subprocess.run(['kaggle','kernels','output',ref,'-p',str(out)],capture_output=True,text=True,timeout=180)
   (ROOT/(arm+'-pull.log')).write_text(pulled.stdout+pulled.stderr)
   row['meaning']='Terminal platform status only; serving, gameplay and comparison require output inspection.'
   row['download_returncode']=pulled.returncode;(ROOT/(arm+'-terminal.json')).write_text(json.dumps(row,indent=2));del pending[arm]
 if pending:time.sleep(60)
(ROOT/'monitor-finished.json').write_text(json.dumps({'pending':pending,'no_retry':True,'elapsed_s':time.monotonic()-start}))
