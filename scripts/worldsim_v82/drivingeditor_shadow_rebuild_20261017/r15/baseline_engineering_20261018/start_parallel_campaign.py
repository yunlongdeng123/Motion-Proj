from pathlib import Path
import subprocess,os,json
E=Path(__file__).resolve().parent
def main():
 assert not (E/'parallel_campaign_launch.json').exists()
 with (E/'parallel_campaign_controller.log').open('x') as f:
  p=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python','-B','-u',str(E/'run_parallel_campaign.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (E/'parallel_campaign_launch.json').write_text(json.dumps(dict(pid=p.pid),indent=2));print(p.pid)
if __name__=='__main__':main()
