"""汇总实际合成/工程检查/独立QA；保留分批来源与全部失败，不重渲染。"""
import argparse
from pathlib import Path
from collections import Counter
from geometry_factory import read,dump

def main(root,out):
    out.mkdir(parents=True,exist_ok=True);(out/'assets').mkdir(exist_ok=True);(out/'contacts').mkdir(exist_ok=True)
    cohorts=[('P / 30帧初始固定世界位移',root/'synthetic_review'),('D / 30帧相机相对轨迹',root/'expanded_factory/synthetic_review'),('W / 官方10帧固定窗口',root/'native_window_factory/synthetic_review')]
    rows=[];reviews=[];validation=[];origins=[]
    for label,folder in cohorts:
        if not (folder/'synthetic_manifest.json').exists():continue
        batch=read(folder/'synthetic_manifest.json')['clips'];qa=read(folder/'independent_synthetic_reviews.json');val=read(folder/'delivery_validation.json')
        assert {r['case_id'] for r in batch}=={r['case_id'] for r in qa['clips']}=={r['case_id'] for r in val['cases']},'QA/validation coverage'
        origins.append({'cohort':label,'review_root':str(folder),'reviewer':{k:v for k,v in qa.items() if k!='clips'}})
        for r in batch:
            cid=r['case_id'];assert cid not in {x['case_id'] for x in rows}
            rows.append(r|{'cohort':label})
            dst=out/'assets'/cid
            if not dst.exists():dst.symlink_to(folder/'assets'/cid,target_is_directory=True)
            for contact in r['contacts']:
                dst=out/contact
                if not dst.exists():dst.symlink_to(folder/contact)
        reviews+=qa['clips'];validation+=val['cases']
    dump(out/'synthetic_manifest.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','cohorts':origins,'clips':rows})
    dump(out/'independent_synthetic_reviews.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','reviewer_model':'gpt-6-sol','reasoning_effort':'xhigh','fast':False,'cohorts':origins,'clips':reviews})
    dump(out/'delivery_validation.json',{'case_count':len(rows),'video_count':len(rows)*4,'preview_frame_count':sum(r['frames'] for r in validation)*4,'cases':validation})
    print('COLLECTED',len(rows),Counter(r['synthetic_status'] for r in reviews),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
