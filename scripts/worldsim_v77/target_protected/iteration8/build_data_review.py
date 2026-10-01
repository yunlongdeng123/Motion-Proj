"""所有候选保留，数据技术评分独立于模型效果与人工评分。"""
from pathlib import Path
import sys,json,shutil
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,read,dump

def page(path,title,intro,data):
    template=(Path(__file__).parent/'review_template.html').read_text()
    template=template.replace('__TITLE__',title).replace('__INTRO__',intro).replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
    path.write_text(template)

def main():
    out=O/'review';out.mkdir(exist_ok=True)
    manifest=read(O/'data_review/synthetic_manifest.json');qa=read(O/'independent_data_reviews.json') if (O/'independent_data_reviews.json').exists() else {'cases':[]};byid={r['case_id']:r for r in qa['cases']}
    catalog=read(O/'dataset_catalog.json') if (O/'dataset_catalog.json').exists() else {'cases':[]};admitted={c['case_id']:c for c in catalog['cases']}
    for role in ['assets','contacts']:
        src=O/'data_review'/role;dest=out/('data_'+role)
        if not dest.exists():dest.symlink_to(src,target_is_directory=True)
    cases=[]
    for c in manifest['clips']:
        r=byid.get(c['case_id'],{});a=admitted.get(c['case_id']);status=a['split'] if a else '未准入／QA备用'
        cases.append({'id':c['case_id'],'scene':c['scene'],'type':c['type'],'group':status,'note':f'{c["camera"]}；红色为人工A，绿线为真实保护实例；灰车RGB只供轮廓诊断，完整车身在条件编码前擦除。{status}。','qa':f'独立AI技术分 {r.get("assistant_score","待审")}：{r.get("reason","")}；人工分未填写。','review_frame':c['review_frame'],'videos':{k:'data_'+v for k,v in c['videos'].items()},'frame_pattern':f'data_assets/{c["case_id"]}/{{i}}_{{role}}.jpg','metrics':{'type':c['type'],'scene':c['scene'],'source':c['source_id'],'asset':c['asset'],'offset_m':[c['offset_longitudinal_m'],c['offset_lateral_m']],'protected_instances':c['protected_instances'],'occlusion_fraction':c['occlusion_fraction'],'edge_mode':c['edge_mode'],'dilation_px':c['mask_dilation_px'],'pixel_contract':c['pixel_metrics'],'human_verdict':None}})
    data={'mode':'data','roles':[{'key':'gt','label':'真实Y：恢复GT'},{'key':'input','label':'人工X：仅诊断'},{'key':'labels','label':'A／真实保护B mask'},{'key':'condition','label':'模型实际遮洞条件'}],'score_roles':[{'key':'data','label':'训练输入质量'}],'cases':cases}
    s=catalog.get('summary',{});intro=f'<p>r8 数据覆盖控制。实际训练 {s.get("training_cases","待准入")} 例／{s.get("training_scene_count","待准入")} 场景。候选 {len(cases)} 例全部保留；每例10帧可逐帧评分。</p><p>尺寸、贴地、碰撞、depth/order、时序及完整擦除做全帧机器检查；独立AI只检查指定的一帧。外观假不单独拒绝，但空间错误、mask泄漏、错遮挡必须拒绝。AI2技术准入不等于模型效果2。两种网格来自已曝光DEV，train/val共享轮廓模板；不声称未见形状泛化。</p>'
    page(out/'data_review.html','v77 r8 · 训练数据逐帧审核',intro,data);dump(out/'data_manifest.json',data)
    print('DATA_HTML',len(cases),flush=True)
if __name__=='__main__':main()
