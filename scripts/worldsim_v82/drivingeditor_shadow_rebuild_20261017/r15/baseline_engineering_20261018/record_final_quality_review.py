"""Record the agent's completed visual inspection; no inferred visual acceptance."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
import numpy as np
from PIL import Image
W=Path(__file__).parent;O=W.parents[1]/'outputs/baseline_engineering_20261018'

def main():
 manifest=json.loads((W/'final_visual_review_manifest.json').read_text())
 assert len(manifest['sheets'])==26
 frames=O/'final_quality_frames';frames.mkdir(exist_ok=True)
 examples=[]
 for model,root,ch,label in [
  ('A','direct_full49','gt','共同输入RGB'),('A','direct_full49','full','A完整重建'),
  ('A','direct_full49','dynamic','A五身份动态集合'),('A','direct_full49','static','A静态仍有近车形与亮白彩色条带'),
  ('A','direct_full49','delete_all','A删除五身份后仍有静态车形'),('A','direct_full49','a06_isolated','A a6独立资产下方白色带状碎片'),
  ('A','direct_full49','a37_isolated','A a37近处银色轿车独立资产'),('A','direct_full49','a37_delete','A删除a37后静态车形仍在'),
  ('B','all_vehicle_full49','dynamic','B纳入20身份的动态集合'),('B','all_vehicle_full49','static','B静态车形与停车区碎片'),
  ('B','all_vehicle_full49','delete_all','B删除纳入集合后仍有残留')]:
  for raw in [128,129]:
   src=W/root/f'0008{raw:03d}'/(ch+'.png');name=f'{model}_raw{raw}_{ch}.png';dst=frames/name
   with Image.open(src) as im:assert im.size==(480,320)
   shutil.copy2(src,dst)
   examples.append(dict(path='final_quality_frames/'+name,label=label,raw=raw,whole_frame=True,sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
 decoder=[]
 for model,root,ch,file in [('A','direct_full49','static','matched5_static.mp4'),('B','all_vehicle_full49','static','allvehicles_static.mp4'),('A','direct_full49','a06_isolated','matched5_a06_isolated.mp4')]:
  decoded=W/f'final_decoded_{model}_{ch}_raw128.png'
  subprocess.run(['ffmpeg','-v','error','-y','-i',str(O/'branch_videos'/file),'-vf','select=eq(n\\,39)','-frames:v','1',str(decoded)],check=True)
  with Image.open(decoded) as im:actual=np.asarray(im.convert('RGB'),dtype=np.float32);assert im.size==(480,320)
  with Image.open(W/root/'0008128'/(ch+'.png')) as im:source=np.asarray(im.convert('RGB'),dtype=np.float32)
  mae=float(np.abs(actual-source).mean());assert mae<8,(file,mae)
  decoder.append(dict(model=model,channel=ch,raw=128,actual_frame_index=39,MAE_RGB_255=mae,whole_frame=True))
 models={
  'A 五身份 GN11 30K＋实例30K':dict(status_label='训练与删除技术合同完成；干净背景和资产质量未通过',observations='全49帧静态及delete-all仍有自车道白／青／彩色车形、条带与暗色足迹。a37近处银色车的独立车身连续，但删除后静态重复车形仍在。a6前方银色车独立输出下方持续出现平面白色带状碎片。a5与a26可见车身连续，删除后仍有暗色投影阴影；远处a18像素小，不能由训练视图判高质量。'),
  'B 交集20身份 GN11 第一阶段30K':dict(status_label='训练与集合删除技术合同完成；干净背景未通过',observations='全49帧动态集合包含纳入的20车辆，但static及delete-all仍有自车道亮白／青色车形、彩色条带与路面暗块，左侧停车区有拉伸碎片或车形残影。新增纳入身份没有得到干净街景；13条相机独有轨迹仍是上下文，不将其合法保留一概当错删或漏删。')}
 review=dict(status='COMPLETED_ACTUAL_FULLFRAME_QUALITY_REVIEW',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  summary='A五身份两阶段30K＋30K、B交集20身份第一阶段30K及最终49帧零更新删除评估已完成；两项均未达到干净街景目标。A／B静态分支和删除图仍保留明显车形、亮白／彩色条带；A的a6独立资产有白色带状碎片。不能把训练完成、低损失或目标alpha清零当作分离质量通过。',
  models=models,quality_accepted=False,human_verdict=None,training_updates=0,
  actors={
   'a6':'前方银色车常被更近的a37遮挡；全49帧独立资产下方有明显平面白带／碎片。删除a6保留a37是正确范围，不能将a37叫作a6。',
   'a37':'最近银色轿车独立车身可辨且连续；逐ID删除后强烈白／青色静态车形仍保留，干净删除失败。',
   'a5':'蓝色对向车可见段连续，约raw110–113后出左画面，随后独立图黑色符合离开视野；删除车身后暗色阴影仍在。',
   'a26':'较晚靠近的深蓝／灰对向车由远变近，raw131–137大幅出左边界；删除车身后仍有暗色阴影，边缘截断符合相机范围。',
   'a18':'远处浅白对向车全49帧较小，逐渐变大；删除影响小受像素尺度与遮挡限制，不据此批准高质量三维资产。'},
  method=dict(reviewer='Codex root actual image inspection',whole_frame=True,ROI=False,source_size=[480,320],
   viewed_sheets=[dict(name=Path(s['path']).name,sha256=s['sha256']) for s in manifest['sheets']],
   coverage='A/B各49帧GT/full/dynamic/static/delete-all；A五身份各49帧isolated/delete；raw89/107/128/129/137全部身份锚帧；另看原生raw128 A static/dynamic及B static',
   contact_sheet_source_pixels_resized=False,actual_browser_UI_verified=False,decoder_samples=decoder),
  limits=['共同49帧均为训练观测，无独立留出测试。','始终被挡住的道路没有配对真值，本次不添加生成补全或窗外训练。','车身mask不覆盖所有阴影；原生外观删除不重算投影阴影。','原生静态运动／衰减正则问题与道路观测不足分别有历史证据；未完成具体残留Gaussian因果归因，不分配原因百分比。'],
  examples=examples)
 for base in [W,O]:(base/'final_quality_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(dict(status=review['status'],viewed_sheets=26,quality_accepted=False,examples=len(examples),decoder=decoder),ensure_ascii=False))
if __name__=='__main__':main()
