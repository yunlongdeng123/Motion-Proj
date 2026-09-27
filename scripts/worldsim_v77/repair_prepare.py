import datetime,subprocess,shutil
from PIL import Image,ImageDraw
from repair_common import *
assert not (ROOT/'registration.json').exists()
ROOT.mkdir(parents=True,exist_ok=True)
p0=read(P0/'registration.json');scenes=[]
for name,aid,cam,start in [('scene_0230','22',5,18),('scene_0255','25',3,65),('official_000','12',0,0)]:
 spec=next(s for s in p0['scenes'] if s['name']==name);data=pathlib.Path(spec['root']);inst=read(data/'instances/instances_info.json');out=ROOT/name
 for sub in ['rgb','sam','mask','raw_video']:(out/sub).mkdir(parents=True,exist_ok=True)
 ids=list(range(start,start+30));allids=sorted(int(p.stem.split('_')[0]) for p in (data/'images').glob('*_0.jpg'))
 fa=inst[aid]['frame_annotations'];donors=sorted(set(list(range(0,max(allids)+1,10))+[max(allids)]))
 # 只使用目标身份存在的时间点，GT结束不能当作车辆消失。
 donors=[f for f in donors if f in fa['frame_idx']]
 frames={str(f):scene_frame(spec,f,inst) for f in sorted(set(ids+donors))}
 for i,f in enumerate(ids):
  im=Image.open(data/'images'/f'{f:03}_{cam}.jpg').convert('RGB').resize((1024,576),Image.Resampling.BILINEAR)
  im.save(out/'rgb'/f'{i:05}.png');im.save(out/'raw_video'/f'{i:05}.jpg',quality=96)
  if name!='official_000':shutil.copy2(FULL/name/f'cam{cam}/target_sam/{f:05}.png',out/'sam'/f'{i:05}.png')
 s={'name':name,'actor':aid,'camera':cam,'source_frames':ids,'donor_frames':donors,'spec':spec,'track_id':inst[aid]['id'],'data':str(data)};scenes.append(s);dump(out/'frames.json',frames)
 sheet=Image.new('RGB',(1024,576*3))
 for n,i in enumerate([0,15,29]):
  f=ids[i];fr=frames[str(f)];b=next(b for b in fr['all_boxes'] if b['actor_id']==aid);c,k=camera(fr,cam);rect=project_bbox(b['pose'],b['size_lwh'],c,k,(576,1024))
  im=Image.open(out/'rgb'/f'{i:05}.png');d=ImageDraw.Draw(im)
  if rect:d.rectangle(rect,outline='yellow',width=3)
  d.text((12,12),f'{name} actor {aid} cam{cam} source f{f}',fill='yellow');sheet.paste(im,(0,576*n))
 sheet.resize((768,1296)).save(out/'selection.jpg',quality=92)
reg={'task_id':'WS-V77-DELETE-REPAIR-20260927','run_id':'r1','registered':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'scenes':scenes,'seed':42,'training_steps':0,'scope':'三场景各30帧10Hz主视图背景入口强控制；搜索六相机整段真实RGB。旧两scene选失败后段，第三scene为已有official_000中的移动黑色MPV actor12，避免超近裁切的truck14。不是新独立测试集。','mask':'SAM2实例轮廓 ∩ GT凸包+3px，只膨胀3px；box仅prompt/一致性约束。','evidence':'原始RGB Ω缓存+GT相机/框+LiDAR尺度/静态支持；源车辆投影贴地包络排除；深度/遮挡/双来源一致性；禁止旧生成背景作真实证据。','arms':['precise_mask_only','evidence_then_residual'],'driveeditor':{'seed':42,'steps':25,'frames':10,'stride':9,'max_windows':24,'previous':'相邻窗同时间重叠composite，初窗无上窗条件；两组相同'},'guard':'GroundingDINO车辆检测+SAM2实例和原图已有非目标匹配；命中目标区的新车辆直接阻断后续Ω，漏检不等于通过；人工verdict null。','budget':'单RTX3090串行，CPU4线程，DriveEditor每窗180秒总2400秒；不扫seed/模型/阈值。','stop_rule':'明确车形/邻车损伤阻断该结果进入Ω；保留负例，不无限重试。若证据无覆盖，不重复同样推理。','failure_ledger_refs':['V77-F02'],'human_verdict':None}
dump(ROOT/'registration.json',reg);print([(s['name'],len(s['donor_frames'])) for s in scenes])
