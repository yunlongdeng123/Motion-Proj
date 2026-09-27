"""物化六相机完整窗口与mask，固定滑窗、角色、资产和米制标定。"""
import ast,datetime,json,pathlib,shutil,sys,subprocess
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from delete_full_common import ROOT,REPO,OLD,DE,SCENES,dump
sys.path.insert(0,str(REPO/'scripts/worldsim_v77'))
from geometry import transform
from video_review import scene_frame

assert not (ROOT/'registration.json').exists(),'不覆盖已登记run'
ROOT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(4);torch.set_num_threads(4)
old=json.loads((OLD/'registration.json').read_text())
tree=ast.parse(subprocess.check_output(['git','-C',str(DE),'show','HEAD:interactive_gui.py'],text=True))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scale_bbox')
ns={'np':np,'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'official_scale_bbox','exec'),ns)
records=[];all_windows=[]
for name,info in SCENES.items():
    src=pathlib.Path('/root/autodl-tmp/data/v76_vadgs')/name;dst=ROOT/name;dst.mkdir(exist_ok=True)
    inst=json.loads((src/'instances/instances_info.json').read_text());fa=inst[info['actor']]['frame_annotations']
    spec=next(s['spec'] for s in old['scenes'] if s['name']==name)
    dump(dst/'instances_snapshot.json',inst);dump(dst/'spec.json',spec)
    shutil.copy2(OLD/name/'actor/actor.glb',dst/'actor.glb')
    rows=[];frames=[]
    for f in range(info['count']):
        frame=scene_frame(spec,f,inst)
        frames.append({'frame':f,'time_s':f/10,'views':frame['views'],'all_boxes':frame['all_boxes'],'origin_world':frame['origin_world']})
    dump(dst/'camera_frames.json',frames)
    for c in range(6):
        torch.manual_seed(42)
        v=spec['views'][c];K=np.asarray(v['intrinsics']);stream=dst/f'cam{c}'
        for sub in ['rgb','mask','target_sam']:(stream/sub).mkdir(parents=True,exist_ok=True)
        active=[];mask_rows=[]
        for f in range(info['count']):
            im=cv2.resize(cv2.imread(str(src/'images'/f'{f:03}_{c}.jpg')),(1024,576),interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(stream/'rgb'/f'{f:05}.png'),im)
            path=OLD/name/'masks'/f'cam{c}'/f'{f:05}.png'
            sam=cv2.resize(cv2.imread(str(path),0),(1024,576),interpolation=cv2.INTER_NEAREST)>0 if path.is_file() else np.zeros((576,1024),bool)
            if name=='scene_0230' and c==5 and sam.any():
                n,labels,stats,_=cv2.connectedComponentsWithStats(sam.astype('uint8'),8)
                sam=labels==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))
            mask=np.zeros((576,1024),np.uint8);mode='empty_target'
            if sam.any():
                active.append(f);yy,xx=np.where(sam)
                if name=='scene_0230':
                    rect=[max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(1024,int(xx.max())+9),min(576,int(yy.max())+25)];mode='SAM_rect_8_8_8_24'
                else:
                    j=fa['frame_idx'].index(f);pose=np.asarray(fa['obj_to_world'][j]);size=np.asarray(fa['box_size'][j])
                    signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
                    cp=transform(transform(signs*size/2,pose),np.linalg.inv(np.asarray(frames[f]['views'][c]['c2w'])))
                    assert cp[:,2].min()>.1
                    uv=cp@K.T;corners=uv[:,:2]/uv[:,2:3]
                    rect=np.round(ns['scale_bbox'](corners,(900,1600),1.9,1.9)*.64).astype(int).tolist();mode='official_expanded_seed42'
                    if c==3 and 15<=f<=24:
                        saved=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DRIVEEDITOR-COMPARE-20260927/r1/scene_0255/mask_b')/f'{f-15:05}.png'
                        sm=np.array(Image.open(saved))>0;y,x=np.where(sm);rect=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1];mode='reused_verified_short_mask'
                    rect=[min(rect[0],int(xx.min())),min(rect[1],int(yy.min())),max(rect[2],int(xx.max())+1),max(rect[3],int(yy.max())+1)]
                x0,y0,x1,y1=rect;mask[y0:y1,x0:x1]=255
                assert (mask[sam]>0).all()
            cv2.imwrite(str(stream/'mask'/f'{f:05}.png'),mask);cv2.imwrite(str(stream/'target_sam'/f'{f:05}.png'),sam.astype('uint8')*255)
            mask_rows.append({'frame':f,'active':bool(sam.any()),'sam_pixels':int(sam.sum()),'edit_pixels':int((mask>0).sum()),'mode':mode})
        windows=[]
        for start in range(0,info['count']-1,9):
            ids=[min(start+i,info['count']-1) for i in range(10)]
            if any(f in active for f in ids):
                w={'scene':name,'camera':c,'start':start,'source_frames':ids,'valid_count':min(10,info['count']-start)};windows.append(w);all_windows.append(w)
        row={'camera':c,'active_frames':active,'mask_rows':mask_rows,'windows':windows};rows.append(row)
        if active:
            picks=sorted(set([active[0],active[len(active)//2],active[-1]]));sheet=Image.new('RGB',(1024,220*len(picks)),'white')
            for n,f in enumerate(picks):
                rgb=cv2.cvtColor(cv2.imread(str(stream/'rgb'/f'{f:05}.png')),cv2.COLOR_BGR2RGB);mask=cv2.imread(str(stream/'mask'/f'{f:05}.png'),0)>0
                sam=cv2.imread(str(stream/'target_sam'/f'{f:05}.png'),0)>0
                over=rgb.copy();over[mask]=(.65*over[mask]+.35*np.array([255,155,15])).astype('uint8')
                cs,_=cv2.findContours(sam.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(over,cs,-1,(0,255,240),2)
                sheet.paste(Image.fromarray(rgb).resize((384,216)),(0,n*220));sheet.paste(Image.fromarray(over).resize((384,216)),(384,n*220));ImageDraw.Draw(sheet).text((775,n*220+15),f'{name} cam{c} f{f}\nmask {mask.mean():.1%}',fill='black')
            sheet.save(dst/f'input_cam{c}.jpg',quality=92)
    dump(dst/'streams.json',rows)
    records.append({'name':name,**info,'source':str(src),'track_id':inst[info['actor']]['id'],'streams':rows,'spec':spec})
assert len(all_windows)<=40,len(all_windows)
reg={'task_id':'WS-V77-DELETE-FULL-20260927','run_id':'r1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'scenes':records,'windows':all_windows,'omega_source':old['omega_source'],'omega_checkpoint':old['frozen_omega_checkpoint'],'input_roles':'六相机RGB；GT仅为SAM历史提示/门控、相机内外参、actor位姿尺寸；框外LiDAR作Ω米制尺度。继承SAM2在低于400px投影及近裁面不完整时不编辑的边界。两个已曝光开发scene，训练重叠未核对。','background':'时刻化B_t资产，不称持久静态或4D世界；六相机联合Ω；灰洞纯几何与RGB背景补足分开输出','driveeditor':{'steps':25,'seed_each_window':42,'frames':10,'stride':9,'overlap':1,'boundary_condition':'上一窗同一重叠时刻的composite图作为首帧条件，避免使用错位时间帧；重复首帧保留旧输出','previous_condition_reset_on_gap':True,'mask_0230':'已验证SAM矩形8/8/8/24','mask_0255':'官方扩大mask，随机seed42物化；已验证15–24帧原mask复用；必要时最小扩展包含SAM','inactive_views':'逐像素原RGB，不调用模型','tail':'重复末帧仅供10帧模型输入；输出不重复尾帧'},'query':'B_t+同一GLB为factual；DELETE仅visible=False；不再运行补景/Ω/生成器','training_steps':0,'seed':42,'resource':'RTX3090串行；CPU4线程；最多40窗，每窗180秒预算、总DriveEditor3600秒；Ω总3600秒','stop_rule':'工程异常停止相应阶段并留证；不扫模型/seed；质量局限如实输出，不作人工通过','failure_ledger_refs':['V77-F02'],'human_verdict':None}
dump(ROOT/'registration.json',reg)
print(json.dumps({'prepared':str(ROOT),'windows':len(all_windows),'active':{s['name']:{str(r['camera']):len(r['active_frames']) for r in s['streams']} for s in records}},ensure_ascii=False),flush=True)
