"""QUERY仅读取背景、GLB渲染层和指令，导出完整连续视频与诊断。"""
import argparse,json,pathlib,subprocess,time,shutil
import numpy as np
from PIL import Image,ImageDraw
from delete_full_common import ROOT,dump,apply_delete
W,H=688,384

def compose_actor(background,rgba,actor_z,background_z,tolerance=.15):
    if rgba.shape[:2]!=background.shape[:2] or actor_z.shape!=background_z.shape:raise ValueError('相机栅格不一致')
    alpha=rgba[...,3].astype('float32')/255
    valid=(alpha>0)&np.isfinite(actor_z)&(actor_z>0)
    visible=valid&((~np.isfinite(background_z))|(actor_z<=background_z+tolerance))
    weight=alpha*visible
    rgb=np.rint(background*(1-weight[...,None])+rgba[...,:3]*weight[...,None]).clip(0,255).astype('uint8')
    return rgb,{'asset_pixels':int(valid.sum()),'visible_asset_pixels':int(visible.sum()),'occluded_asset_pixels':int((valid&~visible).sum())}

def load_rgb(p):return np.array(Image.open(p).convert('RGB').resize((W,H),Image.Resampling.BICUBIC))

def label(im,text,color=(25,35,50)):
    out=Image.fromarray(im).convert('RGB');d=ImageDraw.Draw(out);d.rectangle((0,0,out.width,23),fill=color);d.text((6,6),text,fill='white');return out

class Writer:
    def __init__(self,path,size):
        self.path=path;self.count=0
        exe=shutil.which('ffmpeg')
        if not exe:
            import imageio_ffmpeg
            exe=imageio_ffmpeg.get_ffmpeg_exe()
        self.p=subprocess.Popen([exe,'-hide_banner','-loglevel','error','-y','-filter_threads','1','-filter_complex_threads','1','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r','10','-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def write(self,im):self.p.stdin.write(np.asarray(im,dtype='uint8').tobytes());self.count+=1
    def close(self):self.p.stdin.close();assert self.p.wait()==0,str(self.path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=['scene_0230','scene_0255']);a=p.parse_args()
    reg=json.loads((ROOT/'registration.json').read_text())
    if not a.scene:assert json.loads((ROOT/'omega_state.json').read_text())['state']=='complete'
    review=ROOT/'review';review.mkdir(exist_ok=True);all_summary=[]
    if (review/'summary.json').exists():raise FileExistsError('已有完整QUERY产物；新对照应另存，不能覆盖')
    for s in reg['scenes']:
        if a.scene and s['name']!=a.scene:continue
        name=s['name'];base=ROOT/name;dest=review/name;dest.mkdir(exist_ok=True);frames=json.loads((base/'camera_frames.json').read_text())
        assert all((base/'background_world'/f'{f:03}'/'summary.json').exists() for f in range(s['count'])),'该scene的Ω须先全部完成'
        if (dest/'summary.json').exists():continue
        manifest={'schema':'v77-explicit-delete/1','scene':name,'fps':10,'frame_count':s['count'],'coordinate_system':'per-frame local = GT world minus stored origin_world','background':[{'frame':fr['frame'],'asset':f'background_world/{fr["frame"]:03}/background_points.npz','origin_world':fr['origin_world']} for fr in frames],'actors':[{'actor_id':s['actor'],'asset':'actor.glb','visible':True,'yaw_correction_deg':s['yaw'],'poses':[{'frame':fr['frame'],**next(b for b in fr['all_boxes'] if b['actor_id']==s['actor'])} for fr in frames]}],'render':{'resolution':[W,H],'point_splat_radius':1,'depth_tolerance_m':.15,'camera_source':'GT','active_views':'registered SAM gating; other target observations can remain baked','lighting':'fixed world+sun, no fitted illumination/shadow catcher'},'other_objects':'baked into background, not independently editable','human_verdict':None}
        manifest['asset_root_remote']=str(base)
        deleted=apply_delete(manifest,s['actor']);dump(base/'scene_factual.json',manifest);dump(base/'scene_delete.json',deleted);dump(base/'command.json',{'operation':'DELETE','actor_id':s['actor'],'legality':'no new occupied volume; visual identity checked; no MOVE or traffic-rule validity claim'})
        for p in ['scene_factual.json','scene_delete.json','command.json']:(dest/p).write_text((base/p).read_text())
        writers={k:Writer(dest/f'{k}.mp4',(W,H)) for k in ['original','factual','delete','driveeditor','mask','geometry_factual','geometry_delete']}
        writers.update({f'six_{k}':Writer(dest/f'six_{k}.mp4',(1032,384)) for k in ['original','factual','delete']})
        rows=[];checks={'outside_mask_max':0,'unique_frames':s['count'],'video_frames':{},'factual_delete_difference_outside_asset_max':0};selection=[];contacts=[]
        for fr in frames:
            f=fr['frame'];out=base/'background_world'/f'{f:03}';summary=json.loads((out/'summary.json').read_text());views=[]
            areas=[np.array(Image.open(base/f'cam{c}/target_sam/{f:05}.png')).astype(bool).sum() for c in range(6)]
            c_best=int(np.argmax(areas)) if max(areas)>0 else (5 if name=='scene_0230' else 3);selection.append({'frame':f,'camera':c_best,'sam_area':int(areas[c_best])})
            for c in range(6):
                stream=base/f'cam{c}';original_full=np.array(Image.open(stream/f'rgb/{f:05}.png'));bg_full=np.array(Image.open(stream/f'background/{f:05}.png'));m_full=np.array(Image.open(stream/f'mask/{f:05}.png'))>0
                if (~m_full).any():checks['outside_mask_max']=max(checks['outside_mask_max'],int(np.abs(original_full.astype('int16')-bg_full.astype('int16'))[~m_full].max()))
                original=load_rgb(stream/f'rgb/{f:05}.png');bg=load_rgb(stream/f'background/{f:05}.png');pure=load_rgb(out/f'geometry_cam{c}.png');hybrid=load_rgb(out/f'hybrid_cam{c}.png');z=np.load(out/f'z_cam{c}.npy')
                layer=base/'actor_layers'/f'f{f:03}_cam{c}.png'
                expected_layer=f in s['streams'][c]['active_frames']
                assert layer.exists()==expected_layer,(name,f,c,'渲染层与登记不一致')
                if layer.exists():
                    rgba=np.array(Image.open(layer).convert('RGBA'));az=np.load(layer.with_name(layer.stem+'_depth.npy'))
                    factual,vis=compose_actor(hybrid,rgba,az,z);pure_factual,_=compose_actor(pure,rgba,az,z)
                    changed=(factual!=hybrid).any(-1);assert not (changed&(rgba[...,3]==0)).any()
                else:factual=hybrid.copy();pure_factual=pure.copy();vis={'asset_pixels':0,'visible_asset_pixels':0,'occluded_asset_pixels':0}
                # 实际执行指令读取同一背景；任何网络或资产变形均不属于QUERY。
                deletion=hybrid.copy() if not deleted['actors'][0]['visible'] else factual
                rows.append({'frame':f,'camera':c,**vis,**summary['views'][c]})
                mask=np.array(Image.open(stream/f'mask/{f:05}.png').resize((W,H),Image.Resampling.NEAREST))>0
                sam=np.array(Image.open(stream/f'target_sam/{f:05}.png').resize((W,H),Image.Resampling.NEAREST))>0
                marked=original.copy();marked[sam]=(marked[sam]*.5+np.array([255,180,20])*.5).astype('uint8')
                if sam.any():
                    y,x=np.where(sam);im=Image.fromarray(marked);ImageDraw.Draw(im).rectangle((x.min(),y.min(),x.max(),y.max()),outline='#ffc43d',width=2);marked=np.array(im)
                maskview=original.copy();maskview[mask]=(maskview[mask]*.35+np.array([90,160,255])*.65).astype('uint8')
                views.append({'original':marked,'factual':factual,'delete':deletion,'driveeditor':bg,'mask':maskview,'geometry_factual':pure_factual,'geometry_delete':pure})
            for k in ['original','factual','delete','driveeditor','mask','geometry_factual','geometry_delete']:
                writers[k].write(label(views[c_best][k],f'{name} actor{s["actor"]} | CAM{c_best} | {f/10:.1f}s f{f:03} | {k}'))
            for k in ['original','factual','delete']:
                mosaic=Image.new('RGB',(1032,384))
                for c in range(6):mosaic.paste(label(views[c][k],f'CAM{c} f{f:03} {k}').resize((344,192)),((c%3)*344,(c//3)*192))
                writers[f'six_{k}'].write(mosaic)
            if f in set(np.linspace(0,s['count']-1,10).round().astype(int)):
                panel=Image.new('RGB',(1032,192))
                for i,k in enumerate(['original','factual','delete']):panel.paste(label(views[c_best][k],f'f{f:03} CAM{c_best} {k}').resize((344,192)),(i*344,0))
                panel.save(dest/f'frame_{f:03}.jpg',quality=92);contacts.append(panel)
        for k,w in writers.items():w.close();checks['video_frames'][k]=w.count;assert w.count==s['count']
        assert checks['outside_mask_max']==0
        sheet=Image.new('RGB',(1032,192*len(contacts)))
        for i,panel in enumerate(contacts):sheet.paste(panel,(0,i*192))
        sheet.save(dest/'contact.jpg',quality=93)
        coverage=[r['geometry_coverage'] for r in rows];targetcov=[r['deletion_mask_geometry_coverage'] for r in rows if r['deletion_mask_geometry_coverage'] is not None]
        cross=[]
        for f in range(s['count']):cross.extend(json.loads((base/'background_world'/f'{f:03}'/'summary.json').read_text())['cross_view'])
        total=sum(x['count'] for x in cross);over=sum(x['over_10pct_count'] for x in cross)
        result={'scene':name,'actor_id':s['actor'],'frame_count':s['count'],'duration_s':s['count']/10,'views':6,'checks':checks,'selection':selection,'geometry_coverage_min_mean':[min(coverage),float(np.mean(coverage))],'deletion_mask_geometry_coverage_min_mean':[min(targetcov),float(np.mean(targetcov))],'asset_pixels':sum(r['asset_pixels'] for r in rows),'visible_asset_pixels':sum(r['visible_asset_pixels'] for r in rows),'cross_camera_mask_sample_pairs':total,'cross_camera_relative_depth_over_10pct_fraction':over/total if total else None,'cross_camera_note':'非真值准确率，包含真实遮挡/深度错配，不代表多视角一致性通过','rows':rows,'human_verdict':None}
        dump(dest/'summary.json',result);all_summary.append(result)
    if all((review/s['name']/'summary.json').exists() for s in reg['scenes']):
        all_summary=[json.loads((review/s['name']/'summary.json').read_text()) for s in reg['scenes']]
        dump(review/'summary.json',{'task_id':reg['task_id'],'run':'r1','scenes':all_summary,'training_steps':0,'human_verdict':None});print('QUERY_COMPLETE',flush=True)
    else:print('QUERY_SCENE_COMPLETE',a.scene,flush=True)

if __name__=='__main__':main()
