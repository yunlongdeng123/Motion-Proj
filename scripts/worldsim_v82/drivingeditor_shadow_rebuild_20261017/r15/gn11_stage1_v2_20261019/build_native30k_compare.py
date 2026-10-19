"""将实际完整49PNG编码成全画面同步对比；不调用远端/GPU。"""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,uuid

W=Path(__file__).resolve().parent
O=W.parent.parent/'outputs/gn11_stage1_v2_20261019'
SOURCE=W/'native30k_pair'
CHANNELS={'static':'静态分支','dynamic':'动态分支','composite':'完整重建','gt':'原始观测'}
ARMS={'full':'完整版 · 归属＋跨帧＋供体','real_only':'仅跨帧 · 归属＋跨帧'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def probe(path):
    p=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_frames',
        '-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration:frame=key_frame,pict_type',
        '-of','json',str(path)],capture_output=True,text=True,check=True)
    data=json.loads(p.stdout);s=data['streams'][0]
    assert (s['width'],s['height'])==(480,320) and int(s['nb_read_frames'])==49 and s['r_frame_rate']=='10/1'
    assert len(data['frames'])==49 and all(f['key_frame']==1 and f['pict_type']=='I' for f in data['frames'])
    return s

def main():
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    download=json.loads((W/'native30k_pair_download.json').read_text(encoding='utf-8'))
    assert manifest['PNG_count']==392 and download['all_file_hashes_verified']
    media=O/'native30k_videos';media.mkdir(exist_ok=True)
    previous=O/'native30k_media.json'
    old=json.loads(previous.read_text(encoding='utf-8')) if previous.exists() else {'videos':{}}
    records={}
    for arm in ARMS:
        for channel in CHANNELS:
            key=arm+'_'+channel
            source_rows=sorted([e for e in manifest['files'] if e['path'].startswith(arm+'/'+channel+'/')],key=lambda e:e['path'])
            assert len(source_rows)==49
            source_digest=hashlib.sha256(json.dumps([(e['path'],e['sha256']) for e in source_rows],separators=(',',':')).encode()).hexdigest()
            for e in source_rows:assert sha(SOURCE/e['path'])==e['sha256']
            target=media/(key+'.mp4');prior=old['videos'].get(key)
            if target.exists():
                if not prior or prior['source_sha256']!=source_digest or prior['sha256']!=sha(target):
                    raise RuntimeError('existing video lacks matching provenance: '+key)
            else:
                pending=target.with_name(target.stem+'.encoding_'+uuid.uuid4().hex[:8]+'.mp4')
                subprocess.run(['ffmpeg','-v','error','-nostdin','-framerate','10','-start_number','8089',
                    '-i',str(SOURCE/arm/channel/'%07d.png'),'-frames:v','49','-an','-c:v','libx264',
                    '-preset','slow','-crf','14','-pix_fmt','yuv420p','-g','1','-keyint_min','1',
                    '-sc_threshold','0','-bf','0','-threads','4','-movflags','+faststart',str(pending)],check=True)
                probe(pending);pending.rename(target)
            contract=probe(target)
            poster=media/(key+'_raw089.png');shutil.copyfile(SOURCE/arm/channel/'0008089.png',poster)
            records[key]={'path':target.relative_to(O).as_posix(),'poster':poster.relative_to(O).as_posix(),
                'source_sha256':source_digest,'sha256':sha(target),'bytes':target.stat().st_size,**contract}
            # Save each successful complete encode so interruption never presents a partial video as ready.
            previous.write_text(json.dumps({'videos':{**old['videos'],**records}},indent=2),encoding='utf-8')
    receipt={'status':'READY_TWO_COMPLETED_30K_NATIVE49_VIDEO_COMPARISON','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'arms':list(ARMS),'channels':list(CHANNELS),'frames':list(range(89,138)),'full_frame_wh':[480,320],
        'fps':10,'all_I_frames':True,'videos':records,'source_manifest_sha256':sha(SOURCE/'manifest.json'),
        'archive_sha256':download['archive_sha256'],'additional_training_updates':0,'GPU_used_for_export':False,
        'native_checkpoint_evaluation':True,'independent_delete_evaluation_completed':False,
        'quality_accepted':False,'human_verdict':None,'browser_UI_tested':False}
    previous.write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    shutil.copyfile(SOURCE/'manifest.json',O/'native30k_sources.json')
    svg='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 230"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#8ec5df"/></marker></defs><rect width="1000" height="230" fill="#142536"/><g fill="#223f55" stroke="#7398b2"><rect x="20" y="55" width="210" height="120" rx="12"/><rect x="300" y="20" width="350" height="80" rx="12"/><rect x="300" y="130" width="350" height="80" rx="12"/><rect x="730" y="55" width="250" height="120" rx="12"/></g><g fill="none" stroke="#8ec5df" stroke-width="2" marker-end="url(#a)"><path d="M230 95H265V60H300"/><path d="M230 135H265V170H300"/><path d="M650 60H690V95H730"/><path d="M650 170H690V135H730"/></g><g fill="#edf5fb" text-anchor="middle" font-family="Microsoft YaHei,sans-serif" font-size="19"><text x="125" y="101">相同五身份 / 49帧</text><text x="125" y="139">相同初始化协议</text><text x="475" y="53">完整版 · fresh 30K</text><text x="475" y="82">归属修复＋真实＋供体</text><text x="475" y="163">仅跨帧 · fresh 30K</text><text x="475" y="192">归属修复＋真实</text><text x="855" y="101">原生49帧 / 4通道</text><text x="855" y="139">完整画面 · 同步对比</text></g></svg>'''
    (O/'native30k_compare.svg').write_text(svg,encoding='utf-8')
    data={'fps':10,'first':89,'last':137,'arms':ARMS,'channels':CHANNELS,'videos':records}
    html=(W/'native30k_compare_template.html').read_text(encoding='utf-8').replace('@@DATA@@',json.dumps(data,ensure_ascii=False,separators=(',',':')))
    (O/'native30k_compare.html').write_text(html,encoding='utf-8')
    print(json.dumps({'report':str(O/'native30k_compare.html'),'videos':len(records),'frames_each':49,'bytes':sum(r['bytes'] for r in records.values())}))

if __name__=='__main__':main()
