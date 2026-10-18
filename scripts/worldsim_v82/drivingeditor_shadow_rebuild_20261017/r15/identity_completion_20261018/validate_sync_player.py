"""Validate actual 49-frame video inputs and decoded frame order, without browser claims."""
from pathlib import Path
import json,re,subprocess,shutil,time
import numpy as np
from PIL import Image
W=Path(__file__).parent;O=W.parents[1]/'outputs/identity_comparison_20261018'
page=(O/'sync_player.html').read_text(encoding='utf-8')
data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',page,re.S).group(1))
assert (O/'index.html').read_text(encoding='utf-8')==page
assert data['first']==89 and data['last']==137 and data['fps']==10
assert all(data['models'][m]['ready'] for m in ['old5','r15'])
ff=shutil.which('ffmpeg');probe=shutil.which('ffprobe');records=[]
def inspect(path):
    r=json.loads(subprocess.check_output([probe,'-v','error','-select_streams','v:0','-show_entries',
        'stream=codec_name,width,height,r_frame_rate,nb_frames,duration:frame=key_frame,best_effort_timestamp_time',
        '-of','json',str(path)],text=True))
    stream=r['streams'][0];frames=r['frames']
    assert stream['codec_name']=='h264' and stream['r_frame_rate']=='10/1'
    assert len(frames)==49 and int(stream['nb_frames'])==49
    assert all(f['key_frame']==1 for f in frames)
    assert np.allclose([float(f['best_effort_timestamp_time']) for f in frames],np.arange(49)/10)
    return dict(file=path.relative_to(O).as_posix(),frames=49,fps=10,all_keyframes=True,
        dimensions=[stream['width'],stream['height']])
def source_for(model,key,actor=None):
    src=O/'assets'/model;br=W/'branches'/('branches_'+model)
    if actor is None:
        if key in ['observed','full','full_no_shadow']:return src,{'observed':'gt.png','full':'full.png','full_no_shadow':'full_no_shadow.png'}[key]
        return br,key+'.png'
    if key in ['isolated','isolated_alpha','delete','delete_no_shadow']:
        return src,f'a{actor:02d}_'+{'isolated':'isolated','isolated_alpha':'alpha','delete':'delete','delete_no_shadow':'delete_no_shadow'}[key]+'.png'
    return br,f'a{actor:02d}_'+key+'.png'
for model,m in data['models'].items():
    if not m['ready']:continue
    receipt=json.loads((W/'branches'/('branches_'+model)/'branch_receipt.json').read_text(encoding='utf-8'))
    assert receipt['frames']==49 and receipt['parameter_updates']==0
    assert {(x['frame_id'],x['actor']) for x in receipt['records']}=={(f'0008{raw:03d}',a) for raw in range(89,138) for a in [6,37,5,26,18]}
    assert set(['dynamic','static','dynamic_alpha','static_alpha']).issubset(m['base'])
    items=[(None,k,v) for k,v in m['base'].items()]
    for actor,values in m['actors'].items():
        assert set(['isolated','isolated_alpha','delete','dynamic_delete','dynamic_delete_alpha']).issubset(values)
        items.extend((int(actor),k,v) for k,v in values.items())
    for actor,key,value in items:
        path=O/value;r=inspect(path);src,filename=source_for(model,key,actor)
        assert r['dimensions']==[480,320]
        assert all((src/f'0008{raw:03d}'/filename).is_file() for raw in range(89,138))
        pixels=subprocess.check_output([ff,'-hide_banner','-loglevel','error','-threads','1','-i',str(path),
            '-vf',r'select=eq(n\,0)+eq(n\,24)+eq(n\,48)','-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
        frames=np.frombuffer(pixels,dtype=np.uint8).reshape(3,320,480,3)
        errors=[]
        for i,raw in enumerate([89,113,137]):
            expected=np.asarray(Image.open(src/f'0008{raw:03d}'/filename).convert('RGB'))
            errors.append(float(np.abs(expected.astype(float)-frames[i]).mean()))
        assert max(errors)<6,(value,errors)
        r['sample_decoded_vs_source_mae_255']=errors;r['sample_raw_frames']=[89,113,137];records.append(r)
    print(model+' decoded frame checks complete',flush=True)
exports=[]
for actor in [6,37,5,26,18]:
    r=inspect(O/'sync_exports'/f'a{actor:02d}_old5_vs_r15.mp4')
    assert r['dimensions']==[1260,1140];exports.append(r)
logic=subprocess.check_output(['node',str(W/'test_player_sync.cjs')],text=True)
result=dict(status='PASSED_MEDIA_AND_TRANSPORT_CHECKS',models=list(data['models']),branch_videos=records,
    five_inseparable_overviews=exports,transport=json.loads(logic),actual_browser_acceptance=False,
    browser_limitation='Automatic browser refused file://. No workaround used; local-file UI acceptance not verified.',
    native_MP4s_are_actual_render_composites=True,all_source_images_unmasked=True,human_verdict=None)
(O/'player_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(status=result['status'],branch_videos=len(records),composite_videos=len(exports))))
