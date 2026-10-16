"""Verify actual deliverables and preserve quality failures separately."""
import json,subprocess,shutil,math,re,hashlib,datetime
from pathlib import Path
from PIL import Image
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'outputs/shadow_sequence_v2'
def finite(x):
    if isinstance(x,float):return math.isfinite(x)
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,list):return all(finite(v) for v in x)
    return True
def main():
    errors=[];json_count=0;image_count=0;video_rows=[]
    for p in OUT.rglob('*.json'):
        try:
            value=json.loads(p.read_text(encoding='utf-8-sig'));assert finite(value);json_count+=1
        except Exception as e:errors.append(str(p.relative_to(OUT))+': '+str(e))
    for p in [*OUT.rglob('*.png'),*OUT.rglob('*.jpg')]:
        try:
            with Image.open(p) as im:im.load();assert im.width>0 and im.height>0
            image_count+=1
        except Exception as e:errors.append(str(p.relative_to(OUT))+': '+str(e))
    probe=shutil.which('ffprobe');assert probe
    for p in OUT.rglob('*.mp4'):
        result=subprocess.run([probe,'-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=codec_name,pix_fmt,nb_read_frames,width,height','-of','json',str(p)],capture_output=True,text=True,check=True)
        s=json.loads(result.stdout)['streams'][0]
        if s['codec_name']!='h264' or int(s['nb_read_frames'])!=49:errors.append('Video mismatch '+str(p))
        video_rows.append({'file':p.relative_to(OUT).as_posix(),**s})
    html=(OUT/'report.html').read_text(encoding='utf-8');refs=re.findall(r'(?:src|href)="([^"]+)"',html)
    for ref in refs:
        if ref.startswith(('http:','https:','#','data:')):continue
        if ref=='evidence/final_validation.json':continue # This run creates the linked receipt below.
        if not (OUT/ref).exists():errors.append('Broken HTML reference '+ref)
    visual=json.loads((OUT/'evidence/visual_review.json').read_text(encoding='utf-8'))
    if visual.get('status')!='REVIEWED':errors.append('Primary visual review pending')
    selected=json.loads((OUT/'selected.json').read_text(encoding='utf-8'))
    actual_hash=hashlib.sha256((OUT/selected['training']/'final.pt').read_bytes()).hexdigest()
    if actual_hash!=selected['checkpoint_sha256']:errors.append('Selected addon checksum mismatch')
    accounting=json.loads((OUT/'evidence/training_accounting.json').read_text(encoding='utf-8'))
    if accounting['total_gradient_updates']!=18000:errors.append('Executed gradient count mismatch')
    process=json.loads((OUT/'evidence/completion_process_check.json').read_text(encoding='utf-8'))
    if any(r['alive'] for r in process['jobs']):errors.append('An owned worker is still active')
    gates=json.loads((OUT/'quality_gates.json').read_text(encoding='utf-8'))
    record={'evidence_status':'PASS' if not errors else 'FAIL','errors':errors,'quality_gates':gates,
        'quality_status':'ROAD_SCOPE_PASS_WITH_REMAINING_GOALS' if all(gates.values()) else 'PARTIAL_WITH_UNMET_GATES',
        'json_files':json_count,'decoded_images':image_count,'videos':video_rows,'HTML_references':len(refs),
        'all49_comparison_frames_visually_reviewed':visual['comparison_frames_reviewed']==49,
        'representative_native_edit_views_reviewed':True,'all_stream_frames_manually_reviewed':False,
        'video_playback_not_realtime_human_certification':True,'overall_problem_fully_solved':False,
        'selected_addon_hash_verified':actual_hash,'actual_gradient_updates':accounting['total_gradient_updates'],
        'no_paired_edit_GT':True,'no_unseen_frame_claim':True}
    (OUT/'evidence/final_validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    hashes={p.relative_to(OUT).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':hashes},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='videos'},ensure_ascii=False))
    if errors:raise SystemExit(2)
if __name__=='__main__':main()
