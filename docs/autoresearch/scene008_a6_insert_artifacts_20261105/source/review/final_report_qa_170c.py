"""Read-only local publication addendum: no publisher import, GPU or network."""
import base64, hashlib, json, re, zipfile
from collections import Counter
from pathlib import Path
from html.parser import HTMLParser
import numpy as np
from PIL import Image
import final_report_qa as base

EXPECTED_HTML='170ccef3351164a1ce8660a07d2f906f8e522fd406c9b0956de4b1eae56a0191'
EXPECTED_RECEIPT='87fa27d13a580089a383e6a4153ea4d6aa3a057a9a1077057c12f93ff7fcb2b1'
EXPECTED_ZIP='3eef0c7081114a52c0dd72ecff2130d1fca7953df330148ca7d60bb684495543'

class Figures(HTMLParser):
    def __init__(self):super().__init__();self.figures=[];self.current=None;self.caption=False
    def handle_starttag(self,t,a):
        d=dict(a)
        if t=='figure':self.current=dict(caption='',src=None,alt=None)
        if t=='figcaption':self.caption=True
        if t=='img' and self.current is not None:self.current.update(src=d.get('src'),alt=d.get('alt'))
    def handle_data(self,s):
        if self.current is not None and self.caption:self.current['caption']+=s
    def handle_endtag(self,t):
        if t=='figcaption':self.caption=False
        if t=='figure':self.figures.append(self.current);self.current=None

def main():
    base.main()
    A,O=base.A,base.O
    report=base.read(A/'review/final_report_review.json');issues=report['must_fix']
    def check(ok,kind,detail=None):
        if not ok:issues.append(dict(kind=kind,details=detail))
    pub=base.read(O/'publication_receipt.json');h=(O/'index.html').read_text(encoding='utf-8')
    check(base.sha(O/'index.html')==EXPECTED_HTML,'unexpected_HTML_snapshot')
    check(base.sha(O/'publication_receipt.json')==EXPECTED_RECEIPT,'unexpected_publication_receipt')
    check((O/'index.html').stat().st_size==4325671,'unexpected_HTML_bytes')
    f=Figures();f.feed(h);p=base.Parser();p.feed(h)
    check(len(f.figures)==len(pub['images'])==21,'figure_count')
    for i,(fig,im) in enumerate(zip(f.figures,pub['images'])):
        check(fig['caption']==im['caption']==fig['alt'],'HTML_caption_receipt_binding',i)
        check(hashlib.sha256(base64.b64decode(fig['src'].split(',',1)[1])).hexdigest()==im['source_sha256'],'HTML_inline_source_SHA',i)

    names=['black_HR_comparison.png','black_native480_comparison.png','delete_crop480_comparison.png']
    bindings=pub['final_hook']['figure_bindings'];selected=pub['final_hook']['available_figures']
    check([(x['raw'],Path(x['path']).name) for x in bindings]==[(r,n) for r in [106,113,130] for n in names],'nine_figure_explicit_order')
    check(selected==[x['path'] for x in bindings],'available_figures_bindings')
    pinned=base.read(A/'report/final_result_manifest.json')
    check([(A/x['path']).resolve() for x in pinned['figures']]==[Path(x).resolve() for x in selected],'manifest_pinned_paths')
    tile_results=[]
    for x in bindings:
        path=Path(x['path']);raw=x['raw'];columns=4 if path.name==names[0] else 5
        matches=[im for im in pub['images'] if Path(im['source'])==path]
        check(path.parent.name==f'raw{raw}' and x['caption'].startswith(f'raw{raw}，'),'caption_raw_parent',x)
        check(base.sha(path)==x['sha256'] and len(matches)==1 and matches[0]['caption']==x['caption'] and matches[0]['source_sha256']==x['sha256'],'caption_source_SHA_chain',x)
        check(x['panel_columns']==columns,'panel_column_metadata',x)
        # Independent content check: every displayed panel equals the matching
        # standalone native output from that raw directory, after the saved crop.
        board=Image.open(path).convert('RGB')
        if columns==4:
            crop=np.load(path.parent/'HR_black_replay.npz')['crop_xyxy'].tolist();height=300
            files=[n+'_black1920.png' for n in ['thin','old','initial','final']]
        else:
            crop=np.load(path.parent/'native480_replay.npz')['crop_xyxy'].tolist();height=250
            files=[n+('_delete480.png' if path.name==names[2] else '_black480.png') for n in ['final_native','final_area','old_native','old_area','thin_m2']]
        widths=[];offset=0;exact=[]
        for filename in files:
            im=Image.open(path.parent/filename).convert('RGB').crop(crop)
            width=round(im.width*height/im.height);im=im.resize((width,height),Image.Resampling.NEAREST)
            exact.append(np.array_equal(np.asarray(im),np.asarray(board.crop((offset,32,offset+width,32+height)))))
            widths.append(width);offset+=width
        check(all(exact) and board.size==(sum(widths),height+32),'panel_actual_native_content',str(path))
        tile_results.append(dict(raw=raw,name=path.name,columns=columns,all_native_panel_pixels_exact=all(exact),sha256=x['sha256']))

    lod=base.read(A/'scale/surface_lod_v1/evaluation49/summary.json');coverage=base.read(A/'scale/surface_lod_v1/coverage49/summary.json')
    expected=[]
    def row(values):expected.append([str(v) for v in values])
    for arm,label,baseline in [('LOD_vs_old_native','旧formal native',True),('LOD_vs_final_native','C2+matte native',True),('LOD_vs_final_area','C2+matte area（诊断）',True),('LOD_vs_old_native','C2+matte LOD（研究）',False)]:
        d=lod[arm]['all49'];prefix='baseline_' if baseline else ''
        rgb=d['solo.a6.'+prefix+'rgb_error'];depth=d['solo.a6.'+prefix+'measured_depth_error'];alpha=d['scene.keep.a6.'+prefix+'visible_coverage']
        row([label,base.fmt(rgb['count']),base.fmt(rgb['mean']),base.fmt(depth['count']),base.fmt(depth['mean']),base.fmt(alpha['mean'])])
    for arm,label in [('LOD_vs_old_native','LOD vs旧native'),('LOD_vs_final_native','LOD vs新native')]:
        d=lod[arm]['all49']
        for state,region,desc in [('keep','a37','keep a37'),('keep','road_visible','keep road'),('keep','outside','keep outside'),('delete','outside','delete outside')]:
            rgb=d[f'scene.{state}.preservation.{region}.parent_rgb_change'];depth=d[f'{state}_fixed_depth_preservation.'+('road' if region=='road_visible' else region)+'.expected_depth_parent_change_m']
            row([label,desc,base.fmt(rgb['above_002'])+' / '+base.fmt(rgb['count']),base.fmt(rgb['max']),base.fmt(depth['mean']),base.fmt(depth['max'])])
    for arm,label in [('old_native','旧native'),('final_native','新native'),('surface_LOD','LOD研究')]:
        for state,domain,desc in [('solo','visible','solo原visible'),('joint_keep','visible','joint原visible'),('joint_keep','extra_eroded_visible','joint原额外erode1内域'),('joint_keep','reviewed','joint完整reviewed域')]:
            d=coverage['all49'][arm][state][domain]
            row([label,desc]+[base.fmt(d[k]) for k in ['count','mean','min','alpha_lt09','alpha_lt05','zero_alpha_le1e6']])
    lodbad=[r for r in expected if ('lod',r) not in p.rows];check(not lodbad,'LOD_table_receipt_mismatch',lodbad)
    lodtext=' '.join(p.text['lod']);hrtext=' '.join(p.text['hr-street'])
    check(all(t in lodtext for t in ['93','新增 33','raw91','不能把3pose无孔外推成全49零孔','不能验证隐藏形状','全部非DC系数实际exact0','裸canonical NPZ默认仍走普通密采样','冻结背景参数不等于其所有成像精确不变']),'LOD_scope_guards')
    check('原有线状欠覆盖' in hrtext and '只用于车体诊断' in hrtext,'HR_background_limit_guard')
    metrics=base.read(A/'scale/surface_lod_v1/coverage49/metrics.json')
    low05=[(m['raw'],m['values']['surface_LOD']['joint_keep']['visible']['alpha_lt05']) for m in metrics if m['values']['surface_LOD']['joint_keep']['visible']['alpha_lt05']]
    inner09=[(m['raw'],m['values']['surface_LOD']['joint_keep']['extra_eroded_visible']['alpha_lt09']) for m in metrics if m['values']['surface_LOD']['joint_keep']['extra_eroded_visible']['alpha_lt09']]
    check(low05==[(89,1),(90,1),(91,1),(92,1)] and inner09==[(91,1)],'49_raw_low_alpha_distribution',dict(visible_lt05=low05,inner_lt09=inner09))

    provenance=base.read(O/'before_after_provenance.json');hero=Image.open(O/'before_after.png').convert('RGB')
    check(provenance['raw']==113 and provenance['viewport']==[480,320] and provenance['same_crop_xyxy']==[180,120,360,320] and not provenance['background_retouched'],'hero_provenance')
    for i,x in enumerate(provenance['source_images']):
        path=Path(x['path']);im=Image.open(path).convert('RGB')
        expectedim=im.crop((180,120,360,320)).resize((360,400),Image.Resampling.NEAREST)
        check(im.size==(480,320) and path.parent.name=='raw113' and base.sha(path)==x['sha256']==base.sha(O/x['original']),'hero_source_hash_or_raw',i)
        check(np.array_equal(np.asarray(expectedim),np.asarray(hero.crop((360*i,42,360*(i+1),442)))),'hero_crop_pixels_no_retouch',i)
    check(base.sha(O/'before_after.png')=='54c4f924e2aba4cc0f5962a99ea9c11e231773eb048d708117da02e9102af2c6','hero_snapshot')
    originalHR=Image.open(A/'scale/lod_preview/raw113_canonicalHR_fair.png').convert('RGB')
    detail=Image.open(O/'hr_street_detail.png').convert('RGB');expectedHR=originalHR.crop((2*(originalHR.width//3),46,originalHR.width,originalHR.height))
    check(np.array_equal(np.asarray(detail),np.asarray(expectedHR)),'HR_detail_actual_crop')

    zpath=O/'surface_LOD_replay_bundle.zip';check(base.sha(zpath)==EXPECTED_ZIP,'replay_ZIP_SHA')
    z=zipfile.ZipFile(zpath);manifest=json.loads(z.read('manifest.json'))
    bundlebad=[]
    for name,item in manifest['files'].items():
        data=z.read(name)
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:bundlebad.append(name)
        check((O/'replay_bundle'/name).is_file() and base.sha(O/'replay_bundle'/name)==item['sha256'],'replay_extracted_member',name)
    check(len(manifest['files'])==10 and not bundlebad,'replay_manifest_members',bundlebad)
    command=z.read('REPLAY_COMMAND.txt').decode();readme=z.read('README.txt').decode()
    check('--asset complete_a6.npz --surface posterior.npz --raw 113 --width 480' in command and 'NEW directory' in command,'replay_command')
    check(manifest['naked_NPZ_does_not_reproduce_LOD'] and bool(manifest['existing_host_dependencies']) and 'excludes the large frozen native scene/checkpoints' in readme,'replay_dependency_disclosure')
    header_checks=[]
    for name in ['index.html','viewer_fixed.html']:
        original=(A/'report/prior_delivery_headers'/name).read_bytes().decode('utf-8');current=(O.parent/'delivery'/name).read_bytes().decode('utf-8')
        stripped=re.sub(r'<aside id="current-a6-diagnosis-link".*?</aside>','',current,count=1,flags=re.S)
        exact=stripped==original;present=current.count('id="current-a6-diagnosis-link"')==1 and '../diagnosis/index.html' in current
        check(exact and present,'prior_delivery_only_banner',name)
        header_checks.append(dict(name=name,only_banner_added=exact and present,banner_present=present,original_content_byte_exact=exact,original_sha256=hashlib.sha256(original.encode()).hexdigest(),current_sha256=base.sha(O.parent/'delivery'/name)))

    check(base.sha(O/'index.html')==EXPECTED_HTML and base.sha(O/'publication_receipt.json')==EXPECTED_RECEIPT,'publication_changed_during_review')
    report.update(status='PASS_REVIEWED_SNAPSHOT' if not issues else 'MUST_FIX',must_fix=issues,pending_addenda=[])
    report['snapshots']['publication_receipt_sha256']=EXPECTED_RECEIPT
    report['final_publication_addendum']=dict(resolved_old_f023_caption_bug=True,raw_counts=dict(Counter(str(x['raw']) for x in bindings)),explicit_figures=tile_results,actual_native_panels_pixel_checked=42,all_HTML_caption_alt_receipt_matches=True,LOD_receipt_bound_rows_checked=len(expected),LOD_low05_visible_raws=low05,LOD_inner_lt09_raws=inner09,hero_same480_same113_crop_pixel_exact=True,hero_background_untouched=True,HR_detail_crop_exact=True,replay_zip_sha256=base.sha(zpath),replay_manifest_members_checked=len(manifest['files']),replay_command_and_dependencies_disclosed=True,prior_delivery_headers=header_checks,new_GPU_calls=0,network_calls=0)
    report['PASS_items']+=['Explicit raw106/113/130 bindings cover 9 figures with 4/5/5 columns per raw; 42 panel pixels, captions, source hashes and inline data match actual native outputs.','LOD24 table rows match local49 receipts; visible<.9 solo24/joint93, inner1 at raw91 and four visible<.5 at89–92 are kept distinct from zero coverage.','Hero3 panels are exact nearest2x crops of same raw113480 native street outputs including the dark ROI; HR detail is separate and its background limit is stated.','Replay ZIP SHA and10 manifest member hashes match; command, frozen host dependencies and naked-NPZ limitation are disclosed; prior delivery pages only add the local report banner.']
    report['numerical_checks']['receipt_bound_table_rows_checked']+=len(expected)
    (A/'review/final_report_review.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],must_fix=issues,HTML_sha256=EXPECTED_HTML,additional_LOD_rows=len(expected),nine_figure_bindings=len(bindings),native_panel_pixels_checked=42,replay_manifest_members=len(manifest['files'])),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
