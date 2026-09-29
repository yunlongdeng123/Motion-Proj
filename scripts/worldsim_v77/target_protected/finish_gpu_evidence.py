"""交付后补充实际切窗来源与轻量证据索引；不改变实验结果。"""
import argparse
from pathlib import Path
from geometry_factory import read,dump
from nuscenes.utils.splits import train

def main(root,repo):
    evidence=repo/'docs/autoresearch/worldsim_v77/target_protected_20260929/gpu'
    local_validation=evidence/'local_delivery_validation.json';dump(local_validation,read(local_validation))
    full_candidates=read(root/'native_window_factory/pair_candidates.json')
    lightweight=full_candidates|{'full_remote_evidence':str(root/'native_window_factory/pair_candidates.json'),'selected':[ {k:v for k,v in r.items() if k not in ['frames','ground']}|{'frame_count':len(r['frames'])} for r in full_candidates['selected']]}
    dump(evidence/'native_window_proposals.json',lightweight)
    parent={c['source_id']:c for c in read(root/'expanded_factory/source_manifest.json')['clips']};windows=read(root/'native_window_factory/source_manifest.json')['clips'];checked=0
    for c in windows:
        w=c['window_provenance'];p=parent[w['parent_source_id']];start=w['parent_frame_start'];assert c['scene'] in train and c['scene']==p['scene']
        assert len(c['frames'])==10
        for i,f in enumerate(c['frames']):
            original=p['frames'][start+i]
            for key in ['timestamp','filename','actors','camera_to_world','intrinsics_1024']:assert f[key]==original[key],(c['source_id'],i,key)
            assert f['parent_frame']==original['frame']
            linked=root/'native_window_factory/segmented'/c['source_id']/'sam2_raw'/f'{i:05}.png';assert linked.resolve()==(root/'expanded_factory/segmented'/p['source_id']/'sam2_raw'/f'{start+i:05}.png').resolve();checked+=1
    dump(evidence/'native_window_provenance_validation.json',{'windows_checked':len(windows),'actual_exposures_checked':checked,'RGB_camera_actor_state_exact_parent_slice':True,'SAM2_masks_link_to_exact_parent_frame':True,'all_scenes_in_official_train':True,'copied_exposures_to_fake_longer_duration':False})
    for label,sub in [('P',''),('D','expanded_factory'),('W','native_window_factory')]:
        full={r['case_id']:r for r in read(root/sub/'synthetic_review/synthetic_manifest.json')['clips']};p=evidence/f'{label}_case_index.json';idx=read(p)
        for r in idx['cases']:r['frame_count']=len(full[r['case_id']]['frames'])
        dump(p,idx)
    report=repo/'docs/v77/TARGET_PROTECTED_GPU.md';report.write_text(report.read_text().replace('file:///C:/','C:/'),encoding='utf-8')
    status=repo/'docs/RESEARCH_STATUS.md';status.write_text(status.read_text().replace('49左右全部实际产物','49个全部实际产物'),encoding='utf-8')
    dump(evidence/'code_validation.json',{'pytest_command':'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /root/autodl-tmp/envs/motionproj/bin/python -m pytest -q scripts/worldsim_v77/target_protected/test_data_contract.py scripts/worldsim_v77/target_protected/test_geometry_factory.py','pytest_passed':7,'python_compileall_passed':True,'node_check_local_review_script_exit_code':0,'browser_interaction_verified':False})
    (evidence/'review_link.md').write_text('''# 交付入口

本地最终合成全检：`C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-synthetic/index.html`。

49 case / 11 receiver scenes；28独立通过 / 7拒绝 / 14待定。通过例来自8个receiver scene，共420帧供人工全检。30帧控制与10帧原生窗口分别标注，人工verdict初始为空。不是49个独立合格scene，密集类0、训练准入0。

四列：真实GT Y、合成输入X、A/B/C实例标注、masked-X数据条件。最后一列不是模型生成。网页支持视频同步、逐帧放大、每帧0/1及JSON导出导入。来源预审页仍在邻目录`v77-target-protected/index.html`。

本地3240张预览JPEG实解码、196视频存在/容器头检查；完整196视频实解码在远端完成。JavaScript语法检查通过，未声称浏览器点击播放已实测。

轻量实图：[通过P002](contacts/P002_f15.jpg)、[机盖覆盖D006](contacts/D006_f29.jpg)、[失焦W006](contacts/W006_f05.jpg)、[通过W019](contacts/W019_f05.jpg)、[T028实例污染](contacts/T028_f00_isolated_zoom.png)。所有完整原始与对照见同task/run目录，未删除拒绝例。
''',encoding='utf-8')
    print('WINDOW_PROVENANCE',len(windows),checked,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);a=p.parse_args();main(a.root,a.repo)
