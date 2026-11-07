"""打包离线报告、目标构造证据与实际写回状态；完整目标另有精确私有备份。"""
from pathlib import Path
import argparse,hashlib,json,shutil,tarfile
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);args=ap.parse_args();out=args.report
    assert (out/'index.html').is_file() and not (out/'publication_manifest.json').exists()
    engineering_allow=set(json.loads((RUN/'engineer/engineering_core_manifest.json').read_text())['files'])
    engineering_allow.update(('engineering_core_manifest.json','lean_backup_list.json'))
    copies={}
    def copy(p,rel):
        dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert dst.read_bytes()==p.read_bytes(),rel
        else:shutil.copy2(p,dst)
        copies[rel]=str(p)
    for name in ('root','contract','engineer','targets','targets_edgeblend_v1','red_review'):
        for p in (RUN/name).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or 'baseline_cache' in p.parts:continue
            if name=='engineer' and p.relative_to(RUN/'engineer').as_posix() not in engineering_allow:continue
            if p.suffix=='.npz' and 'parity' in p.parts:continue
            if p.suffix=='.npz' and any(q in p.parts for q in ('projected_BUILD40','projected_HELD9')):continue
            if p.suffix not in ('.py','.json','.md','.svg','.pt','.npz','.png'):continue
            # 结果RGB/depth大浮点仍保留远端；训练目标/Adam/固定mask/atlas均保留。
            assert p.stat().st_size<40_000_000,(str(p),p.stat().st_size)
            rel='repro/'+p.relative_to(RUN).as_posix();copy(p,rel)
    copy(RUN/'round_decision.json','repro/round_decision.json')
    copy(RUN/'results/full49/receipt.json','repro/results/full49_receipt.json')
    copy(RUN/'results/edgeblend49/receipt.json','repro/results/edgeblend49_receipt.json')
    (out/'repro/README.md').write_text('''# 本轮重放文件

共享atlas、固定mask、未采用对照、实际源代码、Adam/RNG/sampler检查点均在对应子目录。
49帧完整目标含实拍副本，已另取回185.8MB精确私有包；C新增数组也已由其离线精确恢复核验，见root/private_target_custody.json。
轻量报告不重复嵌入该目标图大数组；实际manifest、构造源码、审计、atlas与状态保留，远端原49文件不变。
完整原生10K/旧道路1K/车辆资产及原始传感输入仍保存在AutoDL，具体依赖以manifest为准。
最终推理只读持久三维道路材质/几何，不读融合目标图，也不做屏幕空间覆盖。
本包不是全部大模型/输入/浮点渲染现场的备份。
''')
    manifest={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file()}
    (out/'publication_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    archive=RUN/(out.name+'.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname=out.name)
    receipt=dict(archive=str(archive),prefix=out.name,bytes=archive.stat().st_size,sha256=sha(archive),files=len(manifest)+1,
        original_models_backed_up_here=False,AutoDL='KEEP_ON')
    (RUN/'delivery_packet_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
