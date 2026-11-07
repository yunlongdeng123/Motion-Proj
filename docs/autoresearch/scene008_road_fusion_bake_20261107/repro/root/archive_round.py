"""归档自己的结果并更新同一份当前快照，保留其它研究分支。"""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
ARCH=REPO/'docs/autoresearch/scene008_road_fusion_bake_20261107'


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);args=ap.parse_args()
    decision=json.loads((RUN/'round_decision.json').read_text())
    if ARCH.exists():raise FileExistsError(ARCH)
    shutil.copytree(args.report,ARCH)
    (ARCH/'.gitattributes').write_text('* -text -filter whitespace=cr-at-eol,-blank-at-eol,-blank-at-eof\n')
    qa=ARCH/'publication_qa';qa.mkdir()
    for p in (RUN/'publication_qa').glob('*'):
        if p.is_file():shutil.copy2(p,qa/p.name)
    status=REPO/'docs/RESEARCH_STATUS.md';before=status.read_text()
    marker='v44＋a5车顶恢复已完成'
    if marker not in before:raise ValueError('Review latest shared status before updating it; a5 marker changed')
    starting=subprocess.check_output(['git','show','140d3fb03e6ddb9a3b32c029a95ab8a73bebac3a:docs/RESEARCH_STATUS.md'],cwd=REPO,text=True)
    if before[:before.index(marker)]!=starting[:starting.index(marker)]:
        raise ValueError('Another branch changed the current-status prefix; review and preserve it before updating')
    suffix=before[before.index(marker):]
    (qa/'research_status_before.md').write_text(before)
    after='# 当前研究状态\n\n更新：2026-11-07（Asia/Singapore）。\n\n'+decision['research_status'].strip()+'\n\n'+suffix
    status.write_text(after)
    assert status.read_text().endswith(suffix)
    index=REPO/'docs/EXPERIMENTS.md';previous=index.read_text()
    assert previous.startswith('# 实验索引\n') and 'scene008_road_fusion_bake_20261107_r1' not in previous
    index.write_text('# 实验索引\n\n## Scene008道路融合目标与三维写回（2026-11-07）\n\n'+decision['experiment_entry'].strip()+'\n'+previous[len('# 实验索引\n'):])
    card=REPO/'docs/research_failures/entries/V82-F13.md';text=card.read_text()
    heading='## 道路融合目标与三维写回后续证据（2026-11-07）'
    assert heading not in text
    card.write_text(text.rstrip()+'\n\n'+heading+'\n\n'+decision['failure_card'].strip()+'\n')
    receipt=dict(archive=str(ARCH),shared_a5_suffix_bytes=len(suffix.encode()),shared_a5_suffix_SHA256=sha(suffix.encode()),
        preserved_verbatim=True,previous_research_status_saved=True,only_own_packet_and_three_research_docs_changed=True,
        failure_ledger_refs=['V82-F13'],AutoDL='KEEP_ON')
    (RUN/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
