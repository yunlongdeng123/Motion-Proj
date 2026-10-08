"""R001 评分伪标签静态窗容量训练；复用原生主干及 train_one 的完整训练合同。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from prepare_scored_pairs import checked_score


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(source: Path, out: Path):
    source = source.resolve()
    manifest, selection = read(source / "manifest.json"), read(source / "selection.json")
    accepted = selection["selected"]
    expected = [r["candidate_id"] for r in manifest if checked_score(r["score"]) > 1]
    if selection.get('eligible') != expected or selection.get("rule") != "score>1":
        raise ValueError("筛选清单与严格 score>1 规则不一致")
    if selection.get('sampling_strategy')!='best1_per_case' or len(accepted)!=1 or not set(accepted).issubset(expected):
        raise ValueError('第一关仅允许从准入池显式选一张，禁止将准入池全部入训')
    choice = selection['selection_review']
    if accepted != [choice.get('selected_candidate_id')]:
        raise ValueError('实际入训候选与独立选择记录不一致')
    if not accepted:
        raise ValueError("R001 的 score>1 候选为 0；保留全部评分，停止训练")
    by_id = {r["candidate_id"]: r for r in manifest}
    if len(by_id) != len(manifest):
        raise ValueError("候选 ID 重复")
    pairs = []
    for cid in accepted:
        path = source / "pairs" / cid / "pair.json"
        pair = read(path)
        row, qa = by_id[cid], read(Path(pair["qa_record"]))
        if row['source_frame_index'] != choice.get('selected_source_frame_index'):
            raise ValueError('实际入训来源帧与独立选择记录不一致')
        if (pair.get("candidate_id") != cid or pair.get("source_frame_index") != row["source_frame_index"]
            or pair.get("review_slot") != 5 or pair.get("static_repeat_capacity_only") is not True
            or pair.get("supervision_kind") != "reviewed_pseudo_clean_target"
            or checked_score(pair.get("score")) != row["score"] or checked_score(qa.get("score")) != row["score"]):
            raise ValueError(f"{cid}: pair、QA 与原评分清单不一致")
        pairs.append(path)
    import train_one as core
    original_load = core.load_pair
    patched_train = []

    def scored_load(_path, size):
        pair, prepared, train = original_load(pairs[0], size)
        pair["full_scored_manifest"] = manifest
        pair["selection"] = selection
        pair["training_mode"] = "static_repeat_capacity_only"
        pair["step_to_candidate"] = [accepted[i % len(accepted)] for i in range(64)]
        original_loss = train.loss

        def rotating_loss(model, _prepared, seed, protected=None):
            index = (seed - 6201) % len(pairs)
            return original_loss(model, original_load(pairs[index], size)[1], seed, protected)

        train.loss = rotating_loss
        patched_train[:] = [train, original_loss]
        return pair, prepared, train

    core.load_pair = scored_load
    try:
        core.run(SimpleNamespace(pair=pairs[0], outdir=out, steps=64, size=[320, 576], prepare=False, probe=False))
    finally:
        core.load_pair = original_load
        if patched_train:
            patched_train[0].loss = patched_train[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    run(args.prepared, args.outdir)
