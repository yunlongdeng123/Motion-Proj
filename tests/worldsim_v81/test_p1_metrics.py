"""P1 评测输入安全性与官方指标接口的 CPU 测试。"""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "worldsim_v81" / "evaluate_p1.py"
SPEC = importlib.util.spec_from_file_location("evaluate_p1", SCRIPT)
assert SPEC and SPEC.loader
eval_p1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_p1)


def make_manifest(tmp_path: Path):
    video = tmp_path / "placeholder.mp4"
    video.write_bytes(b"dummy")
    cases = []
    for ratio in (0.25, 0.66):
        cases.append({"dataset": "davis2017", "sequence_id": "bear",
                      "source_id": "davis/bear", "mask_total_ratio": ratio,
                      "gt": str(video), "pred": str(video), "comp": str(video)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return manifest, cases


def test_rejects_source_overlap_and_incomplete_ratio_pair(tmp_path):
    manifest, cases = make_manifest(tmp_path)
    with pytest.raises(ValueError, match="重叠"):
        eval_p1.load_manifest(manifest, {"davis/bear"})
    assert len(eval_p1.load_manifest(manifest, {"youtube/train/foo"})) == 2
    manifest.write_text(json.dumps({"cases": cases[:1]}), encoding="utf-8")
    with pytest.raises(ValueError, match="序列集合不同"):
        eval_p1.load_manifest(manifest, set())


def test_paper_youtube_inventory_is_exact_and_no_extra_id(tmp_path):
    assert len(eval_p1.YOUTUBE_IDS) == 60
    manifest, cases = make_manifest(tmp_path)
    for case in cases:
        case["dataset"] = "youtube_vos"
        case["sequence_id"] = "selected-looking-video"
    manifest.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    with pytest.raises(ValueError, match="附录E"):
        eval_p1.load_manifest(manifest, set())


@pytest.mark.parametrize("missing", ["davis2017", "youtube_vos"])
def test_formal_benchmark_rejects_missing_whole_dataset(monkeypatch, missing):
    davis = {f"davis_{index:02d}" for index in range(90)}
    monkeypatch.setattr(eval_p1, "davis_ids", lambda: davis)
    cases = [{"dataset": dataset, "sequence_id": sequence, "mask_total_ratio": ratio}
             for dataset, sequences in (("davis2017", davis), ("youtube_vos", eval_p1.YOUTUBE_IDS))
             for sequence in sequences for ratio in eval_p1.RATIOS]
    eval_p1.require_complete_benchmark(cases)
    incomplete = [case for case in cases if case["dataset"] != missing]
    # 留下的整套基准本身完整，旧any(coverage.values())会错误放行。
    assert all(row["complete"] for row in eval_p1.coverage(incomplete).values())
    with pytest.raises(ValueError, match="基准序列不完整"):
        eval_p1.require_complete_benchmark(incomplete)


def test_davis_inventory_matches_official_2017_trainval(tmp_path):
    expected = eval_p1.davis_ids()
    if expected is None:
        pytest.skip("本环境未准备官方DAVIS ImageSets")
    assert len(expected) == 90
    manifest, cases = make_manifest(tmp_path)
    for case in cases:
        case["sequence_id"] = "not-a-davis-sequence"
    manifest.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    with pytest.raises(ValueError, match="官方2017"):
        eval_p1.load_manifest(manifest, set())


def test_missing_i3d_is_explicit_and_ratio_means_are_separate(tmp_path, monkeypatch):
    _, cases = make_manifest(tmp_path)
    def result(real, fake):
        return {"value": {i: float(i == 0) for i in range(16)}}
    def lpips(real, fake, device):
        return {"value": {i: 0.25 for i in range(16)}}
    monkeypatch.setattr(eval_p1, "official_metrics", lambda _: (result, result, lpips, None))
    monkeypatch.setattr(eval_p1, "read_first_16", lambda *_, **__: object())
    output = eval_p1.evaluate(cases, tmp_path, None, "cpu")
    assert output["protocol_verified"] is False
    assert output["fvd_status"] == "missing_i3d_torchscript_pt"
    pred = output["groups"]["davis2017"]["pred"]
    assert set(pred["by_total_mask_ratio"]) == {"0.25", "0.66"}
    assert pred["mean_of_ratios"]["psnr"] == pytest.approx(1 / 16)
    assert pred["mean_of_ratios"]["lpips"] == pytest.approx(0.25)
    assert pred["mean_of_ratios"]["fvd"] is None
    assert len(output["cases"]) == 4  # 两倍率 × 原生/硬合成
    assert output["coverage"]["davis2017"] == {"n_sequences": 1, "expected": 90,
                                                 "inventory_verified": False, "complete": False}


def test_official_metric_sources_import_at_fixed_revision():
    if not eval_p1.DEFAULT_FYC.is_dir():
        pytest.skip("此环境未准备固定 Follow-Your-Canvas 指标源码")
    functions = eval_p1.official_metrics(eval_p1.DEFAULT_FYC)
    assert len(functions) == 4 and all(callable(fn) for fn in functions)
