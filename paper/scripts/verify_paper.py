"""核对论文引用、数值表与排版日志；不访问数据集或执行模型。"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def expand(path: Path, stack: tuple[Path, ...] = ()) -> str:
    if path in stack:
        raise ValueError(f"循环引用: {path}")
    source = path.read_text(encoding="utf-8")
    source = re.sub(r"(?<!\\)%[^\n]*", "", source)

    def replace(match: re.Match[str]) -> str:
        target = ROOT / match[1]
        if not target.suffix:
            target = target.with_suffix(".tex")
        return expand(target, stack + (path,))

    return re.sub(r"\\input\{([^}]+)\}", replace, source)


def main() -> None:
    bibliography = set(re.findall(r"@\w+\{([^,]+),", (ROOT / "bibliography.bib").read_text(encoding="utf-8")))
    reports = {}
    errors = []
    for name in ("main", "supplement"):
        source = expand(ROOT / f"{name}.tex")
        labels = Counter(re.findall(r"\\label\{([^}]+)\}", source))
        refs = set(re.findall(r"\\(?:eqref|ref|autoref|cref)\{([^}]+)\}", source))
        citations = {key.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", source) for key in group.split(",")}
        missing_refs = sorted(refs - labels.keys())
        duplicate_labels = sorted(key for key, count in labels.items() if count > 1)
        missing_citations = sorted(citations - bibliography)
        log = (ROOT / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        layout = re.findall(r"Overfull [^\n]+", log)
        unresolved = re.findall(r"[^\n]*(?:undefined|multiply defined|multiply-defined)[^\n]*", log)
        reports[name] = {
            "labels": len(labels), "citations": len(citations),
            "missing_refs": missing_refs, "duplicate_labels": duplicate_labels,
            "missing_citations": missing_citations, "overfull": layout, "unresolved": unresolved,
        }
        if missing_refs or duplicate_labels or missing_citations or layout or unresolved:
            errors.append(name)

    evidence = json.loads((ROOT / "results/eas_evidence.json").read_text(encoding="utf-8"))
    geometry = (ROOT / "tables/eas_geometry.tex").read_text(encoding="utf-8")
    returns = (ROOT / "tables/eas_return.tex").read_text(encoding="utf-8")
    g = evidence["m8"]["train_holdout"]
    m = evidence["m8"]["m7_reference"]
    expected_geometry = []
    for record, prefix in ((g, "baseline"), (m, "output"), (g, "output")):
        expected_geometry.extend(f'{100 * record[s][prefix + "_early_rate"]:.2f}' for s in ("all", "hazard", "clear"))
        expected_geometry.extend((f'{1000 * record[prefix + "_mean_chamfer_m"]:.2f}', f'{100 * record[prefix + "_hit_recall"]:.2f}'))
    if any(value not in geometry for value in expected_geometry):
        errors.append("geometry numeric table")
    expected_return = []
    for arm, key in ((evidence["m39"]["metrics"], "m38"), (evidence["m43"]["m39_surface_return"], "m39")):
        for stratum in ("all", "hazard", "clear"):
            row = arm[stratum]
            expected_return.extend(f'{100 * row[k]:.2f}' for k in ("baseline_early_rate", key + "_early_rate", "baseline_hit_rate", key + "_hit_rate"))
    if any(value not in returns for value in expected_return):
        errors.append("categorical numeric table")
    for arm in (evidence["m39"]["metrics"], evidence["m43"]["m39_surface_return"]):
        assert arm["all"]["ray_count"] == arm["hazard"]["ray_count"] + arm["clear"]["ray_count"]
        assert arm["all"]["actor_count"] == arm["hazard"]["actor_count"] + arm["clear"]["actor_count"]
    reports["numeric_checks"] = {"geometry_values": len(expected_geometry), "return_values": len(expected_return), "cohort_sums": "pass"}
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit("核对未通过: " + ", ".join(errors))


if __name__ == "__main__":
    main()
