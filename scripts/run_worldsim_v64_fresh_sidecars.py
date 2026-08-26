"""以 V6.3 已验证提取器生成 V6.4 fresh native sidecar。"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from scripts.run_worldsim_v63_p2_native_sidecars import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--maximum-workers", type=int, default=2)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    overlay = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_path = repo_root / overlay["base_config"]
    resolved = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    resolved["schema_version"] = overlay["schema_version"]
    resolved["task_id"] = overlay["task_id"]
    resolved["hypothesis_id"] = overlay["hypothesis_id"]
    resolved["seed"] = int(overlay["seed"])
    resolved["cohorts"] = overlay["cohorts"]
    resolved["resources"].update(overlay["resources"])
    resolved["locks"].update(overlay["locks"])
    resolved["failure_ledger_refs"] = overlay["failure_ledger_refs"]

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".yaml",
            prefix="worldsim_v64_fresh_sidecars_",
            delete=False,
        ) as handle:
            yaml.safe_dump(
                resolved, handle, sort_keys=False, allow_unicode=True
            )
            temporary_path = Path(handle.name)
        summary = run(
            temporary_path,
            repo_root,
            args.run_dir.resolve(),
            ["fresh_fit", "fresh_evaluation"],
            int(args.maximum_workers),
            None,
            None,
        )
        (args.run_dir.resolve() / "resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
