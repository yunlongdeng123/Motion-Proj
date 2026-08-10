from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from motion_proj.worldsim_v33.source_audit import (
    SCHEMA_VERSION,
    TASK_ID,
    SourceAuditError,
    audit_config,
    run_audit,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _source(**overrides):
    source = {
        "name": "paper-only",
        "official_url": None,
        "paper_url": "https://arxiv.org/abs/example",
        "commit": None,
        "tree_sha": None,
        "license": "unavailable",
        "license_sha256": None,
        "weights": "unavailable",
        "weights_revision": None,
        "weights_sha256": None,
        "python": "unavailable",
        "torch": "unavailable",
        "cuda": "unavailable",
        "single_3090": "not_assessable_without_source",
        "input_schema": "paper_described",
        "output_schema": "paper_described",
        "execution_state": "source_not_released",
        "checkout": None,
        "license_path": None,
    }
    source.update(overrides)
    return source


def _config(tmp_path: Path):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"immutable")
    sources = {f"paper_{index}": _source(name=f"paper-{index}") for index in range(9)}
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": "done",
        "sources": sources,
        "v32_canonical": {
            "assets": {
                "tiny": {
                    "path": str(asset),
                    "bytes": asset.stat().st_size,
                    "sha256": _sha(asset),
                }
            }
        },
        "gates": {
            "v32_canonical_immutable": True,
            "no_training": True,
            "no_model_inference": True,
            "no_large_weight_download": True,
            "s1_authorized": True,
            "s2_authorized": False,
            "s3_authorized": False,
            "s4_authorized": False,
            "s5_authorized": False,
        },
    }


def test_paper_only_sources_and_asset_hash_pass(tmp_path):
    result = audit_config(_config(tmp_path), verify_large_assets=True)
    assert result["source_count"] == 9
    assert result["execution_state_counts"] == {"source_not_released": 9}
    assert result["v32_assets"]["tiny"]["sha256_exact"] is True


def test_git_source_requires_exact_commit_tree_and_license(tmp_path):
    checkout = tmp_path / "repo"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Test"], check=True)
    license_path = checkout / "LICENSE"
    license_path.write_text("Apache-2.0\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "LICENSE"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-q", "-m", "init"], check=True)

    config = _config(tmp_path)
    config["sources"]["paper_0"] = _source(
        name="executable",
        official_url="https://example.com/repo.git",
        commit=_git(checkout, "rev-parse", "HEAD"),
        tree_sha=_git(checkout, "rev-parse", "HEAD^{tree}"),
        license="Apache-2.0",
        license_sha256=_sha(license_path),
        weights="not_required_for_audit",
        python=">=3.10",
        torch=">=2.0",
        cuda=">=11.8",
        single_3090="supported",
        input_schema="images",
        output_schema="masks",
        execution_state="executable",
        checkout=str(checkout),
        license_path="LICENSE",
    )
    result = audit_config(config)
    assert result["sources"]["paper_0"]["commit_exact"] is True
    assert result["sources"]["paper_0"]["license_exact"] is True


def test_missing_required_source_field_fails(tmp_path):
    config = _config(tmp_path)
    del config["sources"]["paper_0"]["input_schema"]
    with pytest.raises(SourceAuditError, match="缺少字段"):
        audit_config(config)


def test_run_directory_is_immutable(tmp_path):
    import yaml

    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_dir = tmp_path / "run"
    project_root = Path(__file__).resolve().parents[1]
    run_audit(
        config_path,
        run_dir,
        verify_large_assets=True,
        project_root=project_root,
    )
    assert (run_dir / "status.json").is_file()
    assert (run_dir / "manifest.json").is_file()
    with pytest.raises(SourceAuditError, match="禁止复用"):
        run_audit(config_path, run_dir, project_root=project_root)
