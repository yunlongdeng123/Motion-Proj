#!/usr/bin/env bash

# V2 统一的缓存、镜像和轻量环境审计入口；本脚本不安装任何依赖。

_dr_v2_probe_url() {
  local name="$1"
  local url="$2"
  local timeout="$3"
  local output="$4"
  local code="000"
  local rc=127
  if command -v curl >/dev/null 2>&1; then
    if code="$(curl -L -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout "$timeout" --max-time "$timeout" "$url")"; then
      rc=0
    else
      rc=$?
    fi
  fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$url" "$code" "$rc" >> "$output"
}

_dr_v2_bootstrap_main() {
  local report_dir=""
  local network_timeout=20
  local python_cmd=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --report-dir)
        [[ $# -ge 2 ]] || { printf '%s\n' "--report-dir 缺少路径" >&2; return 2; }
        report_dir="$2"
        shift 2
        ;;
      --network-timeout)
        [[ $# -ge 2 ]] || { printf '%s\n' "--network-timeout 缺少秒数" >&2; return 2; }
        network_timeout="$2"
        shift 2
        ;;
      -h|--help)
        printf '%s\n' "用法: source scripts/bootstrap_autodl_v2.sh [--report-dir DIR] [--network-timeout SEC]"
        return 0
        ;;
      *)
        printf '%s\n' "未知参数: $1" >&2
        return 2
        ;;
    esac
  done

  export PROJECT_ROOT=/root/autodl-tmp/motion_proj
  export ENV_ROOT=/root/autodl-tmp/envs
  export CACHE_ROOT=/root/autodl-tmp/cache
  export CONDA_PKGS_DIRS=/root/autodl-tmp/cache/conda-pkgs
  export HF_HOME=/root/autodl-tmp/hf_cache
  export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
  export HF_ENDPOINT=https://hf-mirror.com
  export TORCH_HOME=/root/autodl-tmp/cache/torch
  export XDG_CACHE_HOME=/root/autodl-tmp/cache/xdg
  export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip
  export TMPDIR=/root/autodl-tmp/tmp
  export CONDARC="$PROJECT_ROOT/configs/env/autodl_condarc_v2.yaml"
  export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  export PIP_DEFAULT_TIMEOUT=120
  export PIP_RETRIES=3

  mkdir -p \
    "$ENV_ROOT" "$CACHE_ROOT" "$CONDA_PKGS_DIRS" "$HF_HOME" \
    "$HF_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" \
    "$PIP_CACHE_DIR" "$TMPDIR"

  [[ -f "$CONDARC" ]] || { printf '%s\n' "缺少项目 Conda 配置: $CONDARC" >&2; return 1; }
  [[ "$CONDARC" == "$PROJECT_ROOT"/* ]] || { printf '%s\n' "CONDARC 必须位于项目内" >&2; return 1; }

  # 显式加载 conda shell 函数，不执行 conda init，也不改写用户配置。
  if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
  fi
  if command -v python >/dev/null 2>&1; then
    python_cmd="$(command -v python)"
  elif [[ -x /root/miniconda3/bin/python ]]; then
    python_cmd=/root/miniconda3/bin/python
  else
    printf '%s\n' "未找到可用 Python" >&2
    return 1
  fi

  if [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
    export DR_V2_NETWORK_TURBO=available_and_sourced
  else
    export DR_V2_NETWORK_TURBO=unavailable
  fi

  if [[ -z "$report_dir" ]]; then
    report_dir="$TMPDIR/motion_proj_bootstrap_v2"
  fi
  mkdir -p "$report_dir"

  {
    printf 'PROJECT_ROOT=%s\n' "$PROJECT_ROOT"
    printf 'ENV_ROOT=%s\n' "$ENV_ROOT"
    printf 'CACHE_ROOT=%s\n' "$CACHE_ROOT"
    printf 'CONDA_PKGS_DIRS=%s\n' "$CONDA_PKGS_DIRS"
    printf 'HF_HOME=%s\n' "$HF_HOME"
    printf 'HF_HUB_CACHE=%s\n' "$HF_HUB_CACHE"
    printf 'HF_ENDPOINT=%s\n' "$HF_ENDPOINT"
    printf 'TORCH_HOME=%s\n' "$TORCH_HOME"
    printf 'XDG_CACHE_HOME=%s\n' "$XDG_CACHE_HOME"
    printf 'PIP_CACHE_DIR=%s\n' "$PIP_CACHE_DIR"
    printf 'TMPDIR=%s\n' "$TMPDIR"
    printf 'CONDARC=%s\n' "$CONDARC"
    printf 'PIP_INDEX_URL=%s\n' "$PIP_INDEX_URL"
    printf 'DR_V2_NETWORK_TURBO=%s\n' "$DR_V2_NETWORK_TURBO"
  } > "$report_dir/exported_environment.txt"

  "$python_cmd" --version > "$report_dir/python_version.txt" 2>&1 || true
  conda --version > "$report_dir/conda_version.txt" 2>&1 || true
  "$python_cmd" -m pip --version > "$report_dir/pip_version.txt" 2>&1 || true
  "$python_cmd" -m pip config debug > "$report_dir/pip_config_debug.txt" 2>&1 || true
  conda config --show-sources > "$report_dir/conda_sources.txt" 2>&1 || true
  git --version > "$report_dir/git_version.txt" 2>&1 || true

  local connectivity="$report_dir/connectivity.tsv"
  printf 'name\turl\thttp_code\treturn_code\n' > "$connectivity"
  _dr_v2_probe_url "tuna_conda" \
    "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/" \
    "$network_timeout" "$connectivity"
  _dr_v2_probe_url "tuna_pypi" \
    "https://pypi.tuna.tsinghua.edu.cn/simple/pip/" \
    "$network_timeout" "$connectivity"
  _dr_v2_probe_url "hf_mirror" \
    "https://hf-mirror.com/" \
    "$network_timeout" "$connectivity"
  _dr_v2_probe_url "github_official" \
    "https://github.com/" \
    "$network_timeout" "$connectivity"

  "$python_cmd" - "$report_dir" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

report = Path(sys.argv[1])
with (report / "connectivity.tsv").open(encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
for row in rows:
    row["return_code"] = int(row["return_code"])
    row["reachable"] = row["return_code"] == 0 and row["http_code"] not in {"000", ""}
payload = {
    "schema_version": 1,
    "conda": {
        "scope": "project",
        "config": os.environ["CONDARC"],
        "primary": "TUNA",
    },
    "pip": {
        "scope": "process",
        "index_url": os.environ["PIP_INDEX_URL"],
        "primary": "TUNA",
    },
    "huggingface": {
        "endpoint": os.environ["HF_ENDPOINT"],
        "cache": os.environ["HF_HUB_CACHE"],
        "revision_policy": "fixed",
    },
    "github": {
        "primary": "official_with_autodl_network_turbo",
        "revision_policy": "fixed_commit",
    },
    "connectivity": rows,
    "global_config_modified": False,
}
(report / "source_resolution.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

  printf '%s\n' "V2 bootstrap 完成，报告目录: $report_dir"
}

_dr_v2_bootstrap_main "$@"
_dr_v2_bootstrap_rc=$?
unset -f _dr_v2_bootstrap_main _dr_v2_probe_url
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  exit "$_dr_v2_bootstrap_rc"
fi
return "$_dr_v2_bootstrap_rc"
