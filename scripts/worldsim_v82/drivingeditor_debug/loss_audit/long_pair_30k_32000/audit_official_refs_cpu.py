"""Read all public official Git heads/tags; no fetch, checkout, issue or message."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


OFFICIAL = "https://github.com/WangXu-xxx/DrivingEditor.git"
PRIOR_MAIN = "560d57c3825ed2bdbc6a6602e0011ad51e8da60e"


def main():
    command = ["git", "ls-remote", "--heads", "--tags", OFFICIAL]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        sha, ref = line.split("\t", 1)
        rows.append({"ref": ref, "object_id": sha})
    heads = [row for row in rows if row["ref"].startswith("refs/heads/")]
    tags = [row for row in rows if row["ref"].startswith("refs/tags/")]
    report = {
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "official_repository": OFFICIAL,
        "command_argv": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "all_public_heads": heads,
        "all_public_tags": tags,
        "head_count": len(heads),
        "tag_count_including_peeled_refs": len(tags),
        "main_matches_previously_audited_commit": any(
            row == {"ref": "refs/heads/main", "object_id": PRIOR_MAIN}
            for row in heads
        ),
        "new_public_refs_requiring_source_inspection": [
            row for row in rows
            if row != {"ref": "refs/heads/main", "object_id": PRIOR_MAIN}
        ],
        "conclusion_scope": "Public advertised heads/tags at this timestamp only; no claim about private or unpublished author code.",
        "source_downloaded": False,
        "authors_contacted": False,
        "GPU_used": False,
    }
    path = Path(__file__).with_name("official_git_refs.json")
    path.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({"heads": heads, "tag_count": len(tags), "report": str(path)}))


if __name__ == "__main__":
    main()
