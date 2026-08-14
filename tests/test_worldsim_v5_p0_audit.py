from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_worldsim_v5_p0 import validate_freeze_file_set  # noqa: E402
from worldsim_v5_forensics_common import ForensicAuditError  # noqa: E402


def test_freeze_commit_file_set_is_ordered_and_exact() -> None:
    expected = ["configs/a.yaml", "docs/a.md", "scripts/a.py", "tests/test_a.py"]
    validate_freeze_file_set(list(expected), expected)
    with pytest.raises(ForensicAuditError, match="extra"):
        validate_freeze_file_set(expected + ["models/changed.py"], expected)
    with pytest.raises(ForensicAuditError, match="missing"):
        validate_freeze_file_set(expected[:-1], expected)
