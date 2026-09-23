from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_http_ranges.py"
SPEC = spec_from_file_location("download_http_ranges", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_iter_ranges_covers_interval_without_overlap() -> None:
    assert list(MODULE.iter_ranges(5, 17, 4)) == [
        (5, 8),
        (9, 12),
        (13, 16),
    ]


def test_iter_ranges_handles_short_tail_and_empty_interval() -> None:
    assert list(MODULE.iter_ranges(10, 13, 8)) == [(10, 12)]
    assert list(MODULE.iter_ranges(10, 10, 8)) == []


def test_with_query_value_preserves_existing_query() -> None:
    result = MODULE.with_query_value(
        "https://example.test/download?id=file&confirm=t",
        "range_attempt",
        "100-2",
    )
    assert result == (
        "https://example.test/download?id=file&confirm=t&range_attempt=100-2"
    )


def test_group_contiguous_ranges_respects_holes_and_limit() -> None:
    ranges = [(0, 7), (8, 15), (16, 23), (32, 39), (40, 47)]
    assert MODULE.group_contiguous_ranges(ranges, 2) == [
        ((0, 7), (8, 15)),
        ((16, 23),),
        ((32, 39), (40, 47)),
    ]
