import io
import tarfile

from scripts.build_adgs_nuscenes_assets import _scan_one_shard


def test_scan_one_shard_extracts_required_member_and_ignores_other(tmp_path) -> None:
    archive = tmp_path / "v1.0-trainval03_blobs.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, payload in (
            ("./samples/ignored.bin", b"ignored"),
            ("./samples/required.bin", b"required"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    destination = tmp_path / "output"
    rows = _scan_one_shard(
        (str(archive), {"samples/required.bin"}, str(destination))
    )
    assert rows == {
        "samples/required.bin": {
            "shard": archive.name,
            "extracted": True,
        }
    }
    assert (destination / "samples/required.bin").read_bytes() == b"required"
    assert not (destination / "samples/ignored.bin").exists()
