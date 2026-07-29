import numpy as np

from motion_proj.dynamic_recon.pseudo_tracks import (
    PseudoTrackConfig,
    audit_mask_id_continuity,
    read_scalar_vertex_ply,
)


def test_protocol_fingerprint_changes_with_threshold():
    assert PseudoTrackConfig().fingerprint() != PseudoTrackConfig(
        min_support_frames=21
    ).fingerprint()


def test_frozen_mask_audit_does_not_invent_vehicle_class(tmp_path):
    semantic = tmp_path / "semantic"
    semantic.mkdir()
    for frame in range(60):
        for camera in range(3):
            mask = np.zeros((30, 30), dtype=np.uint16)
            if camera == 1 and frame < 20:
                mask[:25, :25] = 9
            np.save(semantic / f"mask_{frame * 3 + camera:06d}.npy", mask)
    audit = audit_mask_id_continuity(tmp_path)
    assert audit["max_support_frames"] == 20
    assert audit["continuity_eligible_without_class_check_count"] == 1
    assert audit["class_label_available_in_frozen_artifact"] is False
    assert audit["vehicle_eligible_count"] == 0


def test_read_scalar_vertex_ply_reads_single_ascii_row(tmp_path):
    path = tmp_path / "point_cloud.ply"
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float obj\n"
        "end_header\n"
        "1.25 1\n"
    )
    vertices = read_scalar_vertex_ply(path)
    assert vertices.shape == (1,)
    assert np.isclose(vertices["x"][0], 1.25)
    assert np.isclose(vertices["obj"][0], 1.0)
