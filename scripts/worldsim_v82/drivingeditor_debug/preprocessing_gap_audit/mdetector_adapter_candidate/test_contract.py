"""Focused CPU failure-case checks; synthetic inputs, no detector/training claims."""
from dataclasses import replace
import unittest
from contract import (BodyPoint, Cloud, ContractError, ExactSynchronizer, FrameKey,
                      PointId, Pose, bind, body_to_world, map_author_output)

IDENTITY = (1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.)


def fixtures(frame_index=0):
    key = FrameKey("a" * 64, frame_index, 1000000000000000 + 100000*frame_index)
    points = tuple(BodyPoint(PointId(key, ret, laser, row, col), i,
                            (float(i+1), 2., 3.), .25, .5, flat)
                   for i, (ret, laser, row, col, flat) in enumerate(
                       ((0,1,0,1,1), (0,2,0,1,-1), (1,1,0,1,9))))
    cloud = Cloud(key, key.frame_timestamp_micros, points,
                  "synthetic coordinate receipt", "synthetic lineage receipt",
                  "synthetic approved time binding", (2,4))
    return cloud, Pose(key, key.frame_timestamp_micros, IDENTITY, "synthetic pose receipt")


class ContractChecks(unittest.TestCase):
    def test_reversed_arrivals_pair_by_identity_and_drain_manifest_order(self):
        c0, p0 = fixtures(); c1, p1 = fixtures(1)
        q = ExactSynchronizer([c0.key, c1.key], c0.reference_time_micros)
        q.add_cloud(c1); q.add_pose(p0)
        self.assertIsNone(q.pop_ready())
        q.add_cloud(c0)
        f0 = q.pop_ready()
        self.assertEqual(f0.cloud.key, f0.pose.key)
        self.assertEqual(f0.detector_time_s, 0.)
        self.assertIsNone(q.pop_ready())
        q.add_pose(p1)
        self.assertEqual(q.pop_ready().detector_time_s, .1)
        q.finish()

    def test_one_microsecond_mismatch_is_rejected(self):
        c, p = fixtures()
        bad_key = replace(p.key, frame_timestamp_micros=p.key.frame_timestamp_micros+1)
        q = ExactSynchronizer([c.key], c.reference_time_micros)
        with self.assertRaises(ContractError): q.add_pose(replace(p, key=bad_key))
        with self.assertRaises(ContractError):
            bind(c, replace(p, reference_time_micros=p.reference_time_micros+1), c.reference_time_micros)

    def test_floating_timestamp_is_not_accepted_as_native_microseconds(self):
        c, p = fixtures()
        with self.assertRaises(ContractError):
            replace(c.key, frame_timestamp_micros=float(c.key.frame_timestamp_micros)).validate()
        with self.assertRaises(ContractError): bind(c, p, float(c.reference_time_micros))

    def test_segment_duplicate_and_missing_pose_fail(self):
        c, p = fixtures(); q = ExactSynchronizer([c.key], c.reference_time_micros)
        q.add_cloud(c)
        with self.assertRaises(ContractError): q.add_cloud(c)
        with self.assertRaises(ContractError): q.finish()
        with self.assertRaises(ContractError):
            q.add_pose(replace(p, key=replace(p.key, segment_sha256="b" * 64)))

    def test_order_and_row_lineage_are_required(self):
        c, p = fixtures()
        with self.assertRaises(ContractError):
            bind(replace(c, points=(c.points[1], c.points[0], c.points[2])), p, c.reference_time_micros)
        duplicate = replace(c.points[1], point_id=c.points[0].point_id)
        with self.assertRaises(ContractError):
            bind(replace(c, points=(c.points[0], duplicate, c.points[2])), p, c.reference_time_micros)

    def test_unresolved_time_units_and_pose_are_blockers(self):
        c, p = fixtures()
        for altered in (replace(c, time_binding_receipt=""),
                        replace(c, coordinate_frame="normalized_scene_xyz")):
            with self.assertRaises(ContractError): bind(altered, p, c.reference_time_micros)
        with self.assertRaises(ContractError):
            bind(c, replace(p, body_to_world_row_major=(2.,)+IDENTITY[1:]), c.reference_time_micros)

    def test_two_returns_same_pixel_remain_distinct(self):
        c, p = fixtures(); f = bind(c, p, c.reference_time_micros)
        a = map_author_output(f, [1,0,0], [1,-1,0], [1,6,0])
        self.assertEqual([x.partition for x in a], ["dynamic", "invalid", "static"])
        self.assertEqual([x.author_binary_label for x in a], [251,9,9])
        self.assertNotEqual(a[0].point_id, a[2].point_id)
        self.assertEqual([x.bin_row for x in a], [0,1,2])
        with self.assertRaises(ContractError): map_author_output(f, [0], [0], [0])
        wrong_return = replace(c.points[2], top_range_flat_index=1)
        with self.assertRaises(ContractError):
            bind(replace(c, points=c.points[:2]+(wrong_return,)), p, c.reference_time_micros)

    def test_metric_body_to_world_rotation_translation(self):
        _, p = fixtures()
        pose = replace(p, body_to_world_row_major=(0.,-1.,0.,10., 1.,0.,0.,20.,
                                                   0.,0.,1.,30., 0.,0.,0.,1.))
        self.assertEqual(body_to_world((1.,2.,3.), pose), (8.,21.,33.))


if __name__ == "__main__": unittest.main(verbosity=2)
