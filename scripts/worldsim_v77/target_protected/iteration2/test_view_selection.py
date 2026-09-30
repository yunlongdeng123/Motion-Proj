import unittest
from view_selection import select_clip_view

class ViewTests(unittest.TestCase):
    def test_identity_and_angular_evidence_required(self):
        good={'track_id':'real','instance_token':'A','quality_status':'pass','angles_deg':[[179,0],[178,0]]}
        self.assertEqual(select_clip_view([[-179,0],[-178,0]],[good],'A'),'real')
        self.assertIsNone(select_clip_view([[90,0],[91,0]],[good],'A'))
        self.assertIsNone(select_clip_view([[-179,0],[-178,0]],[good],'B'))
    def test_no_framewise_view_switch_or_unreviewed(self):
        tracks=[{'track_id':'left','instance_token':'A','quality_status':'pass','angles_deg':[[0,0],[0,0]]},
                {'track_id':'right','instance_token':'A','quality_status':'pass','angles_deg':[[40,0],[40,0]]}]
        self.assertIsNone(select_clip_view([[0,0],[40,0]],tracks,'A'))
        tracks[0]['quality_status']='pending';self.assertIsNone(select_clip_view([[0,0],[0,0]],tracks,'A'))

if __name__=='__main__':unittest.main()
