"""不重复曝光的全局分配控制。"""
import unittest
from sample_windows import match_exposures

class ExposureTests(unittest.TestCase):
    def test_duplicate_nearest_has_feasible_ordered_alternative(self):
        rows=[{'timestamp':t} for t in [50000,150000,250000]]
        chosen=match_exposures(rows,[100000,200000])
        self.assertEqual(chosen,[0,1])

    def test_duplicate_timestamp_is_not_two_frames(self):
        rows=[{'timestamp':50000},{'timestamp':50000}]
        self.assertIsNone(match_exposures(rows,[0,100000]))

    def test_no_padding_across_large_missing_interval(self):
        rows=[{'timestamp':t} for t in [0,100000,400000]]
        self.assertIsNone(match_exposures(rows,[0,100000,200000]))

    def test_greedy_tie_collision_avoided(self):
        # 最近邻可两次选150ms；全局解保留前后相邻真实曝光。
        rows=[{'timestamp':t} for t in [50001,150000,249999]]
        ix=match_exposures(rows,[100000,200000])
        self.assertEqual(len(set(ix)),2)
        self.assertEqual(ix,[0,1])

if __name__=='__main__':unittest.main()
