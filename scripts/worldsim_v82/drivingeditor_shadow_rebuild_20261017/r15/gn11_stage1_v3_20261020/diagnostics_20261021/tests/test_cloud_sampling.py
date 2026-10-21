"""采样率的用户可见语义；无需加载模型或使用GPU。"""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cloud_sampling import rate,ranked_sample

class SamplingTests(unittest.TestCase):
    def test_rate_bounds(self):
        for value in [-1,101,float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError):rate(value)
        self.assertEqual(rate('2.5'),2.5)
    def test_zero_and_full(self):
        rows=np.arange(101)
        self.assertEqual(len(ranked_sample(rows,0)),0)
        np.testing.assert_array_equal(np.sort(ranked_sample(rows,100)),rows)
    def test_exact_count_and_repeatability(self):
        rows=np.arange(100003)
        first=ranked_sample(rows,2)
        self.assertEqual(len(first),2001)
        np.testing.assert_array_equal(first,ranked_sample(rows,2))
        self.assertEqual(len(np.unique(first)),len(first))
    def test_raising_rate_retains_previous_centers(self):
        rows=np.arange(100003)
        previous=set()
        for percent in [0,.5,2,10,25,100]:
            current=set(ranked_sample(rows,percent))
            self.assertTrue(previous<=current)
            previous=current

if __name__=='__main__':unittest.main()
