import json,sys,unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_supervision_transfer as t

class SupportTests(unittest.TestCase):
    def test_positive_bilinear_footprint_matches_PIL(self):
        # Test exact zero-weight neighbors and clipped edges, independent of RGB
        # quantization. Each unit impulse reveals the actual nonzero filter span.
        for source,target in [(6,10),(8,20),(12,20)]:
            for invalid in range(source):
                valid=np.ones((1,source),bool);valid[0,invalid]=False
                impulse=np.zeros((1,source),np.float32);impulse[0,invalid]=1
                effect=np.asarray(Image.fromarray(impulse).resize((target,1),Image.Resampling.BILINEAR))>0
                actual=t.bilinear_all_axis(valid,target,1)
                self.assertTrue(np.array_equal(actual,~effect),(source,target,invalid))
    def test_historical_all_valid_and_full_image_impulse(self):
        valid=np.ones((512,768),bool);self.assertTrue(t.historical_support(valid).all())
        valid[200,300]=False
        impulse=np.zeros((512,768),np.float32);impulse[200,300]=1
        native=np.asarray(Image.fromarray(impulse).resize((1920,1280),Image.Resampling.BILINEAR))
        effect=native.reshape(320,4,480,4).mean(axis=(1,3))>0
        self.assertTrue(np.array_equal(t.historical_support(valid),~effect))
    def test_physical_tile_alignment(self):
        for wh,expected in [([1152,768],(48,48,48)),([768,512],(32,32,32)),([480,320],(20,20,20))]:
            self.assertEqual(t.bounds((80,80,80),wh),expected)
    def test_larger_highres_permission_cannot_expand_common_tiles(self):
        native=np.zeros((1280,1920),bool);native[80:320,80:320]=True
        before=t.candidate_tiles(native)
        low=np.ones((320,480),bool);low[20:40,20:40]=False
        high=np.ones((768,1152),bool)
        common=[tile for tile in before if t.tile_valid(low,tile) and t.tile_valid(high,tile)]
        common_without_high=[tile for tile in before if t.tile_valid(low,tile)]
        self.assertEqual(common,common_without_high);self.assertLess(len(common),len(before))

if __name__=='__main__':unittest.main(verbosity=2)
