"""CPU-only checks of pilot summary semantics using previously labelled fixtures."""
import copy, json, sys, unittest, uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import summarize_pilot as s
WORK=HERE.parents[1]
OLD=WORK/'pixel_diagnosis/pilot_score_cpu_tests.json'

class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        receipt=s.read(OLD);cls.score=Path(receipt['score_file']);cls.data=s.read(cls.score)
        cls.out=HERE/'mock_summary_validation'/uuid.uuid4().hex
        cls.pred=Path(receipt['fixture_predictions']['path']);cls.manifest=WORK/'pixel_diagnosis/pilot_cpu_mock/prepared/pilot_manifest.json'
    def test_01_production_rejects_cpu_mock(self):
        with self.assertRaisesRegex(s.SummaryError,'CPU mock'):s.validate_score(self.data)
    def test_02_incomplete48_rejected(self):
        d=copy.deepcopy(self.data);d['jobs'].pop()
        with self.assertRaisesRegex(s.SummaryError,'combinations'):s.validate_score(d,True)
    def test_03_physical_sigma_mismatch_rejected(self):
        d=copy.deepcopy(self.data);d['jobs'][0]['same_view']['HP']['2.5']['sigma_this_grid_px']=2.5
        with self.assertRaisesRegex(s.SummaryError,'grid/native sigma'):s.validate_score(d,True)
    def test_04_extra_frequency_does_not_enter_common_metric(self):
        j=next(j for j in self.data['jobs'] if j['view']=='full1152');a=s.flatten_job(j,'same_view')
        jj=copy.deepcopy(j);jj['same_view']['PSD']['band_power_ratio'][3]*=1e6
        b=s.flatten_job(jj,'same_view')
        self.assertEqual(a['PSD_common3_mean_abs_log2_energy_ratio'],b['PSD_common3_mean_abs_log2_energy_ratio'])
        self.assertNotEqual(a['PSD_energy_ratio_f0.2_to_0.3_cycles_per_native_px'],b['PSD_energy_ratio_f0.2_to_0.3_cycles_per_native_px'])
    def test_05_symmetric_texture_distance_and_two_base_frames(self):
        self.assertEqual(s.abslogratio(.5),s.abslogratio(2.))
        self.assertEqual(s.abslogratio(1.),0.)
        rows,cases,frames,macro,contrasts=s.make_tables(self.data)
        self.assertEqual(len(rows),96);self.assertEqual(len(cases),48);self.assertEqual(len(contrasts),72)
        for row in macro:
            self.assertEqual(row['base_frame_count'],2)
            if row['domain']=='synthetic_certified_hidden_target':self.assertEqual(row['task_count'],4)
            else:self.assertEqual(row['task_count'],2)
    def test_06_full_mock_summary_artifacts(self):
        r=s.summarize(self.score,self.out,self.pred,self.manifest,True,True)
        self.assertEqual(r['mode'],'CPU_MOCK_VALIDATION_ONLY');self.assertFalse(r['automatic_winner_selected'])
        self.assertEqual(r['ledger_controls']['VAE_controls_completed'],0)
        self.assertTrue((self.out/'synthetic_color_spectrum.png').is_file())
        s.write(HERE/'summary_cpu_validation.json',{'schema':'teacher_48_summary_CPU_validation.v1','passed':True,'tests':6,'mode':'CPU_MOCK_VALIDATION_ONLY',
          'new_GPU_predictions_or_truth_read':False,'legacy_mock_score':s.pin(self.score),'script':s.pin(HERE/'summarize_pilot.py'),'summary_dir':str(self.out)})

if __name__=='__main__':unittest.main(verbosity=2)
