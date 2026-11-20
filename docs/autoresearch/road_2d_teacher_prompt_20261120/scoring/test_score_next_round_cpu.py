"""仅人工CPU fixture：真实60像素门、GT顺序及新prompt字段接线。"""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_next_round as scorer


class ScoreNextRoundCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="score60_CPU_MOCK_", dir=HERE)
        cls.root = Path(cls.temporary.name).resolve()
        assert cls.root.is_relative_to(HERE.resolve())
        cls.entry = scorer.load_entry(scorer.DEFAULT_ENTRY)
        e, r = cls.entry, cls.root
        obs = np.full((1280, 1920, 3), 100, np.uint8)
        model = np.full((768, 1152, 3), 110, np.uint8)
        raw = np.asarray(Image.fromarray(model).resize((1920, 1280), Image.Resampling.BILINEAR))
        w = np.zeros((1280, 1920), bool); w[820:1180, 650:1400] = True
        hidden = np.zeros_like(w); hidden[800:1200, 630:1420] = True
        keep = np.zeros_like(w); keep[750:780, 300:320] = True
        cm = np.zeros_like(w); cm[700:1260, 300:1600] = True; cm &= ~hidden & ~keep
        final = obs.copy(); final[w] = raw[w]
        for name, value in [("observation", obs), ("clean", obs), ("raw_model", model), ("raw_native", raw), ("final_native", final)]:
            Image.fromarray(value).save(r / (name + ".png"))
        for name, value in [("valid", np.ones_like(w)), ("certificate", np.ones_like(w))]:
            Image.fromarray(value.astype(np.uint8) * 255).save(r / (name + ".png"))
        hm = np.asarray(Image.fromarray(hidden.astype(np.uint8)*255).resize((1152,768), Image.Resampling.NEAREST)) > 127
        inp = np.full_like(model, 100); inp[hm] = 127
        Image.fromarray(inp).save(r / "input.png")
        Image.fromarray(hm.astype(np.uint8)*255).save(r / "hidden_model.png")
        Image.fromarray((~hm).astype(np.uint8)*255).save(r / "C_mat_model.png")
        np.savez_compressed(r / "roles.npz", W_output=w, H_input=hidden, K=keep, C_mat=cm,
                            target_core=w, pending_output=np.zeros_like(w))
        np.savez_compressed(r / "roles_model.npz", H_input=hm)
        e.write(r / "construction.json", {"native_from_input":[[5/3,0,0],[0,5/3,0],[0,0,1]]})
        cfg = e.read(e.DEFAULT_CONFIG)
        e.write(r / "config.json", cfg)
        baseline = dict(schema="road_2d_teacher_pilot.v1", complete=True, prompt=cfg["prompts"][0]["text"],
            model="CPU_MOCK_NO_MODEL", checkpoint=dict(path=str(r/"not_loaded.pt"), sha256="a75423e126b93eca22e0ce587782b3dc4edd76654d6a929302118aa45bda14ae"),
            model_configs=[], source_files={}, repository=str(r), cases=[])
        for key in ["scheduler","inference_steps","guidance","eta","strength","dtype","weight_variant"]:
            baseline[key] = cfg["inference"][key]
        for sid in e.IDS:
            view = dict(available=True, model_wh=[1152,768], plan=dict(output_wh=[1152,768],window_xyxy=[0,0,1920,1280]),
                I_geo=e.pin(r/"input.png"),H_input=e.pin(r/"hidden_model.png"),C_mat=e.pin(r/"C_mat_model.png"),
                roles_model=e.pin(r/"roles_model.npz"),construction=e.pin(r/"construction.json"))
            views = {"full1152":view,
                "full768":dict(available=True,model_wh=[768,512],plan=dict(output_wh=[768,512],window_xyxy=[0,0,1920,1280])),
                "crop768":dict(available=True,model_wh=[768,272],plan=dict(output_wh=[768,272],window_xyxy=[0,600,1920,1280]))}
            baseline["cases"].append(dict(id=sid,domain="real_unlabelled" if sid.startswith("scene065") else "synthetic_visible_2d",
                native_hw=[1280,1920],observation=e.pin(r/"observation.png"),roles_native=e.pin(r/"roles.npz"),views=views))
        e.write(r / "baseline.json", baseline)
        history = dict(schema="road_2d_teacher_pilot_predictions.v1",complete=True,attempted_diffusion_calls=48,
            completed_diffusion_calls=48,adapter_unchanged=True,checkpoint_unchanged=True,manifest=e.pin(r/"baseline.json"),jobs=[])
        for sid in e.IDS:
            for arm in scorer.frozen.ARMS:
                for seed in (101,202):
                    history["jobs"].append(dict(id=f"{sid}__{arm}__seed{seed}",case_id=sid,arm=arm,seed=seed,
                        W_outside_RGB_exact=True,K_RGB_exact=True,**{k:e.pin(r/(k+".png")) for k in ("raw_model","raw_native","final_native")}))
        e.write(r / "history.json", history)
        cls.plan = e.prepare(r/"config.json",r/"baseline.json",r/"history.json",r/"plan")
        m = e.read(cls.plan)
        p = dict(schema=e.PREDICTION_SCHEMA,complete=True,CPU_mock=True,manifest=e.pin(cls.plan),attempted_diffusion_calls=60,
            completed_diffusion_calls=60,historical_replays_passed=12,GT_opened=False,training_updates=0,
            automatic_candidate_selection=False,adapter_unchanged=True,checkpoint_unchanged=True,first_historical_replay_passed=True,jobs=[])
        for j in m["jobs"]:
            row=copy.deepcopy(j)
            row.update({k:e.pin(r/(k+".png")) for k in ("raw_model","raw_native","final_native")})
            row["candidate_native_valid"]=e.pin(r/"valid.png")
            row["historical_replay_exact"]={k:True for k in ("raw_model","raw_native","final_native")} if j["historical_replay_expected"] else None
            p["jobs"].append(row)
        cls.good=p;cls.pred=r/"predictions.json";e.write(cls.pred,p)
        truth={"schema":"road_2d_teacher_scoring_truth.v1","cases":{}}
        for c in baseline["cases"]:
            record={"domain":c["domain"]}
            if c["domain"]=="synthetic_visible_2d":record.update(clean_rgb=e.pin(r/"clean.png"),visible_road_certificate=e.pin(r/"certificate.png"))
            truth["cases"][c["id"]]=record
        cls.truth=r/"truth_CPU_FIXTURE.json";e.write(cls.truth,truth)

    @classmethod
    def tearDownClass(cls):
        assert cls.root.is_relative_to(HERE.resolve())
        cls.temporary.cleanup()

    def test_01_production_rejects_labelled_mock(self):
        with self.assertRaisesRegex(scorer.frozen.GateError,"CPU mock"):
            scorer.complete_gate(self.pred,self.plan)

    def assert_truth_not_read(self, prediction):
        calls=[];original=scorer.read
        def watched(path):
            if Path(path).resolve()==self.truth.resolve():calls.append(str(path))
            return original(path)
        with patch.object(scorer,"read",side_effect=watched):
            with self.assertRaises(self.entry.ContractError):
                scorer.score(prediction,self.plan,self.truth,self.root/(prediction.stem+"_score.json"),allow_cpu_mock=True)
        self.assertEqual(calls,[])

    def test_02_missing_sixtieth_blocks_truth(self):
        p=copy.deepcopy(self.good);p["jobs"].pop();p["completed_diffusion_calls"]=59
        path=self.root/"incomplete.json";self.entry.write(path,p)
        self.assert_truth_not_read(path)

    def test_03_actual_pixel_violation_blocks_truth_even_with_new_pin(self):
        p=copy.deepcopy(self.good);bad=scorer.rgb(self.root/"final_native.png");bad[0,0]=0
        path=self.root/"bad_final.png";Image.fromarray(bad).save(path)
        p["jobs"][0]["final_native"]=self.entry.pin(path)
        pred=self.root/"wrong_pixels.json";self.entry.write(pred,prediction:=p)
        self.assert_truth_not_read(pred)

    def test_04_full_canonical_gate_and_prompt_plumbing(self):
        def fixture_metrics(*args,**kwargs):
            return dict(color={"deltaE76_mean":0.0},HP={},PSD={"band_power_ratio":[1.,1.,1.,1.,1.]})
        with patch.object(scorer.frozen,"metrics",side_effect=fixture_metrics):
            result=scorer.score(self.pred,self.plan,self.truth,self.root/"complete_CPU_MOCK_score.json",allow_cpu_mock=True)
        self.assertEqual(result["mode"],"CPU_MOCK_VALIDATION_ONLY")
        self.assertEqual(result["gate"]["completed_jobs"],60)
        self.assertEqual(result["gate"]["historical_replays"],12)
        self.assertEqual(len(result["jobs"]),60)
        self.assertEqual({j["prompt_id"] for j in result["jobs"]},set(scorer.PROMPTS))
        self.assertEqual({j["seed"] for j in result["jobs"]},set(scorer.SEEDS))
        self.assertTrue(result["GT_opened_after_gate"])
        self.assertFalse(result["automatic_winner_selected"])

    def test_05_frozen_metrics_and_common_band_definition(self):
        rng=np.random.default_rng(13);image=rng.integers(80,121,(160,160,3),dtype=np.uint8)
        result=scorer.frozen.metrics(image,image,np.ones((160,160),bool),np.full((160,160),100.),[(40,40,80)])
        result=scorer.add_common_psd(result)
        self.assertAlmostEqual(result["HP"]["2.5"]["RMS_ratio"],1.)
        self.assertAlmostEqual(result["PSD"]["common3_mean_abs_log2_band_power_ratio"],0.)
        result["PSD"]["band_power_ratio"][3]=1e6
        self.assertAlmostEqual(scorer.add_common_psd(result)["PSD"]["common3_mean_abs_log2_band_power_ratio"],0.)


if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ScoreNextRoundCPU)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    scorer.write(HERE/"cpu_mock_gate_validation.json",dict(schema="teacher_prompt_round_score_CPU_validation.v1",
        passed=result.wasSuccessful(),tests=result.testsRun,mode="CPU_MOCK_VALIDATION_ONLY",
        canonical_inference_60_pixel_gate_used=True,real_GPU_predictions_read=False,real_hidden_GT_read=False,
        full_metric_loop_mocked_only_in_plumbing_test=True,frozen_metric_identity_test=True,
        scorer=scorer.pin(HERE/"score_next_round.py"),frozen_metric_helper=scorer.pin(HERE/"metrics_pilot_frozen.py")))
    sys.exit(0 if result.wasSuccessful() else 1)
