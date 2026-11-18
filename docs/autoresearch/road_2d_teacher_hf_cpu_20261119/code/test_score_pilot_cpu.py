"""Labelled CPU fixture validation using only frozen legacy B outputs.
python work/pixel_diagnosis/test_score_pilot_cpu.py
Creates/reuses a fixture; it does NOT run diffusion or prove GPU reproducibility.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import copy, importlib.util, json, sys, unittest, uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'teacher_audit'))
import score_pilot as s
BASE=Path(r'C:/Users/dengyunlong/Documents/Codex/2026-11-07/https-arxiv-org-html-2509-19937/outputs/road_mask_decoupling_20261116/assets/readability_20261118')
OLD=BASE.parents[3]/'work/road_mask_decoupling_20261116/evaluation/results_r1/results.json'
BUILDER=BASE.parents[3]/'work/road_mask_decoupling_20261116/stage/input_builder.py'
FIXTURE=HERE/'pilot_cpu_mock'

def save_binary(path,a):Image.fromarray(a.astype(np.uint8)*255).save(path)

def make_fixture():
    manifest_path=FIXTURE/'prepared/pilot_manifest.json';pred_path=FIXTURE/'predictions/predictions.json';truth_path=FIXTURE/'private_scoring/truth_config.json'
    if all(p.exists() for p in [manifest_path,pred_path,truth_path]):return manifest_path,pred_path,truth_path
    source=s.read(BASE/'source_manifest.json');lookup={c['id']:c for c in source['cases']}
    old={c['id']:c for c in s.read(OLD)['cases']}
    cases=[];jobs=[];predictions=[];truth_cases={};remote_truth={}
    for sid in s.IDS:
        print('CPU fixture',sid,flush=True)
        c=lookup[sid];folder=FIXTURE/'prepared'/sid;folder.mkdir(parents=True,exist_ok=True)
        obs=s.rgb(BASE/c['observation']);r=s.roles(BASE/c['arms']['B']['roles']);rawmodel=s.rgb(BASE/c['arms']['B']['raw_model']);rawnative=s.rgb(BASE/c['arms']['B']['raw'])
        obs_pin=s.pin(BASE/c['observation']);roles_pin=s.pin(BASE/c['arms']['B']['roles'])
        expected={'raw_model':s.pin(BASE/c['arms']['B']['raw_model']),'raw_native':s.pin(BASE/c['arms']['B']['raw']),'final_native':s.pin(BASE/c['arms']['B']['final'])}
        plans={'full768':{'output_wh':[768,512],'window_xyxy':[0,0,1920,1280]},'crop768':{'output_wh':[768,272],'window_xyxy':[0,600,1920,1280]},'full1152':{'output_wh':[1152,768],'window_xyxy':[0,0,1920,1280]}}
        views={};viewoutputs={}
        for name,plan in plans.items():
            vf=folder/name;vf.mkdir(exist_ok=True)
            lx,ly,hx,hy,fx,fy,valid=s.view_neighbors(plan,obs.shape[:2])
            hm=(r['H_input'][ly,lx]|r['H_input'][ly,hx]|r['H_input'][hy,lx]|r['H_input'][hy,hx])&valid
            k=(r['K'][ly,lx]|r['K'][ly,hx]|r['K'][hy,lx]|r['K'][hy,hx])&valid
            wm=(r['W_output'][ly,lx]|r['W_output'][ly,hx]|r['W_output'][hy,lx]|r['W_output'][hy,hx])&valid&~k
            if name=='full768':
                input_pin=s.pin(BASE/c['arms']['B']['input']);hole_pin=s.pin(BASE/c['arms']['B']['mask_model'])
                hm=s.binary(hole_pin['path'])
            else:
                legal=obs.copy();legal[r['H_input']]=127
                inp=s.sample_view(legal,plan);inp[hm]=127
                Image.fromarray(inp).save(vf/'I_geo.png');save_binary(vf/'H_input.png',hm)
                input_pin=s.pin(vf/'I_geo.png');hole_pin=s.pin(vf/'H_input.png')
            np.savez_compressed(vf/'roles_model.npz',H_input=hm,W_output=wm,K=k,C_mat=s.sample_view(r['C_mat'],plan,mask=True))
            s.write(vf/'construction.json',{'native_from_input':s.view_geometry(plan,obs.shape[:2])[-1].tolist(),'CPU_mock_fixture':True})
            views[name]={'available':True,'plan':plan,'model_wh':plan['output_wh'],'I_geo':input_pin,'H_input':hole_pin,'roles_model':s.pin(vf/'roles_model.npz'),'construction':s.pin(vf/'construction.json')}
            out=FIXTURE/'predictions'/sid/name;out.mkdir(parents=True,exist_ok=True)
            if name=='full768':
                model=rawmodel;native=rawnative;rawpin=expected['raw_model'];nativepin=expected['raw_native'];finalpin=expected['final_native']
                _,native_valid=s.inverse_view(model,obs.shape[:2],plan)
            else:
                model=s.sample_view(rawnative,plan);native,native_valid=s.inverse_view(model,obs.shape[:2],plan)
                final=obs.copy();final[r['W_output']]=native[r['W_output']]
                Image.fromarray(model).save(out/'raw_model.png');Image.fromarray(native).save(out/'raw_native.png');Image.fromarray(final).save(out/'final_native.png')
                rawpin=s.pin(out/'raw_model.png');nativepin=s.pin(out/'raw_native.png');finalpin=s.pin(out/'final_native.png')
            save_binary(out/'candidate_native_valid.png',native_valid)
            viewoutputs[name]={'raw_model':rawpin,'raw_native':nativepin,'final_native':finalpin,'candidate_native_valid':s.pin(out/'candidate_native_valid.png')}
        cases.append({'id':sid,'domain':c['domain'],'native_hw':[1280,1920],'observation':obs_pin,'roles_native':roles_pin,'views':views,'legacy_B_expected':expected})
        for seed in s.SEEDS:
            for arm in s.ARMS:
                view=arm.rsplit('_',1)[0]
                j={'id':f'{sid}__{arm}__seed{seed}','case_id':sid,'arm':arm,'view':view,'seed':seed,'adapter':'none' if arm.endswith('_base') else 'T0','legacy_exact_replay_expected':arm=='full768_T0' and seed==101}
                jobs.append(j);predictions.append({**j,**viewoutputs[view],'W_outside_RGB_exact':True,'K_RGB_exact':True,'legacy_replay_exact':{'raw_model':True,'raw_native':True,'final_native':True} if j['legacy_exact_replay_expected'] else None})
        if c['domain']=='synthetic_visible_2d':
            frame='000' if 'raw000' in sid else '050';pins=old[sid]['GT_source_pins']
            truth_cases[sid]={'domain':c['domain'],'clean_rgb':{'path':str(HERE/'truth'/f'scene081_raw{frame}_clean.jpg'),'sha256':pins['clean_rgb_for_scoring_only']['sha256']},'visible_road_certificate':{'path':str(HERE/'truth'/f'scene081_raw{frame}_certificate.png'),'sha256':pins['visible_road_certificate_for_scoring_only']['sha256']}}
            remote_truth[sid]={'domain':c['domain'],'clean_rgb':pins['clean_rgb_for_scoring_only'],'visible_road_certificate':pins['visible_road_certificate_for_scoring_only']}
        else:
            truth_cases[sid]={'domain':c['domain']};remote_truth[sid]={'domain':c['domain']}
    manifest={'schema':'road_2d_teacher_pilot.v1','complete':True,'CPU_mock_fixture':True,'GT_opened':False,'training_updates':0,'seeds':s.SEEDS,'arms':s.ARMS,'planned_diffusion_calls':48,'maximum_diffusion_calls':48,'cases':cases,'jobs':jobs}
    s.write(manifest_path,manifest)
    ledger={'schema':'road_2d_teacher_pilot_predictions.v1','complete':True,'CPU_mock':True,'GPU_used':False,'mock_description':'Legacy B arrays reused for arm/seed labels and geometrically resampled for other views. No new model outcome, diffusion, adapter comparison, GPU replay or scientific result is simulated.','manifest':s.pin(manifest_path),'GT_opened':False,'training_updates':0,'attempted_diffusion_calls':48,'completed_diffusion_calls':48,'adapter_unchanged':True,'checkpoint_unchanged':True,'jobs':predictions}
    s.write(pred_path,ledger);s.write(truth_path,{'schema':'road_2d_teacher_scoring_truth.v1','cases':truth_cases})
    s.write(HERE.parent/'teacher_audit/score_truth_config.remote.json',{'schema':'road_2d_teacher_scoring_truth.v1','independent_scoring_only':True,'generation_must_not_read':True,'cases':remote_truth})
    return manifest_path,pred_path,truth_path

class ScoreGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.pred,cls.truth=make_fixture();cls.out=HERE/'pilot_cpu_mock_tests'/uuid.uuid4().hex;cls.out.mkdir(parents=True)
        cls.receipt={'schema':'teacher_pilot_score_CPU_mock_tests.v1','GPU_used':False,'new_inference':False,'actual_pilot_results':False,'tests':[]}
    def record(self,name,detail):self.receipt['tests'].append({'name':name,'passed':True,'detail':detail})
    def altered_ledger(self,name,mutate):
        p=copy.deepcopy(s.read(self.pred));mutate(p);path=self.out/(name+'.json');s.write(path,p);return path
    def rejected_without_truth_read(self,pred,match):
        original=s.read;truth_read=[]
        def guarded(path):
            if Path(path).resolve()==self.truth.resolve():truth_read.append(str(path));raise AssertionError('truth opened before rejection')
            return original(path)
        with patch.object(s,'read',side_effect=guarded),self.assertRaisesRegex(s.GateError,match):
            s.score(pred,self.manifest,self.truth,self.out/(uuid.uuid4().hex+'.json'),allow_cpu_mock=True)
        self.assertEqual(truth_read,[])
    def test_01_incomplete_rejected_before_truth(self):
        p=self.altered_ledger('incomplete',lambda x:x.update(complete=False))
        self.rejected_without_truth_read(p,'incomplete');self.record('incomplete_gate','47/incomplete cannot read independent truth')
    def test_02_duplicate_rejected_before_truth(self):
        def alter(p):p['jobs'][-1]=copy.deepcopy(p['jobs'][-2])
        p=self.altered_ledger('duplicate',alter);self.rejected_without_truth_read(p,'missing/duplicate');self.record('combination_gate','48 labels with a duplicated arm/seed rejected before truth')
    def test_03_lying_scope_flag_rejected_before_truth(self):
        def alter(p):
            j=next(j for j in p['jobs'] if j['arm']=='full768_base');a=s.rgb(j['final_native']['path']);a[0,0,0]=(int(a[0,0,0])+1)%256
            path=self.out/'scope_lie_final.png';Image.fromarray(a).save(path);j['final_native']=s.pin(path)
        p=self.altered_ledger('scope_lie',alter);self.rejected_without_truth_read(p,'outside frozen W');self.record('scope_gate','true receipt flags plus repinned illegal pixel still rejected before truth')
    def test_04_crop_inverse_rejected_before_truth(self):
        def alter(p):
            j=next(j for j in p['jobs'] if j['view']=='crop768');a=s.rgb(j['raw_native']['path']);a[0,0,0]=128
            path=self.out/'crop_inverse_lie.png';Image.fromarray(a).save(path);j['raw_native']=s.pin(path)
        p=self.altered_ledger('crop_lie',alter);self.rejected_without_truth_read(p,'inverse transform mismatch');self.record('crop_inverse_gate','repinned change outside crop caught by exact inverse transform')
    def test_05_mock_rejected_by_default(self):
        with self.assertRaisesRegex(s.GateError,'CPU mock'):s.freeze_gate(self.pred,self.manifest)
        self.record('mock_label_gate','CPU fixture cannot enter production evaluation by default')
    def test_06_coordinate_matches_actual_builder(self):
        spec=importlib.util.spec_from_file_location('frozen_input_builder',BUILDER);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        rng=np.random.default_rng(9101);a=rng.integers(0,256,(1280,1920,3),dtype=np.uint8)
        for name,plan in [('full768',{'output_wh':[768,512],'window_xyxy':[0,0,1920,1280]}),('crop768',{'output_wh':[768,272],'window_xyxy':[0,600,1920,1280]}),('full1152',{'output_wh':[1152,768],'window_xyxy':[0,0,1920,1280]})]:
            _,_,uv,_=builder._plan(plan,a.shape[:2]);nb=builder._neighbors(uv,a.shape[:2]);ref=np.rint(np.clip(builder._sample(a,nb),0,255)).astype(np.uint8)
            self.assertTrue(np.array_equal(s.sample_view(a,plan),ref),name)
        # Analytic ramp reveals both center offset and600px crop translation.
        yy,xx=np.mgrid[:1280,:1920];ramp=np.stack([xx%251,yy%251,(xx+yy)%251],-1).astype(np.uint8)
        plan={'output_wh':[768,272],'window_xyxy':[0,600,1920,1280]}
        self.assertFalse(np.array_equal(s.sample_view(ramp,plan),s.sample_view(ramp,{'output_wh':[768,272],'window_xyxy':[0,0,1920,680]})))
        self.record('coordinate_transform','full768/crop768/full1152 uint8 RGB exactly match frozen actual builder on random native image; crop600 offset separately distinguished')
    def test_07_full_labelled_mock_score(self):
        out=self.out/'mock_scores.json';result=s.score(self.pred,self.manifest,self.truth,out,allow_cpu_mock=True)
        self.assertEqual(result['mode'],'CPU_MOCK_VALIDATION_ONLY');self.assertTrue(result['complete']);self.assertEqual(result['scored_jobs'],48)
        self.assertTrue(result['GT_opened_after_gate']);self.assertFalse(result['automatic_winner_selected'])
        self.assertEqual({j['case_id'] for j in result['jobs']},set(s.IDS))
        self.assertTrue(all(j['native']['spacing_native_px']==1 for j in result['jobs']))
        for j in result['jobs']:
            expected=2.5 if j['view']!='full1152' else 1920/1152
            self.assertAlmostEqual(j['same_view']['spacing_native_px'],expected)
            for sigma,h in j['same_view']['HP'].items():
                if h.get('status')!='insufficient':self.assertAlmostEqual(h['sigma_this_grid_px']*expected,float(sigma))
        self.record('full48_CPU_mock_scoring','all48 rows scored after gates, exact coordinate/sigma units checked;18 unique reused-array computations; no GPU scientific results')
        self.receipt['score_file']=str(out);self.receipt['score_sha256']=s.sha(out)
    @classmethod
    def tearDownClass(cls):
        cls.receipt['passed']=len(cls.receipt['tests'])==7
        cls.receipt['fixture_predictions']=s.pin(cls.pred);cls.receipt['score_program']=s.pin(HERE.parent/'teacher_audit/score_pilot.py')
        s.write(HERE/'pilot_score_cpu_tests.json',cls.receipt)

if __name__=='__main__':unittest.main(verbosity=2)
