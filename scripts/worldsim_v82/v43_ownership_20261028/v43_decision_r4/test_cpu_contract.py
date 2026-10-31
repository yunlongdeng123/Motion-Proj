"""R4 CPU合同验证；同时读取本轮真实候选验收其冻结语义。"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from r4_core import (METRICS,OPERATIONS,canonical_hash,eligibility,enumerate_routes,
                     execution_plan,freeze_family,register_tasks,shared_route,
                     validate_candidate,validate_legal_evidence)

RUN = None


def fixture():
    metric=dict(numerator=1,denominator=1,status='fixture')
    evidence=dict(role='simulator_sensor_oracle',legal_observation_ids=['allowed/original_sensor'],
        metrics={m:deepcopy(metric) for m in METRICS},
        operation_view={q:deepcopy(metric) for q in OPERATIONS},
        operation_bg={q:deepcopy(metric) for q in OPERATIONS})
    candidate=dict(schema='r4_frozen_group_candidates_v1',legal_input_role='simulator_sensor_oracle',
        static_row_count=10,scene_ancestor='fixture',target_id='actor_0',checkpoint_static_sha256='staticsha',
        row_namespace='staticsha:static_original_row',
        groups=[dict(group_id='g0',original_binding='static',rebind_target='actor_0',original_static_rows=[1,3])])
    return evidence,candidate


def scene_context():
    components=dict(static='staticsha',dynamic='dynamicsha',environment='envsha',training_state='runtimesha')
    return dict(component_sha256=components,
        checkpoint_bundle_sha256=canonical_hash([dict(name=name,sha256=components[name]) for name in sorted(components)]),
        event_sha256='eventsha',gauge_sha256='gaugesha',camera_sha256='camerasha',preparation_sha256='prepsha')


class ContractTests(unittest.TestCase):
    def test_opaque_alpha_does_not_change_foreground(self):
        from legal_mask import validate_RGBA_equivalence
        self.assertTrue(validate_RGBA_equivalence()['actual_PNG_roundtrip_and_shared_decoder'])
    def test_uncalibrated_is_unknown(self):
        e,_=fixture()
        self.assertIsNone(eligibility(e,'rotate',{m:None for m in METRICS})['passed'])

    def test_zero_denominator_is_unknown(self):
        e,_=fixture();e['operation_bg']['delete']=dict(numerator=0,denominator=0)
        out=eligibility(e,'delete',{m:.5 for m in METRICS})
        self.assertIsNone(out['passed']);self.assertIsNone(out['features']['f_bg']['value'])

    def test_invalid_operation_metrics_rejected(self):
        for field in ('operation_view','operation_bg'):
            for bad in (dict(numerator=2,denominator=1),dict(numerator=1,denominator=-1),
                        dict(numerator=True,denominator=1),dict(numerator=1,denominator=True),
                        dict(numerator=float('nan'),denominator=1)):
                e,_=fixture();e[field]['rotate']=bad
                with self.assertRaises(ValueError):eligibility(e,'rotate',{m:.5 for m in METRICS})

    def test_illegal_feature_rejected(self):
        for key in ('semantic_id','student_output','evaluation_only_capability_bin'):
            e,_=fixture();e['metrics']['f_geom'][key]=[1]
            with self.assertRaises(ValueError):validate_legal_evidence(e)

    def test_illegal_rows_rejected(self):
        _,c=fixture()
        for rows in ([-1],[10],[1,1],[3,1],[True]):
            x=deepcopy(c);x['groups'][0]['original_static_rows']=rows
            with self.assertRaises(ValueError):validate_candidate(x)

    def test_defer_and_extra_actions_rejected_in_scene_binding(self):
        _,c=fixture()
        for action in ('defer','delete','split','color'):
            with self.assertRaises(ValueError):shared_route(c,dict(g0=action))
        route=shared_route(c,dict(g0='keep'))
        deferred=execution_plan(route,'rotate',defer=True)
        self.assertFalse(deferred['rendered']);self.assertFalse(deferred['task_completed'])

    def test_operation_is_executor_only(self):
        _,c=fixture();route=shared_route(c,dict(g0='rebind'));before=deepcopy(route)
        hashes={execution_plan(route,q)['route_assignment_hash'] for q in OPERATIONS}
        self.assertEqual(len(hashes),1);self.assertEqual(route,before)
        with self.assertRaises(ValueError):execution_plan(route,'teleport')

    def test_route_tampering_rejected(self):
        _,c=fixture();route=shared_route(c,dict(g0='keep'))
        route['assignments'][0]['binding']='other'
        with self.assertRaises(ValueError):execution_plan(route,'none')

    def test_domain_count_and_incomplete_registration_rejected(self):
        e,c=fixture();routes=enumerate_routes(c)
        self.assertEqual(len(routes),2)
        for bad in (routes[:1],[routes[0],routes[0]]):
            with self.assertRaises(ValueError):register_tasks('x',c,e,bad,'staticsha',scene_context())
        with self.assertRaises(ValueError):register_tasks('x',c,e,routes,'other_checkpoint',scene_context())
        other=deepcopy(c);other['row_namespace']='other:static_original_row'
        with self.assertRaises(ValueError):register_tasks('x',c,e,enumerate_routes(other),'staticsha',scene_context())

    def test_empty_domain_one_keep_state_zero_eligibility_not_success(self):
        e,c=fixture();c['groups']=[]
        routes=enumerate_routes(c);self.assertEqual(len(routes),1)
        tasks=register_tasks('empty',c,e,routes,'staticsha',scene_context())
        self.assertEqual(len(tasks),4)
        self.assertTrue(all(t['search']['evaluated_count']==0 for t in tasks))
        self.assertTrue(all(t['decision']['decision_gap'] is None for t in tasks))

    def test_bundle_gauge_context_is_frozen(self):
        e,c=fixture();routes=enumerate_routes(c);context=scene_context()
        first=register_tasks('x',c,e,routes,'staticsha',context)[0]
        context['gauge_sha256']='another_gauge'
        second=register_tasks('x',c,e,routes,'staticsha',context)[0]
        self.assertNotEqual(first['shared_route_context_hash'],second['shared_route_context_hash'])
        context['component_sha256']['environment']='another_environment'
        with self.assertRaises(ValueError):register_tasks('x',c,e,routes,'staticsha',context)

    def test_actual_registered_denominators_and_null_science(self):
        if RUN is None:self.skipTest('requires actual run')
        summary=json.loads((RUN/'summary.json').read_text())
        self.assertEqual(summary['registered_tasks'],12)
        self.assertEqual(summary['independent_ancestors'],2)
        for scene in summary['scenes']:
            folder=RUN/scene['scene_id'];candidate=json.loads((folder/'candidate_manifest.json').read_text())
            tasks=json.loads((folder/'tasks.json').read_text())
            self.assertEqual(validate_candidate(candidate),scene['candidate_rows'])
            self.assertEqual(len(enumerate_routes(candidate)),scene['possible_shared_routes'])
            self.assertEqual({t['edit_type'] for t in tasks},set(OPERATIONS))
            reference=[a['route_assignment_hash'] for a in tasks[0]['action_evaluations']]
            for t in tasks:
                self.assertEqual(reference,[a['route_assignment_hash'] for a in t['action_evaluations']])
                self.assertIsNone(t['human_verdict']);self.assertIsNone(t['eligibility']['passed'])
                self.assertEqual(t['search']['oracle_status'],'safe_oracle_unavailable')
                self.assertIsNone(t['decision']['decision_gap'])
                self.assertTrue(all(a['loss_vector'] is None and not a['rendered'] for a in t['action_evaluations']))

    def test_actual_frozen_evidence_function_isolation(self):
        if RUN is None:self.skipTest('requires actual run')
        # 重用真实冻结合法证据；独立参考/学生输出既不在参数也不在文件依赖里。
        for path in RUN.glob('*/legal_preedit_evidence.json'):
            evidence=json.loads(path.read_text());thresholds={m:None for m in METRICS}
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'independent_edited_reference').mkdir();(root/'student_output').mkdir()
                (root/'student_output/prediction.json').write_text('{"model":"A","pick":"keep"}')
                before=[eligibility(evidence,q,thresholds) for q in OPERATIONS]
                (root/'student_output/prediction.json').write_text('{"model":"B","pick":"rebind"}')
                after_student=[eligibility(evidence,q,thresholds) for q in OPERATIONS]
                shutil.rmtree(root/'independent_edited_reference');shutil.rmtree(root/'student_output')
                after_delete=[eligibility(evidence,q,thresholds) for q in OPERATIONS]
                self.assertEqual(before,after_student);self.assertEqual(before,after_delete)
        frozen=json.loads((RUN/'frozen_inputs.json').read_text())
        self.assertTrue(all('/privileged/' not in r['path'] and 'student_output' not in r['path'] for r in frozen['inputs']))

    def test_actual_sim_body_denominators_match_native_preparation(self):
        if RUN is None:self.skipTest('requires actual run')
        from legal_mask import read_body_mask
        contract=json.loads((RUN/'run_contract.json').read_text())
        for scene in contract['checkpoint_inputs']:
            if scene['kind']!='sim':continue
            event=json.loads(Path(scene['event']).read_text())
            prep=json.loads((Path(event['source_path'])/'preparation_manifest.json').read_text())
            obs=Path(prep['observations']);manifest=json.loads((obs/'manifest.json').read_text())
            expected={r['id']:r['body_pixels'] for r in prep['frames']}
            actual=0
            for row in manifest['frames']:
                fid=str(row['id']).zfill(6);mask,_=read_body_mask(obs/row['body_mask'])
                self.assertEqual(int(mask.sum()),expected[fid]);actual+=int(mask.sum())
            evidence=json.loads((RUN/scene['id']/'legal_preedit_evidence.json').read_text())
            self.assertEqual(evidence['metrics']['f_geom']['denominator'],actual)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--output',type=Path)
    args,remaining=parser.parse_known_args();RUN=args.run
    program=unittest.main(argv=[__file__]+remaining,exit=False)
    result=program.result
    if args.output:
        args.output.write_text(json.dumps(dict(schema='r4_cpu_contract_tests_v1',tests_run=result.testsRun,
            success=result.wasSuccessful(),skipped=len(result.skipped),failures=len(result.failures),errors=len(result.errors),
            actual_run=str(RUN) if RUN else None,
            isolation_scope='frozen real evidence pure-function invariance; source whitelist and optional diagnostic design; not full adapter replay after deleting original reference directories',
            failure_details=[str(x) for x in result.failures+result.errors]),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
