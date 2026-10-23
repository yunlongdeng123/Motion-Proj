"""CPU diagnostic wiring only; never substitutes for real CUDA smoke."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from v41_gradient_audit import domain_gradient_audit, a37_path_failures, AUDIT_FRAMES


class GradientAuditTests(unittest.TestCase):
    def test_other_actor_gradient_cannot_pass_blocked_a37(self):
        dc = torch.nn.Parameter(torch.tensor([.3,.3])); opacity = torch.nn.Parameter(torch.tensor([-.2,-.2]))
        reach = torch.tensor([1.,0.])
        package = dict(rgb=(dc*reach)[None,None].expand(3,1,2),
            road_visible_alpha=(opacity.sigmoid()*reach)[None,None])
        domains = torch.zeros(5,1,2,dtype=torch.bool);domains[0,0,0]=True;domains[4,0,1]=True
        report = domain_gradient_audit(package,torch.ones(3,1,2)*.6,domains,domains,(dc,opacity))
        self.assertGreater(report['5']['rgb_gradient_l1']['dc'],0)
        self.assertEqual(report['37']['rgb_gradient_l1']['dc'],0)
        self.assertTrue(a37_path_failures({f:report for f in AUDIT_FRAMES}))
        self.assertIsNone(dc.grad);self.assertIsNone(opacity.grad)

    def test_zero_label_stays_unknown_even_with_contribution_gradient(self):
        dc = torch.nn.Parameter(torch.ones(3,1,2)*.3);op = torch.nn.Parameter(torch.ones(1,1,2)*-.2)
        full = torch.ones(5,1,2,dtype=torch.bool);domains=full.clone();domains[4]=False
        report = domain_gradient_audit(dict(rgb=dc,road_visible_alpha=op.sigmoid()),
            torch.ones_like(dc),domains,full,(dc,op))
        self.assertEqual(report['37']['color_status'],'NO_REFERENCE')
        self.assertIsNone(report['37']['rgb_gradient_l1'])
        self.assertGreater(report['37']['full_contribution_gradient_l1']['opacity'],0)


if __name__ == '__main__': unittest.main(verbosity=2)
