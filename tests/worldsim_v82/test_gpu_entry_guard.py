import unittest
from gpu_entry_guard import check_phase

class PhaseTests(unittest.TestCase):
    def test_CPU_contract_blocks_both_GPU_phases(self):
        for phase in ['smoke','formal']:
            with self.subTest(phase=phase),self.assertRaises(PermissionError):check_phase({'GPU_authorized':False},phase)
        self.assertFalse(check_phase({'GPU_authorized':False},'inspect')['GPU_execution'])

    def test_CPU_success_cannot_authorize_formal(self):
        for receipt in [None,{'verified':True,'native_GPU_executed':False}]:
            with self.assertRaises(PermissionError):check_phase({'GPU_authorized':True},'formal',receipt)

if __name__=='__main__':unittest.main()
