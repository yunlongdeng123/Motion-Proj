"""CPU真实Adam：restore后后续更新不得篡改输入checkpoint。"""
import copy,types,unittest
from unittest.mock import patch
import torch
from train_world_material import TrainingState

class RestoreTests(unittest.TestCase):
    def test_saved_optimizer_tensors_remain_immutable_after_nextstep(self):
        low=torch.nn.Parameter(torch.ones(2,2,3));high=torch.nn.Parameter(torch.ones(3,3,3))
        optimizer=torch.optim.Adam([dict(params=[low],lr=.01),dict(params=[high],lr=.003)])
        for _ in range(2):
            low.grad=torch.ones_like(low);high.grad=torch.ones_like(high);optimizer.step()
        s=object.__new__(TrainingState);s.model=types.SimpleNamespace(low=low,high=high,set_stage=lambda stage:None)
        s.optimizer=optimizer;s.binding={};s.purpose='smoke';s.arm='common';s.schedule=[86,110]*400
        saved=dict(binding={},purpose='smoke',arm='common',cursor=2,schedule=s.schedule.copy(),trace=[86,110],
            actual_updates=2,stage='A',params=dict(low=low.detach().clone(),high=high.detach().clone()),
            optimizer=copy.deepcopy(optimizer.state_dict()),RNG={})
        before=copy.deepcopy(saved['optimizer'])
        with patch('train_world_material.restore_rng',lambda value:None):s.restore(saved)
        low.grad=torch.ones_like(low);high.grad=torch.ones_like(high);optimizer.step()
        for index,value in before['state'].items():
            for key,tensor in value.items():self.assertTrue(torch.equal(tensor,saved['optimizer']['state'][index][key]))

if __name__=='__main__':unittest.main()
