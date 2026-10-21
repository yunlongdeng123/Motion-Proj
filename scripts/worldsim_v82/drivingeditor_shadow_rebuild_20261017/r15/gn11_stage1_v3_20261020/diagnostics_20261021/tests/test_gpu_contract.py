import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('viewer_server',Path(__file__).resolve().parents[1]/'src/server.py')
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
class GPUContractTests(unittest.TestCase):
    def query(self,**changes):return dict(dict(action='trace',arm='full',raw=102,x=90,y=255,mode='deleted_five'),**changes)
    def test_valid_trace(self):server.Handler.validate_gpu(self.query())
    def test_native_v1(self):server.Handler.validate_gpu(self.query(arm='gn11_v1'))
    def test_v1_road_not_injected(self):
        with self.assertRaises(ValueError):server.Handler.validate_gpu(self.query(arm='gn11_v1',mode='road_only'))
    def test_valid_bound_action(self):server.Handler.validate_gpu(self.query(action='counterfactual',ids=['full:3184132db573:static:7']))
    def test_unknown_command_rejected(self):
        with self.assertRaises(ValueError):server.Handler.validate_gpu(self.query(command='train.py'))
    def test_raw138_rejected(self):
        with self.assertRaises(ValueError):server.Handler.validate_gpu(self.query(raw=138))
    def test_bool_pixel_rejected(self):
        with self.assertRaises(ValueError):server.Handler.validate_gpu(self.query(x=True))
    def test_outside_pixel_rejected(self):
        with self.assertRaises(ValueError):server.Handler.validate_gpu(self.query(x=480))
    def test_fifth_arm_rejected(self):
        with self.assertRaises(ValueError):server.Handler.validate_gpu(self.query(arm='v4'))
    def test_no_training_action(self):
        with self.assertRaises(ValueError):server.Handler.validate_gpu(self.query(action='train'))
    def test_empty_removal_rejected(self):
        with self.assertRaises(ValueError):server.Handler.validate_gpu(self.query(action='counterfactual',ids=[]))
    def test_large_removal_rejected(self):
        with self.assertRaises(ValueError):server.Handler.validate_gpu(self.query(action='counterfactual',ids=['id']*201))

if __name__=='__main__':unittest.main()
