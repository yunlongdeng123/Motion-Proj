import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image
import build_checkpoint_comparison as builder


class ConsumerTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.root=Path(self.temp.name)
        self.run=self.root/'run';self.run.mkdir();(self.run/'checkpoint_events').mkdir()
        self.args=SimpleNamespace(run=str(self.run),output=str(self.root/'report'),steps=[10000,20000],
                                  camera='image_0',fps=10,title='test',history_url='')
        self.encoder=patch.object(builder,'encode_video',side_effect=self.encode);self.mock=self.encoder.start()
    def tearDown(self): self.encoder.stop();self.temp.cleanup();builder._event_cache.clear()
    def encode(self,paths,destination,fps,resize_width=None):
        destination.write_bytes(b'fixture video')
        return {'path':str(destination),'fps':fps,'frames':len(paths),'width':4,'height':4}
    def event(self,iteration,color=(10,20,30)):
        root=self.run/str(iteration);root.mkdir();components={}
        for name in ('static','dynamic','environment','training_state'):
            path=root/(name+'.pth');path.write_bytes(b'fixture checkpoint')
            components[name]={'path':str(path),'bytes':path.stat().st_size}
        frames=[]
        for i in range(2):
            channels={}
            for channel in builder.CHANNELS:
                path=root/f'{channel}{i}.png';Image.new('RGB',(4,4),color).save(path);channels[channel]=str(path)
            frames.append({'frame_id':str(i),'camera':'image_0','timestamp':i*.1,'channels':channels})
        event={'iteration':iteration,'status':'complete','checkpoints':components,'frames':frames,
               'metrics':{'psnr':12.3},'paper_metrics':{'summary':{'dynamic_psnr_star':{'psnr_roi_mean':8.9}}}}
        self.write(iteration,event);return event
    def write(self,iteration,event):
        path=self.run/'checkpoint_events'/f'iteration_{iteration:06d}.json';path.write_text(json.dumps(event));return path
    def build(self):
        with contextlib.redirect_stdout(io.StringIO()): return builder.build_once(self.args)
    def test_missing_and_incomplete_events_pending(self):
        self.write(10000,{'status':'saving'});report=self.build()
        self.assertEqual([n['status'] for n in report['nodes']],['pending','pending']);self.mock.assert_not_called()
    def test_complete_and_metrics_passthrough(self):
        self.event(10000);report=self.build();node=report['nodes'][0]
        self.assertEqual(node['status'],'ready');self.assertEqual(node['metrics']['psnr'],12.3)
        self.assertEqual(node['paper_metrics']['summary']['dynamic_psnr_star']['psnr_roi_mean'],8.9)
        self.assertEqual(report['frame_count'],2)
    def test_missing_training_state_rejected(self):
        event=self.event(10000);Path(event['checkpoints']['training_state']['path']).unlink()
        self.assertIn('training_state',self.build()['warnings'][0]);self.mock.assert_not_called()
    def test_order_rejected(self):
        event=self.event(10000);event['frames'].reverse();self.write(10000,event)
        self.assertIn('chronological',self.build()['warnings'][0])
    def test_mixed_gt_cannot_join(self):
        self.event(10000);self.event(20000,color=(50,60,70));report=self.build()
        self.assertEqual([n['status'] for n in report['nodes']],['ready','error'])
        self.assertIn('different frames/GT',report['warnings'][0])
    def test_idempotent_then_fps_invalidates(self):
        self.event(10000);first=self.build();calls=self.mock.call_count;second=self.build()
        self.assertEqual(self.mock.call_count,calls);self.assertEqual(first['revision'],second['revision'])
        self.args.fps=12;self.build();self.assertEqual(self.mock.call_count,calls+4)

if __name__=='__main__': unittest.main(verbosity=2)
