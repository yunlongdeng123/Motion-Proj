"""Zero-update replay of the exact historical a5 R2 checkpoint and evaluator."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse,json,sys,traceback
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
OLD=ROOT/'shadow_BC_20261016_v1'
sys.path.insert(0,str(OLD/'implementation_v1'))
from run_experiments import Experiment,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();e=None
    try:
        e=Experiment(SimpleNamespace(output=args.output,steps=1000,warmup=300))
        ckpath=OLD/'runs/pilot_1000_v1/R2/addon_001000.pt'
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        assert ck['step']==1000 and ck['arm']=='R2'
        e.activate('R2');e.active.load_addon_state(ck['road_addon'])
        metrics=e.evaluate('R2');e.verify_sources()
        expected=.006653157528489828;actual=metrics['diagnostic_footprint_clean_linear_MAE']
        result=dict(optimizer_updates=0,checkpoint=str(ckpath),parent_checkpoint=str(ROOT/'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt'),
            historical_parent_not_v41_B=True,expected_footprint_MAE=expected,actual_footprint_MAE=actual,
            absolute_difference=abs(actual-expected),reproduced=abs(actual-expected)<1e-6,metrics=metrics)
        write(Path(args.output)/'replay_receipt.json',result);print(json.dumps(result,indent=2),flush=True)
        assert result['reproduced'],result
    except BaseException:
        dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
        (dest/'replay_failure.txt').write_text(traceback.format_exc());raise
    finally:
        if e:
            import renderer
            renderer.prepare_static=e.original_prepare
            if hasattr(e,'slot'):e.slot.close()

if __name__=='__main__':main()
