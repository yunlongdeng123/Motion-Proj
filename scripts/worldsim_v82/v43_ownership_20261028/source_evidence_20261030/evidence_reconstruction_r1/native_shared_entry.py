"""Install shared baseline gauge, then invoke unchanged native worker/evaluator."""
import argparse
import importlib.util
from pathlib import Path
import sys
from shared_gauge import install

def main():
    p=argparse.ArgumentParser();p.add_argument('--gauge',type=Path,required=True);p.add_argument('--native-code',type=Path,required=True);p.add_argument('--entry',choices=['native_worker','evaluate_edits','render_heldout'],required=True)
    args,rest=p.parse_known_args()
    sys.path.insert(0,str(args.native_code))
    from native_worker import SOURCE
    install(args.gauge,SOURCE)
    spec=importlib.util.spec_from_file_location('shared_gauge_native_entry',args.native_code/(args.entry+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.argv=[str(args.native_code/(args.entry+'.py'))]+rest
    module.main()

if __name__=='__main__':main()
