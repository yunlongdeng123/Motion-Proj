"""CPU-only composition entry used after a separately authorized model run."""
import argparse
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repository',default='/root/autodl-tmp/motion_proj_v82_gn11_v42')
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--raw-prediction',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--diagnostic',action='store_true')
    args=parser.parse_args();sys.path.insert(0,str(Path(args.repository)))
    from scripts.worldsim_v82.road_material_field.input_construction.shadow_bundle import compose_saved_prediction
    print(compose_saved_prediction(args.bundle,args.raw_prediction,args.out,diagnostic=args.diagnostic))


if __name__=='__main__':main()
