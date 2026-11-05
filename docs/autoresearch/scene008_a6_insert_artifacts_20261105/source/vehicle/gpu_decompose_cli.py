"""Replay the isolated diagnosis into a NEW output path; old assets stay read-only."""
import argparse
from pathlib import Path
import gpu_decompose

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,help='A nonexistent independent diagnostic directory on the AutoDL host')
    args=parser.parse_args()
    gpu_decompose.DEST=Path(args.output)
    gpu_decompose.main()
