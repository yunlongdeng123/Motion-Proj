"""Separate external scoring of the frozen full199 development bank."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True


def main():
    p=argparse.ArgumentParser();p.add_argument("--benchmark",type=Path,required=True);args=p.parse_args();root=args.benchmark
    source=Path(__file__).resolve().parent;run=root/"baseline_runs/full199_plane_boundary_median_development_v1"
    if not (run/"frozen_bank_receipt.json").exists():raise RuntimeError("Freeze all12 estimator predictions before external scoring")
    for track in ("appearance_blind","geometry_RGB_blind"):
        for split,label in (("validation","validation"),("sealed_test","test")):
            subprocess.run([sys.executable,str(source/"evaluate_external.py"),"--benchmark",str(root),
                "--prediction",str(run/(track+"_prediction.npz")),
                "--frozen-candidate-receipt",str(run/(track+"_frozen_candidate.json")),
                "--split",split,"--role","same_scene_development_control",
                "--report",str(run/(track+"_"+label+"_evaluation.json"))],check=True)
    subprocess.run([sys.executable,str(source/"compare_frame_budgets.py"),"--benchmark",str(root)],check=True)


if __name__=="__main__":main()
