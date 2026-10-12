#!/usr/bin/env python3
"""Exercise the actual checkpoint restore block without allocating CUDA data."""
import argparse
import ast
import os
from pathlib import Path
import textwrap
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("train_py", type=Path)
cli = parser.parse_args()
tree = ast.parse(cli.train_py.read_text(encoding="utf-8"))
training = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "training")
restore = next(n for n in training.body if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute)
               and n.test.attr == "start_checkpoint")
code = "def restore(args, gaussians, dynamic_gaussians, env_map):\n    first_iter = 0\n"
code += textwrap.indent(ast.unparse(restore), "    ") + "\n    return first_iter\n"


class Model:
    def __init__(self):
        self.received = []

    def restore(self, *args):
        self.received.append(args)


def case(dynamic=False, dynamic_iter=100, light_iter=100):
    checkpoint = "/case/chkpnt100.pth"
    values = {checkpoint: ("static_parameters", 100),
              "/case/dynamic_chkpnt100.pth": ("dynamic_parameters", dynamic_iter),
              "/case/env_light_chkpnt100.pth": ("environment_parameters", light_iter)}
    loaded = []

    def load(path):
        loaded.append(path)
        return values[path]

    namespace = {"torch": SimpleNamespace(load=load), "os": os}
    exec(compile(code, "<actual train.py resume block>", "exec"), namespace)
    args = SimpleNamespace(start_checkpoint=checkpoint)
    static, actor, sky = Model(), Model() if dynamic else None, Model()
    iteration = namespace["restore"](args, static, actor, sky)
    assert iteration == 100
    assert static.received == [("static_parameters", args)]
    assert sky.received == [("environment_parameters", args)]  # optimizer must bind restored base
    if actor is not None:
        assert actor.received == [("dynamic_parameters", args)]
    else:
        assert "/case/dynamic_chkpnt100.pth" not in loaded


case(dynamic=False)
case(dynamic=True)
for dynamic, dynamic_iter, light_iter in [(True, 99, 100), (False, 100, 99)]:
    try:
        case(dynamic, dynamic_iter, light_iter)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched component iteration must be rejected")
print("PASS: static/dynamic checkpoint routing, environment optimizer binding, mismatch rejection")
