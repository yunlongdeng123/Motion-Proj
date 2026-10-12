"""Independent guard for one explicit hypothesis, never a formal-resume relaxation.

No training/CUDA imports or launch methods. A reviewed shared harness may install
the adapter only on the isolated replica's unchanged paper_checkpoint module.
"""
import copy
import hashlib
import json
from pathlib import Path


EXTRA_CONFIG = {
    "diagnostic_background_domain_complement": True,
    "diagnostic_lambda_dynamic_background_opacity": 0.05,
}
ALGORITHM_FILES = {"train.py", "utils/paper_losses.py"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def same_value(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    return a == b


def source_inventory(root):
    """Pin every real replica file; ignore only interpreter/git bookkeeping."""
    root = Path(root).resolve()
    files = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in ("__pycache__", ".git") for part in relative.parts) or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_symlink():
            raise ValueError("Source replica contains a symlink: " + str(relative))
        if path.is_file():
            files[relative.as_posix()] = sha(path)
    if not files:
        raise ValueError("Empty source inventory")
    return dict(sorted(files.items()))


class OutsideExperimentGuard:
    def __init__(self, binding):
        self.binding = copy.deepcopy(binding)
        if binding["experiment_kind"] != "hypothesis_complement_background_domain":
            raise ValueError("Not the approved single-domain hypothesis contract")
        if not same_value(binding["added_training_fields"], EXTRA_CONFIG):
            raise ValueError("Only exact explicit hypothesis flag and .05 weight permitted")
        if set(binding["algorithm_changed_files"]) != ALGORITHM_FILES:
            raise ValueError("Only train.py and paper_losses.py algorithm diff permitted")
        for key, value in (("parent_iteration", 30000), ("first_iteration", 30001), ("stop_iteration", 30300), ("horizon", 50000), ("seed", 0)):
            if type(binding[key]) is not type(value) or binding[key] != value:
                raise ValueError("Continuation invariant changed: " + key)
        self.installed = False

    def verify_sources(self):
        original = source_inventory(self.binding["original_repository"])
        replica = source_inventory(self.binding["repository"])
        if original != self.binding["original_source_sha256"]:
            raise ValueError("Actual original source inventory/pin changed")
        if replica != self.binding["replica_source_sha256"]:
            raise ValueError("Actual hypothesis replica source inventory/pin changed")
        if set(original) != set(replica):
            raise ValueError("Replica source inventory differs from original")
        changed = {k for k in original if original[k] != replica[k]}
        if changed != ALGORITHM_FILES:
            raise ValueError("Hypothesis changed files beyond exact two-file diff")
        if sha(self.binding["algorithm_patch"]) != self.binding["algorithm_patch_sha256"]:
            raise ValueError("Reviewed two-file hypothesis patch changed")
        return {"all_source_inventory_sha_pass": True, "original_files": len(original), "replica_files": len(replica), "changed_files": sorted(changed), "formal_resume_helper_unchanged": original["utils/paper_checkpoint.py"] == replica["utils/paper_checkpoint.py"]}

    def verify_protected_files(self):
        for path, expected in self.binding["protected_original_file_sha256"].items():
            if sha(path) != expected:
                raise ValueError("Original source/kernel/data/checkpoint pin changed: " + path)
        return {"protected_original_file_count": len(self.binding["protected_original_file_sha256"]), "all_protected_original_sha_pass": True}

    def verify_auxiliary_files(self):
        for path, expected in self.binding["auxiliary_file_sha256"].items():
            if sha(path) != expected:
                raise ValueError("Reviewed guard/helper/protocol binding changed: " + path)
        return {"auxiliary_file_count": len(self.binding["auxiliary_file_sha256"]), "all_auxiliary_sha_pass": True}

    def validate_config(self, saved_config, current, original_validator):
        if not isinstance(saved_config, dict) or not isinstance(current, dict):
            raise ValueError("Resolved complete dictionaries required")
        parent_saved = same_value(saved_config, self.binding["parent_training_config"])
        variant_saved = same_value(saved_config, self.binding["training_config"])
        if not (parent_saved or variant_saved):
            raise ValueError("Saved config differs from original saved30k or exact bound hypothesis protocol")
        if not same_value(current, self.binding["training_config"]):
            raise ValueError("Current config differs from bound single-variable experiment")
        expected_added = set(EXTRA_CONFIG) if parent_saved else set()
        if set(current) - set(saved_config) != expected_added or set(saved_config) - set(current):
            raise ValueError("Only two explicit experimental fields may be added")
        if current["model_path"] != self.binding["output"] or current["start_checkpoint"] != self.binding["parent_static_checkpoint"]:
            raise ValueError("Replica output or saved30k selector differs")
        for key, value in EXTRA_CONFIG.items():
            if not same_value(current[key], value):
                raise ValueError("Hypothesis field type/value differs: " + key)
        stripped = {key: value for key, value in current.items() if key not in EXTRA_CONFIG}
        stripped_saved = {key: value for key, value in saved_config.items() if key not in EXTRA_CONFIG}
        # Original validator still checks every original field with exact types;
        # only its existing bookkeeping exceptions model_path/start_checkpoint apply.
        original_validator(stripped_saved, stripped)
        return {"original_validator_called": True, "all_original_fields_strictly_validated": True, "saved_configuration_kind": "original_parent" if parent_saved else "same_bound_hypothesis", "added_fields": copy.deepcopy(EXTRA_CONFIG), "strict_original_config": False, "future_iteration_launch_authorized": False}

    def install_on_isolated_module(self, module):
        if self.installed:
            raise RuntimeError("Hypothesis guard already installed")
        expected = (Path(self.binding["repository"]) / "utils/paper_checkpoint.py").resolve()
        if Path(module.__file__).resolve() != expected:
            raise ValueError("Experiment adapter may only bind isolated replica guard module")
        self.verify_sources()
        self.verify_auxiliary_files()
        original = module._validate_resume_configuration

        def validate(saved, args):
            current = module._config_dict(args)
            return self.validate_config(saved, current, original)

        module._validate_resume_configuration = validate
        self.installed = True
        return original


def load_guard(path):
    return OutsideExperimentGuard(json.loads(Path(path).read_text()))
