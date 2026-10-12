"""Generate transport includes/parameters from fixed vendor inputs, not core patches."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
REVISION = "0645dabe307a99f3b59a5a3d3007acfdcad1a39f"


def verify_vendor():
    receipt = json.loads((ROOT / "vendor_receipt.json").read_text(encoding="utf-8-sig"))
    if receipt["revision"] != REVISION:
        raise ValueError("source revision changed")
    for f in receipt["files"]:
        data = (ROOT / "vendor" / f["path"]).read_bytes()
        if len(data) != f["bytes"] or hashlib.sha256(data).hexdigest() != f["sha256"]:
            raise ValueError("source checksum mismatch: " + f["path"])
    return receipt


def scalar(text):
    text = text.strip()
    if text == "true": return True
    if text == "false": return False
    if text.startswith('"'): return json.loads(text)
    if re.fullmatch(r"[+-]?\d+", text): return int(text)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", text): return float(text)
    raise ValueError("non-scalar or unsupported author YAML: " + text)


def read_waymo_yaml():
    values = {}
    for line in (ROOT / "vendor/config/waymo/waymo1.yaml").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line == "dyn_obj:": continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in values: raise ValueError("duplicate YAML key: " + key)
        values[key] = scalar(value)
    return values


def cpp(value):
    if isinstance(value, bool): return "true" if value else "false"
    if isinstance(value, str): return "std::string(" + json.dumps(value) + ")"
    return repr(value)


def generate():
    verify_vendor()
    values = read_waymo_yaml()
    target = ROOT / "generated"
    target.mkdir(exist_ok=True)
    parameters = ['#pragma once', '#include "compat.hpp"', 'namespace mdetector_candidate {',
                  'inline ros::ParameterMap author_waymo_parameters() {', '  return {']
    for key, value in values.items():
        parameters.append('    {"dyn_obj/' + key + '", ' + cpp(value) + '},')
    parameters.extend(['  };', '}', '}'])
    (target / "author_waymo_parameters.hpp").write_text("\n".join(parameters)+"\n", encoding="utf-8")
    includes = (
        "ros/ros.h", "std_msgs/Header.h", "nav_msgs/Path.h", "nav_msgs/Odometry.h",
        "sensor_msgs/CompressedImage.h", "sensor_msgs/PointCloud2.h",
        "pcl_conversions/pcl_conversions.h", "tf/transform_listener.h",
        "geometry_msgs/PoseWithCovarianceStamped.h", "visualization_msgs/Marker.h",
        "visualization_msgs/MarkerArray.h", "opencv2/opencv.hpp",
        "opencv2/highgui.hpp", "opencv2/highgui/highgui.hpp")
    for include in includes:
        path = target / "transport_include" / include
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#pragma once\n#include "compat.hpp"\n', encoding="utf-8")
    source = (ROOT / "vendor/src/DynObjFilter.cpp").read_text()
    bindings = []
    for match in re.finditer(r'nh\.param<([^>]+)>\("dyn_obj/([^\"]+)",\s*([^,]+),\s*([^;]+)\);', source):
        dtype, key, member, default = match.groups()
        default_value = scalar(re.sub(r"(?<=\d)f$", "", default.strip()))
        bindings.append({"cpp_type": dtype, "yaml_key": key, "member": member.strip(),
                         "author_default": default_value, "yaml_present": key in values,
                         "effective_value": values.get(key, default_value)})
    bound = {b["yaml_key"] for b in bindings}
    audit = {"revision": REVISION, "waymo1_yaml_sha256": hashlib.sha256(
                 (ROOT / "vendor/config/waymo/waymo1.yaml").read_bytes()).hexdigest(),
             "bindings": bindings, "unused_yaml_keys_in_filter_init": sorted(set(values)-bound),
             "core_source_mutations": 0, "config_mutations": 0,
             "scope": "Transport parameter generation only; no detector execution."}
    (target / "parameter_bindings.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    a = generate()
    print(json.dumps({"status":"generated", "bindings":len(a["bindings"]),
                      "unused_yaml_keys":a["unused_yaml_keys_in_filter_init"],
                      "core_source_mutations":0}))
