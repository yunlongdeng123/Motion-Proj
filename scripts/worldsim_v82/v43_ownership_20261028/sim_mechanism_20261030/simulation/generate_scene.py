#!/usr/bin/env python3
"""Blender 3.6 最小前向驾驶仿真；观测输入与独立评估真值分开写盘。

调用示例（先做首帧渲染核验）：
  blender -b --python generate_scene.py -- --output /absolute/preview \
      --frames 16 --frame-ids 0 --reference-keyframes none --samples 16

程序化汽车仅用于非写实机制/接口验证，不支持真实道路泛化结论。
世界坐标以米计，+Z 朝上，车头 +X。相机采用 OpenCV x 右/y 下/z 前。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


SEMANTICS = {0: "sky_or_no_hit", 1: "road", 2: "lane_marking", 3: "roadside", 10: "vehicle"}
EDIT_NAMES = ("none", "delete", "shift", "rotate")


def parse_args():
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame-ids", default="all", help="all 或逗号分隔的帧号；便于先渲染一帧")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--hfov-deg", type=float, default=67.0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("CPU", "CUDA", "OPTIX"), default="CPU")
    parser.add_argument("--seed", type=int, default=431030)
    parser.add_argument("--reference-keyframes", default="auto", help="auto、none 或逗号分隔帧号")
    parser.add_argument("--lidar-beams", type=int, default=24)
    parser.add_argument("--lidar-horizontal-samples", type=int, default=160)
    parser.add_argument("--lidar-max-range", type=float, default=70.0)
    parser.add_argument("--surface-samples-per-class", type=int, default=4096)
    parser.add_argument("--heldout-keyframe", default="auto", help="auto、none 或单个帧号；只写privileged")
    parser.add_argument("--no-save-blend", action="store_true")
    config = parser.parse_args(args)
    if config.frames < 2 or min(config.width, config.height) < 32:
        parser.error("至少两帧，图像各维至少32像素")
    if config.lidar_beams < 1 or config.lidar_horizontal_samples < 2:
        parser.error("LiDAR射线数无效")
    return config


def frame_ids(value, count, auto=False):
    if value == "none":
        return []
    if value == "all":
        return list(range(count))
    if value == "auto" and auto:
        return sorted({0, count // 2, count - 1})
    result = sorted(set(int(item) for item in value.split(",")))
    if any(item < 0 or item >= count for item in result):
        raise ValueError(f"帧号越界：{result}，frames={count}")
    return result


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mat_list(mat):
    return [[float(x) for x in row] for row in mat]


def material(name, color, roughness=0.6, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = (*color, 1.0)
    node.inputs["Roughness"].default_value = roughness
    node.inputs["Metallic"].default_value = metallic
    return mat


def asphalt_material():
    mat = material("asphalt_world_noise", (0.12, 0.13, 0.14), roughness=0.98)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    geo = nodes.new("ShaderNodeNewGeometry")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.5
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.75
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.12
    ramp.color_ramp.elements[0].color = (0.045, 0.052, 0.06, 1.0)
    ramp.color_ramp.elements[1].position = 0.88
    ramp.color_ramp.elements[1].color = (0.17, 0.18, 0.19, 1.0)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.18
    bump.inputs["Distance"].default_value = 0.015
    links.new(geo.outputs["Position"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], nodes.get("Principled BSDF").inputs["Normal"])
    return mat


def tag_object(obj, object_id, semantic_id, parent=None):
    obj["object_id"] = int(object_id)
    obj["semantic_id"] = int(semantic_id)
    obj["dynamic_vehicle"] = bool(semantic_id == 10)
    if parent is not None:
        obj.parent = parent
    return obj


def cube(name, center, dimensions, mat, object_id, semantic_id, parent=None, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        modifier = obj.modifiers.new("rounded_manufactured_edges", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        modifier = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.data.materials.append(mat)
    return tag_object(obj, object_id, semantic_id, parent)


def mesh_object(name, vertices, faces, mat, object_id, semantic_id, parent=None):
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return tag_object(obj, object_id, semantic_id, parent)


def cylinder(name, center, radius, depth, mat, object_id, parent, vertices=32):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler[0] = math.pi / 2.0  # 轮轴沿世界/车辆局部 y 轴
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = len(polygon.vertices) == 4
    return tag_object(obj, object_id, 10, parent)


def build_scene(config):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.render.engine = "CYCLES"
    scene.cycles.samples = config.samples
    # 配对参考要求相同物理场景逐像素一致。r1的OIDN/8bit输出出现了少量1LSB差异；
    # r2固定sample预算并关闭后处理与量化抖动，保留严格相同场景检查。
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces = 5
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.seed = config.seed
    scene.render.threads_mode = "FIXED"
    scene.render.threads = config.threads
    scene.render.resolution_x = config.width
    scene.render.resolution_y = config.height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.camera = None
    scene.world = bpy.data.worlds.new("blue_sky_illumination")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.34, 0.50, 0.72, 1.0)
    background.inputs["Strength"].default_value = 0.55

    device_description = "CPU"
    scene.cycles.device = "CPU"
    if config.device != "CPU":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = config.device
        preferences.get_devices()
        devices = [device for device in preferences.devices if device.type == config.device]
        if not devices:
            raise RuntimeError(f"请求{config.device}但Cycles未发现对应设备")
        for device in preferences.devices:
            device.use = device in devices
        scene.cycles.device = "GPU"
        device_description = ", ".join(device.name for device in devices)

    road = asphalt_material()
    line = material("off_white_lane_paint", (0.72, 0.73, 0.69), 0.93)
    curb = material("concrete", (0.36, 0.34, 0.30), 0.94)
    grass = material("dry_roadside", (0.20, 0.24, 0.13), 1.0)
    wall = material("background_masonry", (0.40, 0.36, 0.31), 0.95)
    red = material("vehicle_red_paint", (0.53, 0.018, 0.012), 0.28, 0.32)
    glass = material("opaque_dark_glass_sensor_proxy", (0.035, 0.075, 0.105), 0.16, 0.25)
    rubber = material("rubber", (0.012, 0.014, 0.017), 0.89)
    metal = material("wheel_alloy", (0.31, 0.33, 0.35), 0.31, 0.72)
    black = material("dark_trim", (0.015, 0.019, 0.023), 0.67)
    rear_light = material("rear_lamp", (0.28, 0.008, 0.006), 0.2)
    head_light = material("head_lamp", (0.72, 0.74, 0.68), 0.22, 0.1)

    cube("road_asphalt", (15.0, 0.0, -0.15), (80.0, 10.4, 0.30), road, 1, 1)
    cube("roadside_ground", (15.0, 0.0, -0.26), (100.0, 55.0, 0.16), grass, 2, 3)
    object_id = 10
    for side in (-1.0, 1.0):
        cube(f"curb_{side:+g}", (15.0, side * 5.3, 0.10), (80.0, 0.18, 0.20), curb, object_id, 3)
        object_id += 1
        cube(f"walkway_{side:+g}", (15.0, side * 6.0, 0.04), (80.0, 1.2, 0.10), curb, object_id, 3)
        object_id += 1
    for y in (-4.55, 4.55):
        cube(f"solid_lane_{y:+g}", (15.0, y, 0.006), (76.0, 0.09, 0.010), line, object_id, 2)
        object_id += 1
    for x in np.arange(-20.0, 55.0, 5.0):
        cube(f"dashed_center_{x:+g}", (float(x), 0.0, 0.006), (2.7, 0.10, 0.010), line, object_id, 2)
        object_id += 1
    # 静态接缝和局部补丁提供真实多视角纹理，不添加污染Gaussian或人工伪影。
    patch = material("asphalt_patch", (0.075, 0.078, 0.08), 0.99)
    for j, (x, y, sx, sy) in enumerate(((2.5, -1.4, 1.1, 0.7), (8, 2.8, 2.0, 0.4), (14, -2.2, 1.4, 0.85))):
        cube(f"road_patch_{j}", (x, y, 0.003), (sx, sy, 0.004), patch, object_id, 1)
        object_id += 1
    for j, x in enumerate((7.0, 18.0, 31.0)):
        for side in (-1.0, 1.0):
            height = 1.6 + 0.5 * (j % 2)
            cube(f"background_block_{j}_{side:+g}", (x, side * 10.5, height / 2 - 0.05), (4.0, 3.0, height), wall, object_id, 3)
            object_id += 1

    car = bpy.data.objects.new("vehicle_track_0", None)
    bpy.context.collection.objects.link(car)
    cube("vehicle_underbody", (0, 0, 0.47), (4.25, 1.78, 0.24), black, 100, 10, car, 0.08)
    cube("vehicle_body", (0, 0, 0.82), (4.42, 1.86, 0.69), red, 101, 10, car, 0.16)
    cube("vehicle_hood", (1.34, 0, 1.105), (1.52, 1.74, 0.16), red, 102, 10, car, 0.07)
    cube("vehicle_rear_deck", (-1.68, 0, 1.06), (0.85, 1.72, 0.17), red, 103, 10, car, 0.06)
    # 不透明玻璃代理避免光学折射与first-hit深度之间出现隐性不一致。
    cabin_vertices = [(-1.31, -0.80, 1.04), (-1.31, 0.80, 1.04), (1.00, 0.80, 1.04), (1.00, -0.80, 1.04),
                      (-0.77, -0.71, 1.68), (-0.77, 0.71, 1.68), (0.47, 0.71, 1.68), (0.47, -0.71, 1.68)]
    mesh_object("vehicle_glass_cabin", cabin_vertices, [(1, 2, 3, 0), (4, 5, 1, 0), (2, 6, 7, 3), (3, 7, 4, 0), (5, 6, 2, 1), (7, 6, 5, 4)], glass, 104, 10, car)
    cube("vehicle_roof", (-0.15, 0.0, 1.70), (1.39, 1.49, 0.105), red, 105, 10, car, 0.045)
    # B柱、门槛和后视镜让车体拥有可见的独立表面部件。
    for side in (-1, 1):
        cube(f"vehicle_Bpillar_{side}", (-0.17, side * 0.777, 1.36), (0.10, 0.055, 0.59), black, 106 if side < 0 else 107, 10, car)
        cube(f"vehicle_sill_{side}", (0.0, side * 0.921, 0.59), (3.50, 0.06, 0.12), red, 108 if side < 0 else 109, 10, car, 0.015)
        cube(f"vehicle_mirror_{side}", (0.70, side * 1.02, 1.18), (0.22, 0.25, 0.14), red, 110 if side < 0 else 111, 10, car, 0.035)
    wheel_id = 120
    for x in (-1.37, 1.38):
        for side in (-1, 1):
            cylinder(f"tire_{x:+g}_{side}", (x, side * 0.91, 0.39), 0.39, 0.24, rubber, wheel_id, car)
            cylinder(f"alloy_{x:+g}_{side}", (x, side * 1.037, 0.39), 0.235, 0.015, metal, wheel_id + 1, car)
            wheel_id += 2
    for side in (-1, 1):
        cube(f"rear_lamp_{side}", (-2.217, side * 0.59, 0.95), (0.035, 0.43, 0.17), rear_light, 140 if side < 0 else 141, 10, car, 0.015)
        cube(f"head_lamp_{side}", (2.217, side * 0.59, 0.93), (0.035, 0.43, 0.19), head_light, 142 if side < 0 else 143, 10, car, 0.015)
    cube("rear_number_plate", (-2.225, 0, 0.69), (0.02, 0.48, 0.15), head_light, 144, 10, car)

    bpy.ops.object.light_add(type="SUN", location=(-4.0, -5.0, 9.0))
    sun = bpy.context.object
    sun.name = "fixed_sun"
    sun.rotation_euler = (math.radians(28.0), math.radians(-23.0), math.radians(-31.0))
    sun.data.energy = 2.7
    sun.data.angle = math.radians(7.0)

    bpy.ops.object.camera_add(location=(-4.5, -3.0, 2.2))
    camera = bpy.context.object
    camera.name = "front_camera"
    camera.data.type = "PERSP"
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0
    camera.data.lens = 18.0 / math.tan(math.radians(config.hfov_deg) / 2.0)
    camera.data.clip_start = 0.05
    camera.data.clip_end = 150.0
    camera.data.lens_unit = "MILLIMETERS"
    scene.camera = camera
    bpy.context.view_layer.update()
    objects = sorted((obj for obj in scene.objects if obj.type == "MESH"), key=lambda obj: int(obj["object_id"]))
    return scene, camera, car, objects, device_description


def set_frame(frame_id, config, camera, car, edit="none", shadow=True):
    t = frame_id / float(config.frames - 1)
    center = Vector((6.0 + 4.0 * t, 0.85 + 0.16 * math.sin(math.pi * t), 0.0))
    yaw = 0.045 * math.sin(math.pi * t)
    car.location = center
    car.rotation_euler = (0, 0, yaw)
    if edit == "shift":
        car.location.y += 1.5
    elif edit == "rotate":
        car.rotation_euler.z += math.radians(30.0)
    # 相机姿态仅依原始轨迹，编辑不会重新瞄准汽车。
    camera.location = (-4.5 + 3.5 * t, -3.0 + 0.7 * t, 2.20 + 0.06 * math.sin(math.pi * t))
    target = center + Vector((0.5, 0.0, 0.85))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    for obj in car.children:
        obj.hide_render = edit == "delete"
        # 被删除车辆的shadow开关物理上无作用，不再向渲染器发送无意义的不同visibility状态。
        obj.visible_shadow = True if edit == "delete" else bool(shadow)
    bpy.context.scene.cycles.seed = config.seed + frame_id  # 相同帧的所有编辑使用相同采样种子
    bpy.context.scene.frame_set(frame_id + 1)
    bpy.context.view_layer.update()


def camera_metadata(frame_id, config, camera):
    fx = config.width / (2.0 * math.tan(math.radians(config.hfov_deg) / 2.0))
    # 像素坐标以左上角为(0,0)，射线穿过(u+0.5,v+0.5)。
    intrinsics = [[fx, 0.0, config.width / 2.0], [0.0, fx, config.height / 2.0], [0.0, 0.0, 1.0]]
    blender_to_cv = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
    c2w = camera.matrix_world @ blender_to_cv
    return {"id": frame_id, "frame_id": frame_id, "timestamp_s": frame_id / config.fps,
            "width": config.width, "height": config.height, "K": intrinsics, "intrinsics": intrinsics,
            "camera_to_world": mat_list(c2w), "world_to_camera": mat_list(c2w.inverted())}


class IndependentGeometry:
    """从仿真mesh建立固定三角面编号；不依赖任何重建Gaussian。"""

    def __init__(self, objects):
        self.objects = objects
        self.parts = []
        vertices, vertex_owner, triangles = [], [], []
        triangle_owner, semantic = [], []
        offset = 0
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in objects:
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            mesh.calc_loop_triangles()
            local = np.asarray([vertex.co[:] for vertex in mesh.vertices], dtype=np.float64)
            faces = np.asarray([triangle.vertices[:] for triangle in mesh.loop_triangles], dtype=np.int32)
            part = {"object": obj, "local": local, "triangles": faces,
                    "surface_id": np.arange(len(triangle_owner) + 1, len(triangle_owner) + len(faces) + 1, dtype=np.int32)}
            self.parts.append(part)
            vertices.extend(local.tolist())
            vertex_owner.extend([int(obj["object_id"])] * len(local))
            triangles.extend((faces + offset).tolist())
            triangle_owner.extend([int(obj["object_id"])] * len(faces))
            semantic.extend([int(obj["semantic_id"])] * len(faces))
            offset += len(local)
            evaluated.to_mesh_clear()
        self.vertices_local = np.asarray(vertices, dtype=np.float32)
        self.vertex_object_id = np.asarray(vertex_owner, dtype=np.int32)
        self.triangles = np.asarray(triangles, dtype=np.int32)
        self.triangle_object_id = np.asarray(triangle_owner, dtype=np.int32)
        self.triangle_semantic_id = np.asarray(semantic, dtype=np.int16)
        self.surface_ids = np.arange(1, len(triangles) + 1, dtype=np.int32)

    def save_mesh(self, path):
        np.savez_compressed(path, vertices_local=self.vertices_local, vertex_object_id=self.vertex_object_id,
                            triangles=self.triangles, surface_id=self.surface_ids,
                            triangle_object_id=self.triangle_object_id, triangle_semantic_id=self.triangle_semantic_id)

    def build_bvh(self):
        vertices, triangles, surface, owner, semantic = [], [], [], [], []
        for part in self.parts:
            obj = part["object"]
            if obj.hide_render:
                continue
            transform = np.asarray(obj.matrix_world, dtype=np.float64)
            world = part["local"] @ transform[:3, :3].T + transform[:3, 3]
            offset = len(vertices)
            vertices.extend(world.tolist())
            triangles.extend((part["triangles"] + offset).tolist())
            surface.extend(part["surface_id"].tolist())
            owner.extend([int(obj["object_id"])] * len(part["triangles"]))
            semantic.extend([int(obj["semantic_id"])] * len(part["triangles"]))
        self.bvh = BVHTree.FromPolygons(vertices, triangles, all_triangles=True, epsilon=0.0)
        self.hit_surface = np.asarray(surface, dtype=np.int32)
        self.hit_owner = np.asarray(owner, dtype=np.int32)
        self.hit_semantic = np.asarray(semantic, dtype=np.int16)

    def cast(self, origin, directions, max_range):
        origin = Vector(origin)
        count = len(directions)
        ranges = np.zeros(count, dtype=np.float32)
        xyz = np.zeros((count, 3), dtype=np.float32)
        normals = np.zeros((count, 3), dtype=np.float32)
        surface = np.zeros(count, dtype=np.int32)
        owner = np.zeros(count, dtype=np.int32)
        semantic = np.zeros(count, dtype=np.int16)
        for index, direction in enumerate(directions):
            point, normal, face, distance = self.bvh.ray_cast(origin, Vector(direction), max_range)
            if face is None:
                continue
            ranges[index] = distance
            xyz[index] = point
            normals[index] = normal
            surface[index] = self.hit_surface[face]
            owner[index] = self.hit_owner[face]
            semantic[index] = self.hit_semantic[face]
        return {"range_m": ranges, "xyz_world": xyz, "normal_world": normals,
                "surface_id": surface, "object_id": owner, "semantic_id": semantic}

    def save_surface_samples(self, path, per_class, seed):
        rng = np.random.default_rng(seed)
        result = {key: [] for key in ("xyz_object", "xyz_world_frame0", "normal_object", "object_id", "semantic_id", "surface_id", "barycentric", "area_weight_m2")}
        for semantic_id in (1, 2, 3, 10):
            candidates = []
            for part in self.parts:
                if int(part["object"]["semantic_id"]) != semantic_id:
                    continue
                points = part["local"][part["triangles"]]
                areas = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=-1) / 2
                for j, area in enumerate(areas):
                    if area > 1e-10:
                        candidates.append((part, j, float(area)))
            weights = np.asarray([candidate[2] for candidate in candidates])
            total_area = float(weights.sum())
            selected = rng.choice(len(candidates), size=per_class, p=weights / total_area)
            r = rng.random((per_class, 2))
            root = np.sqrt(r[:, 0])
            barycentric = np.stack((1 - root, root * (1 - r[:, 1]), root * r[:, 1]), axis=1)
            for index, selected_index in enumerate(selected):
                part, triangle_index, _ = candidates[selected_index]
                points = part["local"][part["triangles"][triangle_index]]
                point = barycentric[index] @ points
                normal = np.cross(points[1] - points[0], points[2] - points[0])
                normal /= np.linalg.norm(normal)
                transform = np.asarray(part["object"].matrix_world)
                result["xyz_object"].append(point)
                result["xyz_world_frame0"].append(transform[:3, :3] @ point + transform[:3, 3])
                result["normal_object"].append(normal)
                result["object_id"].append(int(part["object"]["object_id"]))
                result["semantic_id"].append(semantic_id)
                result["surface_id"].append(int(part["surface_id"][triangle_index]))
                result["barycentric"].append(barycentric[index])
                result["area_weight_m2"].append(total_area / per_class)
        arrays = {key: np.asarray(value, dtype=np.int32 if key in ("object_id", "surface_id", "semantic_id") else np.float32) for key, value in result.items()}
        np.savez_compressed(path, **arrays)

    def object_poses(self):
        return {str(int(obj["object_id"])): mat_list(obj.matrix_world) for obj in self.objects}


def dense_raycast(geometry, camera_info, config):
    k = np.asarray(camera_info["K"], dtype=np.float64)
    c2w = np.asarray(camera_info["camera_to_world"], dtype=np.float64)
    yy, xx = np.meshgrid(np.arange(config.height) + 0.5, np.arange(config.width) + 0.5, indexing="ij")
    rays = np.stack(((xx - k[0, 2]) / k[0, 0], (yy - k[1, 2]) / k[1, 1], np.ones_like(xx)), axis=-1).reshape(-1, 3)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    hit = geometry.cast(c2w[:3, 3], rays @ c2w[:3, :3].T, 150.0)
    # camera-z 是沿相机光轴的深度；range 是到光心的欧氏距离。
    hit["depth_z_m"] = hit["range_m"] * rays[:, 2]
    hit["valid"] = hit["surface_id"] > 0
    for key, value in list(hit.items()):
        hit[key] = value.reshape((config.height, config.width) + value.shape[1:])
    return hit


def save_mask(path, mask):
    height, width = mask.shape
    image = bpy.data.images.new("temporary_binary_mask", width=width, height=height, alpha=True, is_data=True)
    # Blender图片buffer底行在前，numpy的相机像素坐标顶行在前。
    data = np.repeat(np.flipud(mask.astype(np.float32))[..., None], 4, axis=-1)
    data[..., 3] = 1.0
    image.pixels.foreach_set(data.ravel())
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def lidar_raycast(geometry, camera_info, config):
    # 扫描线垂直角非均匀覆盖车辆和近地面；不从dense depth抽点，也不保存owner/表面标签。
    c2w = np.asarray(camera_info["camera_to_world"], dtype=np.float64)
    horizontal = np.linspace(-config.hfov_deg / 2, config.hfov_deg / 2, config.lidar_horizontal_samples)
    vertical = np.linspace(-12.0, 24.0, config.lidar_beams)  # 相机y向下：正角朝地面
    vv, hh = np.meshgrid(np.deg2rad(vertical), np.deg2rad(horizontal), indexing="ij")
    directions = np.stack((np.tan(hh), np.tan(vv), np.ones_like(hh)), axis=-1).reshape(-1, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    hit = geometry.cast(c2w[:3, 3], directions @ c2w[:3, :3].T, config.lidar_max_range)
    valid = hit["range_m"] > 0
    xyz = hit["xyz_world"][valid]
    z = hit["range_m"][valid] * directions[valid, 2]
    k = np.asarray(camera_info["K"])
    uv = directions[valid, :2] / directions[valid, 2:] * (k[0, 0], k[1, 1]) + (k[0, 2], k[1, 2])
    return {"xyz": xyz, "xyz_world": xyz, "intensity": np.ones(len(xyz), dtype=np.float32),
            "depth_z_m": z.astype(np.float32), "range_m": hit["range_m"][valid],
            "uv": uv.astype(np.float32), "beam_index": np.repeat(np.arange(config.lidar_beams), config.lidar_horizontal_samples)[valid].astype(np.int16),
            "sensor_to_world": c2w.astype(np.float64)}


def render(scene, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(path)
    before = time.monotonic()
    bpy.ops.render.render(write_still=True)
    return time.monotonic() - before


def renderer_contract():
    return {"denoising": False, "adaptive_sampling": False, "animated_seed": False,
            "output_dither_intensity": 0.0, "sampling_seed": "base_seed + frame_id, same for all edits of a frame",
            "delete_shadow_state": "all vehicle parts hide_render=True; visible_shadow canonical True for both requested states",
            "delete_pair_acceptance": "strict decoded RGB maximum difference == 0; no tolerance and no shared-render copy",
            "shadow_pair_interpretation": "finite-sample RGB references with common seed; per-pixel difference may include Monte Carlo estimator variation and is not pure shadow ground truth; no sample-convergence claim"}


def track_metadata(frame_id, config, car, body_mask):
    yy, xx = np.where(body_mask)
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1] if len(xx) else None
    return {"frame_id": frame_id, "timestamp_s": frame_id / config.fps, "track_id": 0,
            "label": "vehicle", "object_to_world": mat_list(car.matrix_world),
            "box_center_world": [float(car.location.x), float(car.location.y), 0.86],
            "box_size_lwh_m": [4.50, 2.13, 1.76], "heading_world_rad": float(car.rotation_euler.z),
            "bbox_xyxy": bbox, "source": "simulator_sensor_oracle"}


def main():
    config = parse_args()
    output = config.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"输出目录必须不存在或为空，防止混合旧现场：{output}")
    selected = frame_ids(config.frame_ids, config.frames)
    references = frame_ids(config.reference_keyframes, config.frames, auto=True)
    if not set(references).issubset(selected):
        raise ValueError("reference keyframes必须属于本次--frame-ids")
    started = time.monotonic()
    obs = output / "observations"
    privileged = output / "privileged"
    for directory in (obs / "rgb", obs / "body_masks", obs / "sky_masks", obs / "lidar", privileged / "original", privileged / "poses", privileged / "geometry"):
        directory.mkdir(parents=True, exist_ok=True)
    scene, camera, car, objects, device = build_scene(config)
    set_frame(0, config, camera, car)
    geometry = IndependentGeometry(objects)
    geometry.save_mesh(privileged / "geometry" / "mesh_reference.npz")
    geometry.save_surface_samples(privileged / "geometry" / "fixed_surface_samples.npz", config.surface_samples_per_class, config.seed)
    object_catalog = [{"object_id": int(obj["object_id"]), "name": obj.name,
                       "semantic_id": int(obj["semantic_id"]), "dynamic": bool(obj["dynamic_vehicle"])} for obj in objects]
    dump_json(privileged / "geometry" / "objects.json", {"semantics": SEMANTICS, "objects": object_catalog,
              "surface_identity": "stable evaluated-mesh triangle index + 1; zero is miss; independent of reconstruction",
              "vertex_coordinates": "each vertex is in the local frame of vertex_object_id; apply per-frame object_to_world"})
    cameras, frames, tracks, timing, reference_entries = [], [], [], [], []
    for frame_id in selected:
        frame_started = time.monotonic()
        stem = f"{frame_id:06d}"
        set_frame(frame_id, config, camera, car, "none", True)
        calibration = camera_metadata(frame_id, config, camera)
        geometry.build_bvh()
        dense = dense_raycast(geometry, calibration, config)
        body_mask = dense["semantic_id"] == 10
        sky_mask = ~dense["valid"]
        save_mask(obs / "body_masks" / (stem + ".png"), body_mask)
        save_mask(obs / "sky_masks" / (stem + ".png"), sky_mask)
        lidar = lidar_raycast(geometry, calibration, config)
        np.savez_compressed(obs / "lidar" / (stem + ".npz"), **lidar)
        np.savez_compressed(privileged / "original" / (stem + ".npz"), **dense)
        dump_json(privileged / "poses" / (stem + ".json"), {"frame_id": frame_id, "object_to_world": geometry.object_poses()})
        seconds = render(scene, obs / "rgb" / (stem + ".png"))
        frames.append({**calibration, "rgb": f"rgb/{stem}.png", "body_mask": f"body_masks/{stem}.png",
                       "sky_mask": f"sky_masks/{stem}.png", "lidar": f"lidar/{stem}.npz"})
        cameras.append(calibration)
        tracks.append(track_metadata(frame_id, config, car, body_mask))
        timing.append({"frame_id": frame_id, "render_s": seconds, "total_s": time.monotonic() - frame_started,
                       "lidar_points": int(len(lidar["xyz"])), "vehicle_pixels": int(body_mask.sum()), "sky_pixels": int(sky_mask.sum())})
        # 每帧就刷新观测manifest，首帧完成即可做接口/图像核验。
        dump_json(obs / "manifest.json", observation_manifest(config, frames, complete=False))
        dump_json(obs / "cameras.json", {"convention": coordinate_contract(), "frames": cameras})
        dump_json(obs / "ideal_tracks.json", {"source": "simulator_sensor_oracle", "tracks": tracks})
        print(f"SIM_FRAME_READY {stem} points={len(lidar['xyz'])} car_pixels={int(body_mask.sum())} render_s={seconds:.2f} total_s={timing[-1]['total_s']:.2f}", flush=True)

    # 原始序列全部保存独立first-hit depth/surface；编辑参考只对关键帧。
    for frame_id in selected:
        stem = f"{frame_id:06d}"
        destination = privileged / "references" / "none" / "shadow_on" / "rgb" / (stem + ".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(obs / "rgb" / (stem + ".png"), destination)
        reference_entries.append({"frame_id": frame_id, "edit": "none", "vehicle_shadow": True,
                                  "rgb": str(destination.relative_to(privileged)), "geometry": f"original/{stem}.npz"})
    for frame_id in references:
        stem = f"{frame_id:06d}"
        for edit in EDIT_NAMES:
            set_frame(frame_id, config, camera, car, edit, True)
            calibration = camera_metadata(frame_id, config, camera)
            geometry_path = privileged / "references" / edit / "geometry" / (stem + ".npz")
            geometry_path.parent.mkdir(parents=True, exist_ok=True)
            if edit == "none":
                shutil.copy2(privileged / "original" / (stem + ".npz"), geometry_path)
            else:
                geometry.build_bvh()
                np.savez_compressed(geometry_path, **dense_raycast(geometry, calibration, config))
            dump_json(geometry_path.with_suffix(".poses.json"), {"frame_id": frame_id, "edit": edit, "object_to_world": geometry.object_poses()})
            for shadow in (True, False):
                if edit == "none" and shadow:
                    continue
                set_frame(frame_id, config, camera, car, edit, shadow)
                rgb_path = privileged / "references" / edit / ("shadow_on" if shadow else "shadow_off") / "rgb" / (stem + ".png")
                seconds = render(scene, rgb_path)
                reference_entries.append({"frame_id": frame_id, "edit": edit, "vehicle_shadow": shadow,
                                          "rgb": str(rgb_path.relative_to(privileged)), "geometry": str(geometry_path.relative_to(privileged)), "render_s": seconds})
                print(f"SIM_REFERENCE_READY frame={stem} edit={edit} shadow={shadow} render_s={seconds:.2f}", flush=True)
    heldout_entries = []
    heldout_frame = config.frames // 2 if config.heldout_keyframe == "auto" else (None if config.heldout_keyframe == "none" else int(config.heldout_keyframe))
    if heldout_frame is not None and heldout_frame in selected:
        stem = f"{heldout_frame:06d}"
        set_frame(heldout_frame, config, camera, car, "none", True)
        camera.location.y += 1.0
        target = car.location + Vector((0.5, 0.0, 0.85))
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        calibration = camera_metadata(heldout_frame, config, camera)
        geometry.build_bvh()
        dense = dense_raycast(geometry, calibration, config)
        heldout_path = privileged / "heldout"
        heldout_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(heldout_path / (stem + ".npz"), **dense)
        seconds = render(scene, heldout_path / (stem + ".png"))
        heldout_entries.append({**calibration, "view_id": "lateral_y_plus_1m", "rgb": f"heldout/{stem}.png",
                                "geometry": f"heldout/{stem}.npz", "world_camera_translation_delta_m": [0.0, 1.0, 0.0],
                                "train_input": False, "render_s": seconds})
        dump_json(heldout_path / "manifest.json", {"purpose": "privileged_only_novel_camera_validation", "frames": heldout_entries})
        print(f"SIM_HELDOUT_READY frame={stem} render_s={seconds:.2f}", flush=True)
    set_frame(0, config, camera, car, "none", True)
    if not config.no_save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(privileged / "scene.blend"))
    dump_json(obs / "manifest.json", observation_manifest(config, frames, complete=True))
    dump_json(privileged / "reference_manifest.json", {"schema": "v43_sim_reference_v1", "references": reference_entries,
              "keyframes": references, "heldout": heldout_entries, "edits": {"none": {}, "delete": {"hide_vehicle": True}, "shift": {"translation_world_m": [0, 1.5, 0]}, "rotate": {"yaw_delta_deg": 30.0}},
              "renderer_contract": renderer_contract(),
              "shadow_contract": "Only dynamic vehicle Cycles visible_shadow is toggled; this removes vehicle cast and self shadow visibility. Static geometry/lights/exposure/seed remain fixed.",
              "geometry_contract": "first-hit geometric surfaces; independent of reconstruction; PNG anti-aliased edges can differ from center-ray IDs"})
    receipt = {"schema": "v43_sim_dataset_v1", "status": "complete", "output": str(output),
               "blender_version": bpy.app.version_string, "render_device": device,
               "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()},
               "observations_manifest": "observations/manifest.json", "privileged_reference_manifest": "privileged/reference_manifest.json",
               "coordinate_contract": coordinate_contract(), "asset_provenance": "procedural vehicle constructed in generate_scene.py; no downloaded vehicle mesh",
               "renderer_contract": renderer_contract(),
               "evidence_scope": "non-photoreal forward-simulation mechanism/interface validation; no real-world generalization claim",
               "no_synthetic_reconstruction_contamination": True, "timing": timing, "elapsed_s": time.monotonic() - started,
               "failure_ledger_refs": [], "failure_ledger_delta": "none"}
    dump_json(output / "dataset_manifest.json", receipt)
    print("SIM_DATASET_COMPLETE " + json.dumps({"output": str(output), "elapsed_s": receipt["elapsed_s"], "frames": len(frames), "references": len(reference_entries)}), flush=True)


def coordinate_contract():
    return {"unit": "metre", "world_axes": "+X vehicle forward, +Y vehicle left, +Z up",
            "camera_axes": "OpenCV +x right, +y down, +z forward", "matrix_layout": "row-major 4x4; column homogeneous points",
            "projection": "u=fx*x/z+cx; v=fy*y/z+cy; ray through pixel center (column+0.5,row+0.5)",
            "depth_z_m": "camera forward z coordinate", "range_m": "Euclidean sensor-origin to hit distance",
            "invalid_depth": 0, "sky_definition": "no scene-mesh hit within 150m", "lidar_coordinates": "xyz and xyz_world are identical world-space metres"}


def observation_manifest(config, frames, complete):
    return {"schema": "v43_sim_observations_v1", "status": "complete" if complete else "in_progress", "frames": frames,
            "cameras_path": "cameras.json", "tracks_path": "ideal_tracks.json", "coordinate_contract": coordinate_contract(),
            "renderer_contract": renderer_contract(),
            "declared_input_mode": "ideal_sensor_and_tracks_interface_validation",
            "sensor_oracle_inputs": ["body_masks", "sky_masks", "ideal_tracks", "camera_calibration", "noise_free_sparse_raycast_lidar"],
            "oracle_scope": "body/sky masks and tracks are simulator sensor oracles, not deployed RGB predictions; mask excludes cast shadow",
            "training_input_whitelist": ["rgb/*.png", "body_masks/*.png", "sky_masks/*.png", "lidar/*.npz", "cameras.json", "ideal_tracks.json", "manifest.json"],
            "privileged_inputs_forbidden": ["dense_depth", "surface_id", "object_id", "scene_mesh", "surface_samples", "edited_reference_rgb"],
            "source_scene": "procedural_non_photoreal_driving_scene", "nominal_sequence_frames": config.frames,
            "rendered_frame_ids": [frame["id"] for frame in frames], "width": config.width, "height": config.height,
            "lidar": {"type": "forward_sparse_multibeam_raycast", "beams": config.lidar_beams,
                      "horizontal_samples": config.lidar_horizontal_samples, "max_range_m": config.lidar_max_range,
                      "measurement_noise": "none; declared ideal sensor condition", "owner_labels_in_input": False}}


if __name__ == "__main__":
    main()
