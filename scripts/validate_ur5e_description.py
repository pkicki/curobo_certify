#!/usr/bin/env python3
"""Generate and independently FK-check official UR5e ROS2 description xacro."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml

PIN = '78c5afd8bfee78a459c0265304340a85f1d44792'
JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _degrees(loader, node):
    return math.radians(float(loader.construct_scalar(node)))


class _LimitLoader(yaml.SafeLoader):
    pass


_LimitLoader.add_constructor('!degrees', _degrees)


def rot_rpy(rpy):
    roll, pitch, yaw = rpy or (0.0, 0.0, 0.0)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def axis_rotation(axis, angle):
    axis = np.asarray(axis if axis is not None else [1.0, 0.0, 0.0], dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, v = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([[x*x*v+c, x*y*v-z*s, x*z*v+y*s],
                     [y*x*v+z*s, y*y*v+c, y*z*v-x*s],
                     [z*x*v-y*s, z*y*v+x*s, z*z*v+c]])


def parser_fk(robot, base, target, positions):
    by_child = {j.child: j for j in robot.joints}
    chain = []
    link = target
    while link != base:
        joint = by_child[link]
        chain.append(joint)
        link = joint.parent
    transform = np.eye(4)
    for joint in reversed(chain):
        origin = joint.origin
        block = np.eye(4)
        block[:3, :3] = rot_rpy(origin.rpy if origin else None)
        block[:3, 3] = np.asarray(origin.xyz if origin else [0, 0, 0], dtype=float)
        transform = transform @ block
        if joint.type in ('revolute', 'continuous'):
            rotation = np.eye(4)
            rotation[:3, :3] = axis_rotation(joint.axis, positions[joint.name])
            transform = transform @ rotation
    return transform


def generate(official: Path, urdf_out: Path, report_out: Path, *, samples=250, seed=20261111):
    sha_process = subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip()
    if sha_process != PIN:
        raise ValueError(f'official source revision mismatch: {sha_process} != {PIN}')
    xacro = shutil.which('xacro')
    if not xacro:
        raise RuntimeError('xacro executable not found in current environment')
    config = official / 'config/ur5e'
    limits_file = config / 'joint_limits.yaml'
    kine_file = config / 'default_kinematics.yaml'
    limit_data = yaml.load(limits_file.read_text(), Loader=_LimitLoader)['joint_limits']
    with tempfile.TemporaryDirectory(prefix='ur5e_xacro_overlay_') as tmp:
        overlay = Path(tmp) / 'ur_description'
        shutil.copytree(official, overlay)
        substitutions = []
        for path in overlay.rglob('*.xacro'):
            content = path.read_text()
            changed = content.replace('$(find ur_description)', str(overlay))
            if changed != content:
                substitutions.append(str(path.relative_to(overlay)))
                path.write_text(changed)
        main = overlay / 'urdf/ur.urdf.xacro'
        result = subprocess.run([xacro, str(main), 'name:=ur5e', 'ur_type:=ur5e',
                                 'tf_prefix:=ur5e_', 'safety_limits:=false'],
                                check=True, text=True, capture_output=True)
        generated = result.stdout.replace(str(main), '<pinned:ur_description/urdf/ur.urdf.xacro>')
        urdf_out.parent.mkdir(parents=True, exist_ok=True)
        urdf_out.write_text(generated)

    from urdf_parser_py.urdf import URDF as ParsedURDF
    from yourdfpy import URDF as YourURDF
    parsed = ParsedURDF.from_xml_file(str(urdf_out))
    independent = YourURDF.load(str(urdf_out), build_scene_graph=True,
                                build_collision_scene_graph=True,
                                load_meshes=False, load_collision_meshes=False)
    expected_names = ['ur5e_' + name for name in JOINTS]
    if independent.actuated_joint_names != expected_names:
        raise ValueError(f'joint order mismatch: {independent.actuated_joint_names}')
    joints = {joint.name: joint for joint in parsed.joints}
    bounds = {}
    urdf_limit_checks = {}
    for name in expected_names:
        short = name.removeprefix('ur5e_')
        if name not in joints or short not in limit_data:
            raise ValueError(f'missing joint {name} or official limit {short}')
        urdf_joint = joints[name]
        limit = limit_data[short]
        if urdf_joint.type not in ('revolute', 'continuous'):
            raise ValueError(f'active joint has unexpected type: {name} {urdf_joint.type}')
        bounds[name] = [float(limit['min_position']), float(limit['max_position'])]
        if urdf_joint.limit is None:
            raise ValueError(f'active URDF joint lacks a limit element: {name}')
        checks = {
            'lower_rad': abs(float(urdf_joint.limit.lower) - bounds[name][0]) < 1e-10,
            'upper_rad': abs(float(urdf_joint.limit.upper) - bounds[name][1]) < 1e-10,
            'velocity_rad_s': abs(float(urdf_joint.limit.velocity) - float(limit['max_velocity'])) < 1e-10,
            'effort_Nm': abs(float(urdf_joint.limit.effort) - float(limit['max_effort'])) < 1e-10,
        }
        if not all(checks.values()):
            raise ValueError(f'generated URDF limit mismatch for {name}: {checks}')
        urdf_limit_checks[name] = checks

    base, target = 'ur5e_base_link', 'ur5e_tool0'
    if base not in independent.link_map or target not in independent.link_map:
        raise ValueError(f'expected frames missing: {base}, {target}')
    rng = np.random.default_rng(seed)
    max_translation_error = 0.0
    max_rotation_error = 0.0
    for _ in range(samples):
        q = {name: float(rng.uniform(*bounds[name])) for name in expected_names}
        independent.update_cfg(q)
        ta = independent.get_transform(frame_to=target, frame_from=base)
        tb = parser_fk(parsed, base, target, q)
        max_translation_error = max(max_translation_error, float(np.linalg.norm(ta[:3, 3]-tb[:3, 3])))
        delta = ta[:3, :3].T @ tb[:3, :3]
        skew = np.array([delta[2, 1]-delta[1, 2], delta[0, 2]-delta[2, 0], delta[1, 0]-delta[0, 1]])
        angle = math.atan2(float(np.linalg.norm(skew)) / 2.0,
                           float(np.trace(delta)-1.0) / 2.0)
        max_rotation_error = max(max_rotation_error, angle)

    collision_links = [link.name for link in parsed.links if link.collision]
    collision_shapes = sum(len(link.collision if isinstance(link.collision, list) else [link.collision])
                           for link in parsed.links if link.collision)
    collision_meshes = []
    for link in parsed.links:
        collisions = link.collision if isinstance(link.collision, list) else ([link.collision] if link.collision else [])
        for collision in collisions:
            mesh = getattr(collision.geometry, 'mesh', None)
            if mesh is None:
                continue
            filename = mesh.filename
            if filename.startswith('package://ur_description/'):
                asset = official / filename.removeprefix('package://ur_description/')
            else:
                asset = Path(filename)
            if not asset.is_file():
                raise ValueError(f'collision mesh missing from pinned official asset: {filename}')
            collision_meshes.append({'link': link.name, 'uri': filename,
                                     'source_relative_path': str(asset.relative_to(official)),
                                     'sha256': sha(asset)})
    axes = {name: [float(x) for x in (joints[name].axis or [1, 0, 0])] for name in expected_names}
    official_limits_status = {name: {'position_rad': bounds[name],
                                     'velocity_rad_s': float(limit_data[name.removeprefix('ur5e_')]['max_velocity']),
                                     'has_acceleration_limits': bool(limit_data[name.removeprefix('ur5e_')].get('has_acceleration_limits', False)),
                                     'effort_Nm': float(limit_data[name.removeprefix('ur5e_')]['max_effort'])}
                              for name in expected_names}
    report = {
        'scope': 'UR5E_DESCRIPTION_AND_FK_AUDIT; NOT_TIMEOPT_READY',
        'official_source': 'Universal_Robots_ROS2_Description', 'official_revision': PIN,
        'official_source_dirty': bool(subprocess.check_output(['git','-C',str(official),'status','--porcelain'],text=True).strip()),
        'xacro_package_lookup_overlay': {'transformation': 'replace only $(find ur_description) package-index lookups by the pinned checkout path in disposable copied xacro files',
                                         'files_touched': substitutions},
        'urdf_sha256': sha(urdf_out),
        'joint_limits_sha256': sha(limits_file), 'kinematics_sha256': sha(kine_file),
        'active_joint_order': expected_names, 'joint_axes': axes,
        'supported_frames': {'base': base, 'flange': 'ur5e_flange', 'tool': target,
                              'wrist': 'ur5e_wrist_3_link'},
        'joint_limits': official_limits_status,
        'urdf_limit_crosscheck': {'all_pass': all(all(v.values()) for v in urdf_limit_checks.values()),
                                  'per_joint': urdf_limit_checks},
        'collision_geometry': {'links_with_collision': collision_links, 'shape_count': collision_shapes,
                               'mesh_assets': collision_meshes,
                               'self_collision_exclusions': 'NOT_SPECIFIED_BY_DESCRIPTION; cuRobo exclusions not built'},
        'fk_crosscheck': {'samples': samples, 'seed': seed, 'independent_parsers': ['yourdfpy', 'urdf_parser_py'],
                          'max_translation_m': max_translation_error, 'max_rotation_rad': max_rotation_error,
                          'pass': max_translation_error < 1e-9 and max_rotation_error < 1e-9},
        'timeopt_limits': {'acceleration': 'UNAVAILABLE_IN_OFFICIAL_SOURCE',
                           'jerk': 'UNAVAILABLE_IN_OFFICIAL_SOURCE',
                           'status': 'BLOCKED_NO_DEFENSIBLE_ACCELERATION_OR_JERK_LIMITS'},
        'curobo_config': 'NOT_GENERATED: no supported UR5e asset/config or defensible acceleration/jerk limits',
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--official-checkout', type=Path, required=True)
    p.add_argument('--urdf-out', type=Path, required=True)
    p.add_argument('--report-out', type=Path, required=True)
    p.add_argument('--samples', type=int, default=250)
    a = p.parse_args()
    result = generate(a.official_checkout, a.urdf_out, a.report_out, samples=a.samples)
    print(json.dumps({'fk_crosscheck': result['fk_crosscheck'], 'urdf_sha256': result['urdf_sha256'],
                      'timeopt_limits': result['timeopt_limits']}, indent=2))


if __name__ == '__main__':
    main()
