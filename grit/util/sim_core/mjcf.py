"""``merge_mjcfs`` — write a top-level MJCF that ``<include>``s several files."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from .utils import indent_xml


def merge_mjcfs(
    mjcf_model_name="scene",
    included_mjcf_files=(),
    timestep=0.002,
    integrator="implicitfast",
    noslip=None,
    memory=None,
    armature=None,
    damping=None,
    cone=None,
    exclude_body_list=None,
    output_xml_path="merged_scene.xml",
    verbose=True,
):
    """Create ``output_xml_path`` that includes every file in ``included_mjcf_files``.

    Adds ``<option timestep integrator [noslip_iterations] [cone]>``, an
    optional ``<size memory=...>``, optional joint ``<default>`` armature /
    damping and optional ``<contact><exclude>`` pairs. Include paths are
    written relative to the output file's directory. Returns the output path.
    """
    root = ET.Element("mujoco", attrib={"model": mjcf_model_name})
    out_dir = os.path.dirname(os.path.abspath(output_xml_path))
    for f in included_mjcf_files:
        rel = os.path.relpath(os.path.abspath(str(f)), out_dir)
        ET.SubElement(root, "include", attrib={"file": rel})
    if armature is not None or damping is not None:
        d = ET.SubElement(root, "default")
        attrs = {}
        if armature is not None:
            attrs["armature"] = str(armature)
        if damping is not None:
            attrs["damping"] = str(damping)
        ET.SubElement(d, "joint", attrib=attrs)
    opt = {"timestep": str(timestep), "integrator": str(integrator)}
    if noslip is not None:
        opt["noslip_iterations"] = str(int(noslip))
    if cone is not None:
        opt["cone"] = str(cone)
    ET.SubElement(root, "option", attrib=opt)
    if memory is not None:
        ET.SubElement(root, "size", attrib={"memory": memory if isinstance(memory, str) else "1G"})
    if exclude_body_list:
        c = ET.SubElement(root, "contact")
        for pair in exclude_body_list:
            if len(pair) != 2:
                raise ValueError(f"exclude pair must have length 2: {pair}")
            ET.SubElement(c, "exclude", attrib={"body1": str(pair[0]), "body2": str(pair[1])})
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_xml_path, "w", encoding="utf-8") as fh:
        fh.write(indent_xml(root))
    if verbose:
        print(f"[merge_mjcfs] merged {len(list(included_mjcf_files))} MJCF files → {output_xml_path}")
        for i, f in enumerate(included_mjcf_files):
            print(f"  - [{i}] {f}")
    return output_xml_path
