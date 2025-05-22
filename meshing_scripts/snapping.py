"""
This file creates a rectangular strip with an imposed end-to-end shortening which results in buckling. By dialling
the preferred curvature, we cause it to snap, similar to
Polat, Duygu Sezen, et al. "Spontaneous snap-through of strongly buckled liquid crystalline networks." (2024).
"""

import numpy as np
from pathlib import Path
import distmesh as dm
import os
import math

import matplotlib.pyplot as plt
import matplotlib.tri as tri

from mesh_processing_module import write_VTK


def rectangle_mesh(length, width, linear_element_count, seam_positions=[], excess_nodes: int = 4):
    """ Creates a rectangular mesh, with seams at provided positions. """
    element_size = length / linear_element_count  # Desired size of an element

    length_extended = length + element_size * excess_nodes  # Increase length so we have some nodes that we can clamp.

    dist = lambda p: dm.drectangle0(p, -length_extended / 2, length_extended / 2, -width / 2,
                                    width / 2)  # Defines geometry
    coarsening = lambda p: dm.huniform(p)  # Defines if certain regions have smaller elements than the rest
    bounds = 1.1 * np.array((-length_extended / 2, -width / 2, length_extended / 2,
                             width / 2))  # Rectangular domain that is larger than the pattern that we want

    # Calculating positions of nodes, which will give us straight seams between active and passive segments
    y_fixed = np.linspace(-width / 2, width / 2, math.ceil(width / element_size) + 1, endpoint=True)
    p_fix = np.empty([0, 2])
    seam_positions.append(-length / 2)
    seam_positions.append(length / 2)
    for x in seam_positions:
        x_arr = np.full_like(y_fixed, x)
        xy_arr = np.transpose([x_arr, y_fixed])
        p_fix = np.vstack([p_fix, xy_arr])

    nodes, triangles = dm.distmesh2d(dist, coarsening, element_size, bounds, pfix=p_fix)  # Mesh generation
    print("Finished creating mesh.")
    print(f"Nodes: {nodes.shape[0]}")
    print(f"Triangles: {triangles.shape[0]}")
    return nodes, triangles


def ansatz_shape(nodes, length, compression, clamping_angle_left=0, clamping_angle_right=0):
    s = nodes[:, 0]
    y = nodes[:, 1]
    alpha = length * np.sqrt(compression) / np.pi
    x = s * (1 - np.pi ** 2 * alpha ** 2 / length ** 2) + np.pi * alpha ** 2 * np.sin(4 * np.pi * s / length) / (
            4 * length)
    z = alpha * (1 + np.cos(2 * np.pi * s / length))

    x = np.where(s < -length / 2, - length / 2 * (1 - compression) + (s + length / 2) * np.cos(clamping_angle_left), x)
    z = np.where(s < -length / 2, (s + length / 2) * np.sin(clamping_angle_left), z)

    x = np.where(s > length / 2, length / 2 * (1 - compression) + (s - length / 2) * np.cos(clamping_angle_right), x)
    z = np.where(s > length / 2, (s - length / 2) * np.sin(clamping_angle_right), z)

    return np.transpose([x, y, z])


def deformation_metric(triangles):
    preferred_metric = np.ones([triangles.shape[0], 3])
    preferred_metric[:, 1] = 0
    return preferred_metric


def deformation_bend(nodes, triangles, active_length: float, preferred_curvature_magnitude: float):
    centroids = np.mean(nodes[triangles], axis=1)
    preferred_bend = np.zeros([triangles.shape[0], 3])
    preferred_bend[:, 0] = np.where(np.abs(centroids[:, 0]) < active_length / 2, preferred_curvature_magnitude, 0)
    return preferred_bend


def is_clamped(nodes, length, eps=1e-4):
    x_ref = nodes[:, 0]
    return np.where(np.abs(x_ref) > length / 2 - eps, 1, 0)


def snapping_setup(length, width, active_ratio, compressed_length, curvature, linear_element_count=80):
    name = f"snapping_L0={length:.1f}_L={compressed_length:.1f}_w={width:.1f}_a={active_ratio:.2f}"
    nodes, triangles = rectangle_mesh(length, width, linear_element_count,
                                      seam_positions=[-length / 2 * active_ratio, length / 2 * active_ratio])
    compression = 1 - compressed_length / length
    ansatz_nodes = ansatz_shape(nodes, length, compression)

    tri_tags = np.zeros(triangles.shape[0])
    ref_shear_moduli = -np.ones_like(tri_tags)  # If negative, all triangles are uniform
    ref_thicknesses = -np.ones_like(tri_tags)  # If negative, all triangles are uniform
    node_tags = np.zeros(nodes.shape[0])
    clamp_indicators = is_clamped(nodes, length)

    working_directory = Path(os.getcwd()).parent / "input_files"
    working_directory.mkdir(parents=True, exist_ok=True)

    write_VTK((working_directory / f"{name}.vtk").__str__(),
              ' ',
              nodes,
              triangles,
              deformation_metric(triangles),
              deformation_bend(nodes, triangles, length * active_ratio, curvature),
              ref_thicknesses,
              ref_shear_moduli,
              tri_tags,
              clamp_indicators,
              node_tags)

    write_VTK((working_directory / f"{name}_ansatz.vtk").__str__(),
              'dial_factor = 0.0',
              ansatz_nodes,
              triangles,
              deformation_metric(triangles),
              deformation_bend(nodes, triangles, active_ratio * length, 0),
              ref_thicknesses,
              ref_shear_moduli,
              tri_tags,
              clamp_indicators,
              node_tags)


if __name__ == "__main__":
    snapping_setup(16, 4, 0.4, 15, -0.5)
