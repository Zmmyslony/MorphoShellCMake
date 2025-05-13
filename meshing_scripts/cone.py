"""
This file creates an input file of a disk with an azimuthal director, that will morph into a cone on actuation.
"""

import numpy as np
from pathlib import Path
import distmesh as dm
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from mesh_processing_module import write_VTK


def disk_mesh(radius, linear_element_count):
    dist = lambda p: dm.dcircle(p, 0, 0, radius) # Defines geometry
    coarsening = lambda p: dm.huniform(p) # Defines if certain regions have smaller elements than the rest
    element_size = 2 * radius / linear_element_count # Desired size of an element
    bounds = 1.1 * radius * np.array((-1, -1, 1, 1)) # Rectangular domain that is larger than the pattern that we want

    nodes, triangles = dm.distmesh2d(dist, coarsening, element_size, bounds) # Mesh generation
    print("Finished creating mesh.")
    print(f"Nodes: {nodes.shape[0]}")
    print(f"Triangles: {triangles.shape[0]}")
    return nodes, triangles

def rectangle_mesh(length, width, linear_element_count):
    ... """ Read distmesh documentation """

# def azimuthal_director(nodes, triangles):
#     centroids = np.mean(nodes[triangles], axis=1)
#
#     radius = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)
#     polar_angle = np.arctan2(centroids[:, 1], centroids[:, 0])
#     return 1 * polar_angle + np.pi / 2
#
# def radial_director(nodes, triangles):
#     centroids = np.mean(nodes[triangles], axis=1)
#     polar_angle = np.arctan2(centroids[:, 1], centroids[:, 0])
#     return 1 * polar_angle + 0

def topological_defect_director(nodes, triangles, charge, angular_offset=0.):
    centroids = np.mean(nodes[triangles], axis=1) # We need director at triangle, so we need triangle position, which we take to be centroids.
    polar_angle = np.arctan2(centroids[:, 1], centroids[:, 0]) # We compute triangles' polar angles
    return charge * polar_angle + angular_offset # Calculate topological defect director

def radial_director(nodes, triangles):
    return topological_defect_director(nodes, triangles, 1, 0)

def azimuthal_director(nodes, triangles):
    return topological_defect_director(nodes, triangles, 1, np.pi / 2)


def deformation_metric_info(director_angle, elongation, poisson_ratio=0.5, is_lce_mode=False):
    """ The LCE mode (setting 3 in the do_dialling.cpp) takes in [theta, lambda, poisson], else
    a_xx, a_xy, a_yy. """
    preferred_metric = np.empty([director_angle.shape[0], 3])
    if is_lce_mode:
        preferred_metric[:, 0] = director_angle
        preferred_metric[:, 1] = np.full_like(director_angle, elongation)
        preferred_metric[:, 2] = np.full_like(director_angle, poisson_ratio)
    else:
        preferred_metric[:, 0] = elongation ** 2 * np.cos(director_angle) ** 2 + elongation ** (- 2 * poisson_ratio) * np.sin(director_angle) ** 2
        preferred_metric[:, 1] = (elongation ** 2 - elongation ** (-2 * poisson_ratio)) * np.sin(director_angle) * np.cos(director_angle)
        preferred_metric[:, 2] = elongation ** 2 * np.sin(director_angle) ** 2 + elongation ** (- 2 * poisson_ratio) * np.cos(director_angle) ** 2
    return preferred_metric

def deformation_bend(director_angle, elongation):
    """ As our system is purely metric driven, there is no preferred bend in this
    particular design. """
    return np.zeros([director_angle.shape[0], 3])

def ansatz_nodes(nodes, steepness=-0.1):
    r = np.linalg.norm(nodes, axis=1)
    return np.transpose((nodes[:, 0], nodes[:, 1], r * steepness))


if __name__ == "__main__":
    target_elongation = 0.9

    nodes, triangles = disk_mesh(10, 50)
    director_angle = topological_defect_director(nodes, triangles, 0.5)

    tri_tags = np.zeros(triangles.shape[0])
    ref_shear_moduli = -np.ones_like(tri_tags) # If negative, all triangles are uniform
    ref_thicknesses = -np.ones_like(tri_tags) # If negative, all triangles are uniform

    node_tags = np.zeros(nodes.shape[0])
    constraint_indicators = np.zeros_like(node_tags) # Which nodes are clamped

    working_directory = Path(os.getcwd()).parent / "input_files"
    working_directory.mkdir(parents=True, exist_ok=True)
    write_VTK((working_directory / "topological_defect_0.5.vtk").__str__(),
              ' ',
              nodes,
              triangles,
              deformation_metric_info(director_angle, target_elongation),
              deformation_bend(director_angle, target_elongation),
              ref_thicknesses,
              ref_shear_moduli,
              tri_tags,
              constraint_indicators,
              node_tags)

    write_VTK((working_directory / "cone_ansatz.vtk").__str__(),
              'dial_factor = 1.0',
              ansatz_nodes(nodes),
              triangles,
              deformation_metric_info(director_angle, target_elongation),
              deformation_bend(director_angle, target_elongation),
              ref_thicknesses,
              ref_shear_moduli,
              tri_tags,
              constraint_indicators,
              node_tags)