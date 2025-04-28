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
    dist = lambda p: dm.dcircle(p, 0, 0, radius)
    coarsening = lambda p: dm.huniform(p)
    element_size = 2 * radius / linear_element_count
    bounds = 1.1 * radius * np.array((-1, -1, 1, 1))

    nodes, triangles = dm.distmesh2d(dist, coarsening, element_size, bounds)
    print("Finished creating mesh.")
    print(f"Nodes: {nodes.shape[0]}")
    print(f"Triangles: {triangles.shape[0]}")
    return nodes, triangles


def azimuthal_director(nodes, triangles):
    centroids = np.mean(nodes[triangles], axis=1)
    return np.mod(np.arctan2(centroids[:, 1], centroids[:, 0]) + np.pi / 2, np.pi)


def deformation_metric_info(director_angle, elongation, poisson_ratio=0.5):
    """ The LCE mode (setting 3 in the do_dialling.cpp) takes in
     [theta, lambda, poisson]"""
    preferred_metric = np.empty([director_angle.shape[0], 3])
    preferred_metric[:, 0] = director_angle
    preferred_metric[:, 1] = np.full_like(director_angle, elongation)
    preferred_metric[:, 2] = np.full_like(director_angle, poisson_ratio)
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
    working_directory = Path(os.getcwd()) / "output"
    working_directory.mkdir(parents=True, exist_ok=True)

    director_angle = azimuthal_director(nodes, triangles)

    tri_tags = np.zeros(triangles.shape[0])
    ref_shear_moduli = -np.ones_like(tri_tags)
    ref_thicknesses = -np.ones_like(tri_tags)

    node_tags = np.zeros(nodes.shape[0])
    constraint_indicators = np.zeros_like(node_tags)

    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    # plt.tripcolor(triangulation, director_angle)
    # plt.colorbar()
    # plt.show()

    write_VTK((working_directory / "cone_ref.vtk").__str__(),
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