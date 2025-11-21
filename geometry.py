import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

TREE_COORDS = [
    (0.0, 0.8), (0.125, 0.5), (0.0625, 0.5),
    (0.2, 0.25), (0.1, 0.25), (0.35, 0.0),
    (0.075, 0.0), (0.075, -0.2), (-0.075, -0.2),
    (-0.075, 0.0), (-0.35, 0.0), (-0.1, 0.25),
    (-0.2, 0.25), (-0.0625, 0.5), (-0.125, 0.5)
]

class TreePoly:
    def __init__(self, uid):
        self.uid = uid
        self.base_poly = Polygon(TREE_COORDS)
        self.vertices = np.array(TREE_COORDS) 

    def get_transformed_poly(self, x, y, rotation_degrees):
        """
        Robust method: Returns a Shapely Polygon using affine transformations.
        """
        rotated = rotate(self.base_poly, rotation_degrees, origin=(0, 0))
        final_poly = translate(rotated, x, y)
        return final_poly

    def get_transformed_coords(self, x, y, rotation_degrees):
        """
        Fast method: Returns raw coordinates using NumPy for bounds calculation.
        """
        rad = np.radians(rotation_degrees)
        c, s = np.cos(rad), np.sin(rad)
        rotation_matrix = np.array([[c, -s], [s, c]])
        rotated = self.vertices @ rotation_matrix.T
        return rotated + [x, y]