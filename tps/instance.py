import numpy
import torch
from torch.nn import Parameter
import meshzoo

from . import functions

__all__ = ['TPS']


class TPS(torch.nn.Module):
    """The thin plate spline deformation warpping.
    """

    def __init__(self, resolution, lambda_=0):
        super().__init__()

        points, faces = meshzoo.uv_sphere(num_points_per_circle=resolution,
            num_circles=resolution+2, radius=1.0)
        self.sphere_points = torch.from_numpy(points).cuda()
        self.points = Parameter(torch.from_numpy(points).cuda())
        self.lambda_ = lambda_
        self.solver = "exact"

    def forward(self, source_points):
        coefficient = functions.find_coefficients(
            self.points.unsqueeze(0), self.sphere_points.unsqueeze(0),
            self.lambda_, self.solver)
        return functions.transform(source_points,
                self.points.unsqueeze(0), coefficient)
