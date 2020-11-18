from viz import tm, RobotViz, x_axis, y_axis, z_axis
import torch
from torch import tensor
from math import pi


def test_robot_viz():
    """modern robotics example 4.5 """

    W1, W2, L1, L2, H1, H2 = 0.109, 0.082, 0.425, 0.392, 0.089, 0.095  # m

    screws = tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, -H1, 0, 0],
        [0, 1, 0, -H1, 0, L1],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, -H1, 0, L1 + L2],
        [0, 0, -1, -W1, L1 + L2, 0],
        [0, 1, 0, -H1 + H2, 0, L1 + L2]
    ]).T

    M = torch.tensor([
        [-1, 0, 0, L1 + L2],
        [0, 0, 1, W1 + W2],
        [0, 1, 0, H1 - H2],
        [0, 0, 0, 1],
    ], dtype=torch.float)

    joints = [tm(), tm(0, 0, H1), tm(0, W1, H1), tm(L1, W1, H1),
              tm(L1, 0, H1), tm(L1 + L2, 0, H1), tm(L1 + L2, W1, H1), M]

    joints_facing = [z_axis, z_axis, y_axis, y_axis, y_axis, y_axis, z_axis, y_axis]

    viz = RobotViz(joints, screws, joints_facing)

    t = 0
    while t < 4 * pi:
        theta = torch.ones(8) * t
        viz.update(theta)
        t += 0.01
