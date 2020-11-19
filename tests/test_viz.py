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

    x = tensor([1., 0., 0.])
    y = tensor([0., 1., 0.])
    z = tensor([0., 0., 1.])

    def _r(x, y, z):
        return torch.stack((x, y, z), dim=0)

    z_axis_r = _r(x, y, z)
    y_axis_r = _r(x, z, y)

    joints = [tm(),
              tm(0, 0, H1),
              tm(0, W1, H1, R=y_axis_r),
              tm(L1, W1, H1, R=y_axis_r),
              tm(L1, 0, H1, R=y_axis_r),
              tm(L1 + L2, 0, H1, R=y_axis_r),
              tm(L1 + L2, W1, H1),
              M]

    viz = RobotViz(joints, screws)

    t = 0
    while t < 4 * pi:
        theta = torch.ones(8) * t
        viz.update(theta)
        t += 0.01


def test_puma():

    """ modern robotics example 4.5 """

    l1, l2 = 1.0, 1.0

    x = tensor([1., 0., 0.])
    y = tensor([0., 1., 0.])
    z = tensor([0., 0., 1.])

    def _r(x, y, z):
        return torch.stack((x, y, z), dim=0)

    joints = [tm(R=_r(x, y, z)),
              tm(R=_r(z, y, x)),
              tm(R=_r(x, z, y)),
              tm(R=_r(x, z, y)),
              tm(0, l1, 0, R=_r(z, y, x)),
              tm(0, l1+l2, 0, R=_r(x, z, y))]

    screws = tensor([
        [0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 0, 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 1., 0, 0, 0.],
        [1., 0., 0., 0, 0, -l1],
        [0., 1., 0., 0, 0., 0.]
    ]).T

    robot = RobotViz(joints, screws)

    for i in range(len(joints)):
        theta_list = torch.zeros(6)
        t = 0.0
        while t < 4 * pi:
            theta_list[i] = t
            robot.update(theta_list, fps=60)
            t += 0.01