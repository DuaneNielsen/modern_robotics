from viz import tm, RobotViz, x_axis, y_axis, z_axis, FrameViz, origin, _v
import torch
from torch import tensor
from math import pi
from vpython import ellipsoid
from robot import jacobian_space, manipualibility_ellipsoid, singular


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
    end_effector_viz = FrameViz(thickness=0.02)

    t = 0
    while t < 4 * pi:
        theta = torch.ones(8) * t
        E = viz.update(theta)
        end_effector_viz.update(E)
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
    end_effector_viz = FrameViz(thickness=0.02)

    for i in range(len(joints)):
        theta_list = torch.zeros(6)
        t = 0.0
        while t < 4 * pi:
            theta_list[i] = t
            E = robot.update(theta_list, fps=60)
            end_effector_viz.update(E)
            t += 0.01


def test_puma_manip():

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
    #end_effector_viz = FrameViz(thickness=0.02)
    man = ellipsoid()

    # for i in range(len(joints)):
    theta_list = torch.randn(6)
    t = 0.0
    while t < 4 * pi:
        js = jacobian_space(screws, theta_list * t)
        angular_js = js[0:3]
        vel_js = js[3:6]
        if not singular(js):
            A, lam, v = manipualibility_ellipsoid(js)
            lengths = lam.sqrt()
            E = robot.update(theta_list, fps=60)
            #end_effector_viz.update(E)

            man.pos = origin(E)
            man.axis = _v(v[:, 0])
            man.length = lengths[0, 0].item()
            man.height = lengths[1, 0].item()
            man.width = lengths[2, 0].item()
        else:
            print('singular')

        t += 0.01


def test_kuka_manip():

    L1, L2, L3, L4 = 0.34, 0.4, 0.4, 0.15

    x = tensor([1., 0., 0.])
    y = tensor([0., 1., 0.])
    z = tensor([0., 0., 1.])

    def _r(x, y, z):
        return torch.stack((x, y, z), dim=0)

    joint_home_list = [
        tm(z=L1/2, R=_r(x, y, z)),
        tm(z=L1, R=_r(z, y, x)),
        tm(z=L1 + (L2/2), R=_r(x, y, z)),
        tm(z=L1 + L2, R=_r(z, y, x)),
        tm(z=L1 + L2 + (L3/2), R=_r(x, y, z)),
        tm(z=L1 + L2 + L3, R=_r(z, y, x)),
        tm(z=L1 + L2 + L3 + L4, R=_r(x, y, z)),
    ]

    s_list = tensor([
        [0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., L1, 0.],
        [0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., L1+L2, 0.],
        [0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., L1+L2+L3, 0.],
        [0., 0., 1., 0., 0., 0.],
    ]).T

    theta_list = torch.linspace(1, 7, 7) * pi / 16
    t = 0.0

    robot = RobotViz(joint_home_list, s_list)
    end_effector = FrameViz()
    man = ellipsoid()

    while t < 4 * pi:

        robot.update(theta_list)

        js = jacobian_space(s_list, theta_list)

        man.opacity = 0.2

        A, lam, v = manipualibility_ellipsoid(js)
        lengths = lam.sqrt()
        E = robot.update(theta_list, fps=12)
        end_effector.update(E)

        #if not singular(js):

        man.pos = origin(E)
        man.axis = _v(v[:, 0])
        man.length = lengths[0, 0].item() * 0.3
        man.height = lengths[1, 0].item() * 0.3
        man.width = lengths[2, 0].item() * 0.3

        t += 0.01
        theta_list += 0.01