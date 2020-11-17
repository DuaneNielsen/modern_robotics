from math import sin, cos

from robot import *
from se3 import *
from matplotlib import pyplot as plt
from vpython import cylinder, rate, vector, box, textures, canvas, compound, color
from torch import tensor, allclose

from viz import frame, origin, x_axis, y_axis, z_axis, FrameViz


def _v(tensor):
    return vector(tensor[0], tensor[1], tensor[2])


def position(body):
    return _v(body[0:3, 0])


def axis(body):
    return _v(body[0:3, 1])


def joint_transform(theta, l):
    return torch.tensor([
        [cos(theta), -sin(theta), 0, l],
        [sin(theta), cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float)


class Plot:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        plt.ion()

    def joint(self, T, color):
        end_effector = premul(T, torch.zeros(3))
        circle = plt.Circle((end_effector[0], end_effector[1]), radius=0.1, color=color)
        self.ax.add_artist(circle)

    def update(self):
        plt.xlim(left=-5, right=5)
        plt.ylim(bottom=-5, top=5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pause(0.03)
        plt.cla()


def forward_kinematics_dh(theta1, theta2, theta3):
    T01 = joint_transform(theta1, 0)
    T12 = joint_transform(theta2, 1.0)
    T23 = joint_transform(theta3, 1.0)
    T34 = joint_transform(0, 1.0)
    J1 = T01
    J2 = T01.matmul(T12)
    J3 = T01.matmul(T12.matmul(T23))
    J4 = T01.matmul(T12.matmul(T23.matmul(T34)))
    return J1, J2, J3, J4


def tm(x=0.0, y=0.0, z=0.0, R=None):
    t = torch.tensor([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ], dtype=torch.float)
    if R is not None:
        t[0:3, 0:3] = R
    return t


def test_simple_chain_denavit_hartenberg():
    plot = Plot()

    t = 0.0

    while t < 2 * 3.147:
        theta1 = t
        theta2 = t
        theta3 = t

        J1, J2, J3, J4 = forward_kinematics_dh(theta1, theta2, theta3)

        plot.joint(J1, 'b')
        plot.joint(J2, 'm')
        plot.joint(J3, 'g')
        plot.joint(J4, 'r')

        plot.update()
        t += 0.03


def test_simple_chain():
    """
    Modern robotics fig 4.1
    """
    plot = Plot()

    L1 = 1.0
    L2 = 1.0
    L3 = 1.0

    M = torch.tensor([
        [1, 0, 0, L1 + L2 + L3],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float)

    J2 = torch.tensor([
        [1, 0, 0, L1 + L2],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float)

    J1 = torch.tensor([
        [1, 0, 0, L1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float)

    J0 = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float)

    S3_book = torch.tensor([
        [0, -1, 0, 0],
        [1, 0, 0, -(L1 + L2)],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=torch.float)

    S2_book = torch.tensor([
        [0, -1, 0, 0],
        [1, 0, 0, -L1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=torch.float)

    S1_book = torch.tensor([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=torch.float)

    def twist_matrix(twist):
        V = torch.zeros(4, 4)
        R = cross_matrix(twist[0:3])
        V[0:3, 0:3] = R
        V[0:3, 3] = twist[3:6]
        return V

    S1 = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float)
    S2 = torch.tensor([0, 0, 1, 0, -1, 0], dtype=torch.float)
    S3 = torch.tensor([0, 0, 1, 0, -2, 0], dtype=torch.float)

    assert torch.allclose(twist_matrix(S1), S1_book)
    assert torch.allclose(twist_matrix(S2), S2_book)
    assert torch.allclose(twist_matrix(S3), S3_book)

    center = torch.tensor([
        [0, 0, 0, 1],
        [0, 0, 0.3, 1],
    ], dtype=torch.float).T

    j1 = cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(0, 0, 1))
    j2 = cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(0, 1, 1))
    j3 = cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(0, 1, 0))
    j4 = cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(1, 1, 0))

    t = 0.0
    while t < 4 * 3.1417:
        T1_dh, T2_dh, T3_dh, T4_dh = forward_kinematics_dh(t, t, t)

        plot.joint(J0, 'm')

        V1 = matrix_exp_screw(S1, t)
        T2 = V1.matmul(J1)
        plot.joint(T2, 'c')
        assert torch.allclose(translation(T2_dh), translation(T2), atol=1e-6)
        body = T2.matmul(center)
        j2.pos = position(body)
        j2.axis = axis(body) - position(body)

        V2 = matrix_exp_screw(S2, t)
        T3 = V1.matmul(V2).matmul(J2)
        plot.joint(T3, 'r')
        assert torch.allclose(translation(T3_dh), translation(T3), atol=1e-6)
        body = T3.matmul(center)
        j3.pos = position(body)
        j3.axis = axis(body) - position(body)

        V3 = matrix_exp_screw(S3, t)
        T4 = V1.matmul(V2).matmul(V3).matmul(M)
        plot.joint(T4, 'b')
        assert torch.allclose(translation(T4_dh), translation(T4), atol=1e-6)
        body = T4.matmul(center)
        j4.pos = position(body)
        j4.axis = axis(body) - position(body)

        plot.update()
        t += 0.05


def test_3r_open_chain():
    canvas(width=1200, height=600)

    L1 = 5.0
    L2 = 2.0

    M = torch.tensor([
        [0, 0, 1, L1],
        [0, 1, 0, 0],
        [-1, 0, 0, -L2],
        [0, 0, 0, 1]
    ])

    J1 = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float)

    J2 = torch.tensor([
        [0, 0, 0, L1],
        [0, 0, -1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float)

    J3 = torch.tensor([
        [0, 0, 1, L1],
        [0, 1, 0, 0],
        [-1, 0, 0, -L2],
        [0, 0, 0, 1]
    ], dtype=torch.float)

    S1 = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float)
    S2 = torch.tensor([0, -1, 0, 0, 0, -L1], dtype=torch.float)
    S3 = torch.tensor([1, 0, 0, 0, -L2, 0], dtype=torch.float)

    body = torch.tensor([
        [0, 0, 0, 1],
        [0, 0, 0.3, 1],
    ], dtype=torch.float).T

    end_effector_position = torch.tensor([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ], dtype=torch.float).T

    ground = box(color=vector(0, 0, 1))
    ground.pos = vector(0, 0, -3)
    ground.size = vector(1, 1, 0.05)

    stone = {'file': textures.stones, 'place': 'right'}

    j1 = cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(0, 1.0, 0.3))
    j2 = cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(0, 1.0, 0.3))
    j3 = cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(0, 1.0, 0.3))
    end_effector = cylinder(axis=vector(0, 0, 0.2), radius=0.1, color=vector(1.0, 0.3, 0.3))

    l1 = cylinder(pos=ground.pos, axis=j1.pos - ground.pos, radius=0.1, color=vector(0.6, 0.6, 0.6))
    l2 = cylinder(pos=ground.pos, axis=j1.pos - ground.pos, radius=0.1, color=vector(0.6, 0.6, 0.6))
    l3 = cylinder(pos=ground.pos, axis=j1.pos - ground.pos, radius=0.1, color=vector(0.6, 0.6, 0.6))

    t = 0

    while t < 10 * pi:
        s1 = matrix_exp_screw(S1, t)

        p = s1.matmul(J1).matmul(body)
        j1.pos = position(p)
        j1.axis = axis(p) - position(p)

        l1.pos = ground.pos
        l1.axis = j1.pos - ground.pos

        s2 = matrix_exp_screw(S2, t)
        p = s1.matmul(s2).matmul(J2).matmul(body)
        j2.pos = position(p)
        j2.axis = axis(p) - position(p)

        l2.pos = j1.pos
        l2.axis = j2.pos - j1.pos

        s3 = matrix_exp_screw(S3, t)
        p = s1.matmul(s2).matmul(s3).matmul(J3).matmul(body)
        j3.pos = position(p)
        j3.axis = axis(p) - position(p)

        l3.pos = j2.pos
        l3.axis = j3.pos - j2.pos

        p = s1.matmul(s2).matmul(s3).matmul(J3).matmul(end_effector_position)
        end_effector.pos = position(p)
        end_effector.axis = axis(p) - position(p)

        rate(24)
        t += 0.05


def test_6r_kinemetic_chain():
    """modern robotics fig 4.4"""

    L = 1.0

    M = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 3 * L],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float)

    screws = [
        torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, L], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, 2 * L], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float)
    ]

    joints = [
        torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float),
        torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float),
        torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float),
        torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, L],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float),
        torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 2 * L],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float),
        torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 3 * L],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float),
    ]

    body = torch.tensor([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=torch.float).T

    canvas(width=1200, height=600)
    joints_viz = []
    links_viz = []
    for _ in range(6):
        joints_viz += [cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(0, 1.0, 0.3))]

    joints_viz[5].color = vector(1, 0, 0)

    for _ in range(3):
        links_viz += [cylinder(radius=0.1, color=vector(0.6, 0.6, 0.6))]

    t = 0.0

    while t < 6 * pi:
        transforms = []
        for screw in screws:
            transforms += [matrix_exp_screw(screw, t)]

        T = torch.eye(4)

        for i in range(6):
            T = T.matmul(transforms[i])
            p = T.matmul(joints[i]).matmul(body)
            joints_viz[i].pos = position(p)
            joints_viz[i].axis = (axis(p) - position(p)) * 0.3

            if i > 2:
                links_viz[i - 3].pos = joints_viz[i - 1].pos
                links_viz[i - 3].axis = joints_viz[i].pos - joints_viz[i - 1].pos

        rate(6)
        t += 0.05


def test_rrprrr_open_chain():
    """ modern robotics example 4.4"""

    l1, l2 = 1.0, 1.0

    screws = [
        torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float),
        torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([1, 0, 0, 0, 0, -l1], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float),
    ]

    joints = [tm(), tm(), tm(), tm(), tm(0, l1, 0), tm(0, l1 + l2, 0)]

    canvas(width=1200, height=600)
    joints_viz = [cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(0, 1.0, 0.3)) for _ in range(6)]
    joints_facing = [z_axis, x_axis, y_axis, y_axis, x_axis, y_axis]

    links_viz = [None, None]
    links_viz += [cylinder(radius=0.1, color=vector(0.6, 0.6, 0.6))]
    links_viz += [None]
    links_viz += [cylinder(radius=0.1, color=vector(0.6, 0.6, 0.6)) for _ in range(2)]

    handle = cylinder(size=vector(1, .2, .2), color=vector(0.72, 0.42, 0))

    head = box(size=vector(.2, .6, .2), pos=vector(1.1, 0, 0), color=color.gray(.6))

    hammer = compound([handle, head])
    hammer.axis = vector(1, 1, 0)

    t = 0.0

    def id(t):
        return t

    while t < 6 * pi:

        fs = [id, id, abs, id, id, id]
        ts = [matrix_exp_screw(s, f(2 * sin(t * 0.5 * pi))) for s, f in zip(screws, fs)]

        tb = tm()
        prev_joint_pos = vector(0, 0, 0)

        for tf, j, jv, jf, lv in zip(ts, joints, joints_viz, joints_facing, links_viz):
            tb = tb.matmul(tf)
            joint_frame = tb.matmul(j).matmul(frame)
            jv.pos = origin(joint_frame)
            jv.axis = (jf(joint_frame) - origin(joint_frame)) * 0.2
            if lv:
                lv.pos = prev_joint_pos
                lv.axis = jv.pos - prev_joint_pos
            prev_joint_pos = origin(joint_frame)

        end_effector_frame = tb.matmul(joints[-1]).matmul(frame)
        hammer.pos = origin(end_effector_frame)
        hammer.axis = (y_axis(end_effector_frame) - origin(end_effector_frame)) * 0.2

        rate(3)
        t += 0.05


def test_UR5_6R():
    """modern robotics example 4.5 """

    W1, W2, L1, L2, H1, H2 = 0.109, 0.082, 0.425, 0.392, 0.089, 0.095  # m

    screws = [
        torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1, 0, L1], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1, 0, L1 + L2], dtype=torch.float),
        torch.tensor([0, 0, -1, -W1, L1 + L2, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1 + H2, 0, L1 + L2], dtype=torch.float),
    ]

    M = torch.tensor([
        [-1, 0, 0, L1 + L2],
        [0, 0, 1, W1 + W2],
        [0, 1, 0, H1 - H2],
        [0, 0, 0, 1],
    ], dtype=torch.float)

    assert torch.allclose(
        matrix_exp_screw(screws[1], -pi / 2),
        torch.tensor([
            [0, 0, -1, 0.089],
            [0, 1, 0, 0],
            [1, 0, 0, 0.089],
            [0, 0, 0, 1],
        ])
    )

    assert torch.allclose(
        matrix_exp_screw(screws[4], pi / 2),
        torch.tensor([
            [0, 1, 0, 0.708],
            [-1, 0, 0, 0.926],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
    )

    assert torch.allclose(
        matrix_exp_screw(screws[1], -pi / 2).matmul(matrix_exp_screw(screws[4], pi / 2)).matmul(M),
        torch.tensor([
            [0, -1, 0, 0.095],
            [1, 0, 0, 0.109],
            [0, 0, 1, 0.988],
            [0, 0, 0, 1],
        ])
    )


def test_UR5_6R_visual():
    """modern robotics example 4.5 """

    W1, W2, L1, L2, H1, H2 = 0.109, 0.082, 0.425, 0.392, 0.089, 0.095  # m

    screws = [
        torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float),
        torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1, 0, L1], dtype=torch.float),
        torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1, 0, L1 + L2], dtype=torch.float),
        torch.tensor([0, 0, -1, -W1, L1 + L2, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1 + H2, 0, L1 + L2], dtype=torch.float),
    ]

    M = torch.tensor([
        [-1, 0, 0, L1 + L2],
        [0, 0, 1, W1 + W2],
        [0, 1, 0, H1 - H2],
        [0, 0, 0, 1],
    ], dtype=torch.float)

    joints = [tm(), tm(0, 0, H1), tm(0, W1, H1), tm(L1, W1, H1), tm(L1, 0, H1), tm(L1 + L2, 0, H1), tm(L1 + L2, W1, H1),
              M]

    canvas(width=1200, height=1200)

    joints_viz = [cylinder(axis=vector(0, 0, 0.0), radius=0.04, color=vector(0.1, 0.3, 0.4), visible=False) for _ in
                  range(7)]
    joints_facing = [z_axis, z_axis, y_axis, y_axis, y_axis, y_axis, z_axis, y_axis]

    links_viz = [None]
    links_viz += [cylinder(radius=0.03, color=vector(0.6, 0.6, 0.6), visible=False) for _ in range(5)]
    links_viz += [None, None]

    end_effector_viz = FrameViz()

    def normalize_half(radians):
        return ((radians + pi) % (2 * pi) - pi)

    drivers = [normalize_half for _ in screws]

    t = 0.0

    while t < 10 * pi:

        ts = [matrix_exp_screw(s, driver(t)) for s, driver in zip(screws, drivers)]

        tb = tm()
        prev_joint_pos = vector(0, 0, 0)
        end_frame = None

        for tf, j, jv, jf, lv in list(zip(ts, joints, joints_viz, joints_facing, links_viz)):
            tb = tb.matmul(tf)
            joint_frame = tb.matmul(j).matmul(frame)
            jv.visible = True
            jv.pos = origin(joint_frame)
            jv.axis = (jf(joint_frame) - origin(joint_frame)) * 0.05
            if lv:
                lv.visible = True
                lv.pos = prev_joint_pos
                lv.axis = jv.pos - prev_joint_pos
            prev_joint_pos = origin(joint_frame)
            end_frame = joint_frame

        end_effector_viz.update(end_frame)

        rate(12)
        t += 0.01


def test_rrrrrr_body_form():
    """ modern robotics example 4.6 PoE in body form"""

    L = 1.0

    screws_body = [
        torch.tensor([0, 0, 1, -3 * L, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, -3 * L], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, -2 * L], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, -L], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float)
    ]

    screws_body = torch.stack(screws_body, dim=1)

    screws_spacial = [
        torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, L], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, 2 * L], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float)
    ]

    screws_spacial = torch.stack(screws_spacial, dim=1)

    M = tm(0, 3 * L, 0)
    inv_adjoint_M = inv_adjoint(M)

    screws_body_converted = inv_adjoint_M.matmul(screws_spacial)
    assert allclose(screws_body, screws_body_converted)

    joints = [tm(0, 0, 0), tm(0, 0, 0), tm(0, 0, 0), tm(0, L, 0), tm(0, 2 * L, 0), M]

    def cyl():
        return cylinder(axis=vector(0, 0, 0.0), radius=0.2, color=vector(0.1, 0.3, 0.4), visible=False)

    canvas(width=1200, height=1200)

    # joints_viz = [cyl() for _ in range(6)]
    joints_facing = [z_axis, y_axis, x_axis, x_axis, x_axis, y_axis]

    def link():
        return cylinder(radius=0.1, color=vector(0.6, 0.6, 0.6), visible=False)

    # links = [None, None, None, link(), link(), link()]

    end_effector_body_viz = FrameViz()
    end_effector_spacial_viz = FrameViz()

    def id(t):
        return t

    t = 0.0

    def body_exp_screw(B, S, thetadot):
        S_M = twist_matrix(S)
        S_M = B.matmul(S_M)
        S_M = twist_vector(S_M)
        return matrix_exp_screw(S_M, thetadot)

    while t < 6 * pi:

        end_effector = M
        for s in screws_body.T:
            # end_effector = end_effector.matmul(body_exp_screw(B, s, t))
            end_effector = end_effector.matmul(matrix_exp_screw(s, t))

        end_effector_body_frame = end_effector.matmul(frame)
        end_effector_body_viz.update(end_effector_body_frame)

        T = tm()

        for s in screws_spacial.T:
            T = T.matmul(matrix_exp_screw(s, t))
        end_effector_spacial_frame = T.matmul(M).matmul(frame)
        end_effector_spacial_viz.update(end_effector_spacial_frame)

        assert torch.allclose(end_effector_spacial_frame, end_effector_body_frame, atol=1e-5)

        rate(12)
        t += 0.01


def test_velocity_kinematics():
    """modern robotics chapter 5.0"""

    L1, L2 = 1.0, 1.0

    def j(theta):
        return torch.tensor([
            [-L1 * sin(theta[0]) - L2 * sin(theta[0] + theta[1]), -L2 * sin(theta[0] + theta[1])],
            [L1 * cos(theta[0]) + L2 * cos(theta[0] + theta[1]), L2 * cos(theta[0] + theta[1])],
        ])

    assert torch.allclose(
        j(torch.tensor([0, pi / 4])),
        torch.tensor([
            [-0.71, -0.71],
            [1.71, 0.71]
        ]),
        atol=1e-1)

    assert torch.allclose(
        j(torch.tensor([0, 3 * pi / 4])),
        torch.tensor([
            [-0.71, -0.71],
            [0.29, -0.71]
        ]),
        atol=1e-1)

    r = torch.linspace(0, 2 * pi, 20)
    x = torch.cos(r)
    y = torch.sin(r)
    s = torch.stack((x, y))

    m = j(torch.tensor([0, pi / 4])).matmul(s)

    plt.plot(m[0], m[1])
    plt.title('manipulability plot theta0 = 0, theta1 = 1')
    plt.xlabel('thetadot 0')
    plt.ylabel('thetadot 1')
    plt.show()


def test_space_jacobian_RRRP():
    """ modern robotics example 5.2"""

    l1, l2 = 1.0, 1.0

    theta = torch.tensor([0, 0, 0, 0])

    def j(theta):
        return torch.tensor([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, l1 * sin(theta[0]), -l1 * cos(theta[0]), 0],
            [0, 0, 1, l1 * sin(theta[0]) + l2 * sin(theta[1]), -l1 * cos(theta[0]) - l2 * cos(theta[1]), 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=torch.float).T

    assert torch.allclose(
        j(theta),
        torch.tensor([
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [1., 1., 1., 0.],
            [0., 0., 0., 0.],
            [0., -1., -2., 0.],
            [0., 0., 0., 1.]
        ])
    )

    theta = torch.tensor([0, pi / 2, 0, 0], dtype=torch.float)

    assert torch.allclose(
        j(theta),
        torch.tensor([
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [1., 1., 1., 0.],
            [0., 0., 1., 0.],
            [0., -1., -1., 0.],
            [0., 0., 0., 1.]
        ], dtype=torch.float)
    )


def test_space_jacobian_RRPRRR():
    """ modern robotics example 5.3"""

    l1, l2 = 1.0, 1.0

    screws = tensor([
        [0., 0., 1., 0., 0., 0.],
        [-1., 0., 0., 0., -l1, 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 1., l2, 0., 0.],
        [-1., 0., 0., 0, -l1, l2],
        [0., 1., 0., -l1, 0., 0.]
    ]).T

    def jacobian_analytical(theta):
        s = torch.sin(theta)
        c = torch.cos(theta)

        ws1 = tensor([0., 0., 1.])
        q1 = tensor([0., 0., l1])
        v1 = -ws1.cross(q1)
        assert allclose(v1, tensor([0., 0., 0.]))

        ws2 = tensor([-cos(theta[0]), -sin(theta[0]), 0.])
        q2 = tensor([0., 0., l1])
        v2 = -ws2.cross(q2)
        assert allclose(v2, tensor([l1 * s[0], -l1 * c[0], 0.]))

        ws3 = tensor([0., 0., 0.])
        z_ax = tensor([0., 0., 1.])
        x_ax = tensor([1., 0., 0.])
        y_ax = tensor([0., 1., 0.])
        r3 = matrix_exp_rotation(z_ax, theta[0]).matmul(matrix_exp_rotation(x_ax, -theta[1]))
        v3 = r3.matmul(y_ax.T)
        v3_check = tensor([-s[0] * c[1], c[0] * c[1], -s[1]])

        assert allclose(v3, v3_check)

        qw = tensor([0, 0, l1]) + r3.matmul(tensor([0., l2 + theta[2], 0.]))

        assert allclose(qw, tensor([
            -(l2 + theta[2]) * s[0] * c[1],
            (l2 + theta[2]) * c[0] * c[1],
            l1 - (l2 + theta[2]) * s[1]
        ]))

        ws4 = r3.matmul(z_ax)

        assert allclose(ws4, tensor([
            -s[0] * s[1],
            c[0] * s[1],
            c[1]
        ]))

        r5 = r3.matmul(matrix_exp_rotation(z_ax, theta[3]))
        ws5 = r5.matmul(-x_ax)

        assert allclose(ws5, tensor([
            -c[0] * c[3] + s[0] * c[1] * s[3],
            -s[0] * c[3] - c[0] * c[1] * s[3],
            s[1] * s[3]
        ]))

        r6 = r5.matmul(matrix_exp_rotation(x_ax, -theta[4]))
        ws6 = r6.matmul(y_ax)

        assert allclose(ws6, torch.tensor([
            -c[4] * (s[0] * c[1] * c[3] + c[0] * s[3]) + s[0] * s[1] * s[4],
            c[4] * (c[0] * c[1] * c[3] - s[0] * s[3]) - c[0] * s[1] * s[4],
            -s[1] * c[3] * c[4] - c[1] * s[4]
        ]))

        j = torch.zeros(6, 6, dtype=torch.float)

        j[0:3, 0] = ws1

        j[0:3, 1] = ws2
        j[3:6, 1] = -ws2.cross(q2)

        j[3:6, 2] = v3

        j[0:3, 3] = ws4
        j[3:6, 3] = -ws4.cross(qw)

        j[0:3, 4] = ws5
        j[3:6, 4] = -ws5.cross(qw)

        j[0:3, 5] = ws6
        j[3:6, 5] = -ws6.cross(qw)
        return j

    theta = tensor([0., 0., 0., 0., 0., 0.])
    torch.set_printoptions(sci_mode=False, profile='short')
    # print('')
    # print(jacobian_space(screws, theta))
    # print(jacobian_analytical(theta))
    # print('')
    assert allclose(jacobian_space(screws, theta), jacobian_space(screws, theta))
    assert allclose(jacobian_analytical(theta), jacobian_space(screws, theta))

    theta = tensor([1., 1., 1., 1., 1., 1.])
    assert allclose(jacobian_analytical(theta), jacobian_space(screws, theta), atol=1e7)

    for l in torch.linspace(0, 4, 12):
        theta = tensor([1., 1., 1., 1., 1., 1.]) * pi * l
        assert allclose(jacobian_analytical(theta), jacobian_space(screws, theta), atol=1e7)


def test_jacobian_body_form():
    """ modern robotics 5.3
    with an extra joint RRPRRRR
    testing body jacobian by computing space jacobian, applying some joint angles
    and changing the frame
    """

    l1, l2 = 1.0, 1.0

    screws_s = tensor([
        [0., 0., 1., 0., 0., 0.],
        [-1., 0., 0., 0., -l1, 0.],
        [0., 0., 0., 0., 1., 0.],
        [0., 0., 1., l2, 0., 0.],
        [-1., 0., 0., 0, -l1, l2],
        [0., 1., 0., -l1, 0., 0.],
        [0., 1., 0., -l1, 0., 0.],
    ]).T

    M = tm(0, l2, l1)

    screws_b = inv_adjoint(M).matmul(screws_s)

    thetas = torch.ones(7)

    _, J = screws_b.shape

    assert J == 7

    body_j = jacobian_body(screws_b, thetas)
    space_j = jacobian_space(screws_s, thetas)

    thetadots = torch.ones(7)

    E = torch.eye(4)
    for i in range(J):
        E = E.matmul(matrix_exp_screw(screws_s[:, i], thetas[i]))
    E = E.matmul(M)

    Vb = body_j.matmul(thetadots)
    Vs = space_j.matmul(thetadots)

    assert allclose(Vs, adjoint(E).matmul(Vb))
    assert allclose(Vb, inv_adjoint(E).matmul(Vs))


def test_fkin_space():

    """ modern robotics example 4.5 """

    W1, W2, L1, L2, H1, H2 = 0.109, 0.082, 0.425, 0.392, 0.089, 0.095  # m

    screws = tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, -H1, 0, 0],
        [0, 1, 0, -H1, 0, L1],
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

    theta_list = torch.zeros(6)
    E = fkin_space(M, screws, theta_list)
    assert allclose(E, M)

    theta_list = tensor([0, -pi / 2, 0, 0, 0, 0])

    E = fkin_space(torch.eye(4), screws, theta_list)
    expected = tensor([
        [0, 0, -1, 0.089],
        [0, 1, 0, 0],
        [1, 0, 0, 0.089],
        [0, 0, 0, 1],
    ])

    assert allclose(E, expected)

    theta_list = tensor([0, 0, 0, 0, pi/2, 0])
    E = fkin_space(torch.eye(4), screws, theta_list)
    expected = tensor([
        [0, 1, 0, 0.708],
        [-1, 0, 0, 0.926],
        [0, 0, 1, 0.],
        [0, 0, 0, 1],
    ])

    assert allclose(E, expected)

    theta_list = tensor([0, -pi/2, 0, 0, pi/2, 0])
    E = fkin_space(M, screws, theta_list)
    expected = tensor([
        [0, -1, 0, 0.095],
        [1, 0, 0, 0.109],
        [0, 0, 1, 0.988],
        [0, 0, 0, 1],
    ])

    assert allclose(E, expected)


def test_fkin_body():

    """ modern robotics example 4.7 """

    L1, L2, L3 = 0.550, 0.300, 0.060
    W1 = 0.045

    M = tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, L1 + L2 + L3],
        [0, 0, 0, 1],
    ])

    screws_body = tensor([
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, L1+L2+L3, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, L2+L3, 0, W1],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, L3, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ], dtype=torch.float).T

    theta_list = tensor([0, pi/4, 0, -pi/4, 0, -pi/2, 0])

    E = fkin_body(M, screws_body, theta_list)

    expected = tensor([
        [0, 0, -1, 0.3157],
        [0, 1, 0, 0],
        [1, 0, 0, 0.6571],
        [0, 0, 0, 1]
    ])

    assert allclose(E, expected, atol=1e-4)


def test_space_to_body():

    """ modern robotics Example 4.6 """

    L = 1.0

    screws_body = [
        torch.tensor([0, 0, 1, -3 * L, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, -3 * L], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, -2 * L], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, -L], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float)
    ]

    screws_body = torch.stack(screws_body, dim=1)

    screws_space = [
        torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, L], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, 2 * L], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float)
    ]

    screws_space = torch.stack(screws_space, dim=1)

    M = tm(0, 3 * L, 0)

    assert allclose(space_to_body(M, screws_space), screws_body)
    assert allclose(body_to_space(M, screws_body), screws_space)