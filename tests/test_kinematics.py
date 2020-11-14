from math import sin, cos
from se3 import *
from matplotlib import pyplot as plt
from vpython import cylinder, rate, vector, box, textures, canvas, compound, color

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

    S1 = torch.tensor([0, 0, 1, 0,  0, 0], dtype=torch.float)
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
        [0, 0, 1,   L1],
        [0, 1, 0,    0],
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
        [0, 1, 0, 3*L],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float)

    screws = [
        torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, 0], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, L], dtype=torch.float),
        torch.tensor([-1, 0, 0, 0, 0, 2*L], dtype=torch.float),
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
            [0, 1, 0, 2*L],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float),
        torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 3*L],
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
                links_viz[i-3].pos = joints_viz[i-1].pos
                links_viz[i-3].axis = joints_viz[i].pos - joints_viz[i-1].pos

        rate(6)
        t+= 0.05


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

    joints = [tm(), tm(), tm(), tm(), tm(0, l1, 0), tm(0, l1+l2, 0)]

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
        t+= 0.05


def test_UR5_6R():

    """modern robotics example 4.5 """

    W1, W2, L1, L2, H1, H2 = 0.109, 0.082, 0.425, 0.392, 0.089, 0.095  # m

    screws = [
        torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1, 0, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1, 0, L1], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1, 0, L1+L2], dtype=torch.float),
        torch.tensor([0, 0, -1, -W1, L1+L2, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1+H2, 0, L1+L2], dtype=torch.float),
    ]

    M = torch.tensor([
        [-1, 0, 0, L1+L2],
        [0, 0, 1, W1+W2],
        [0, 1, 0, H1-H2],
        [0, 0, 0, 1],
    ], dtype=torch.float)

    assert torch.allclose(
        matrix_exp_screw(screws[1], -pi/2),
        torch.tensor([
            [0, 0, -1, 0.089],
            [0, 1, 0, 0],
            [1, 0, 0, 0.089],
            [0, 0, 0, 1],
        ])
    )

    assert torch.allclose(
        matrix_exp_screw(screws[4], pi/2),
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
        torch.tensor([0, 1, 0, -H1, 0, L1+L2], dtype=torch.float),
        torch.tensor([0, 0, -1, -W1, L1+L2, 0], dtype=torch.float),
        torch.tensor([0, 1, 0, -H1+H2, 0, L1+L2], dtype=torch.float),
    ]

    M = torch.tensor([
        [-1, 0, 0, L1+L2],
        [0, 0, 1, W1+W2],
        [0, 1, 0, H1-H2],
        [0, 0, 0, 1],
    ], dtype=torch.float)

    joints = [tm(), tm(0, 0, H1), tm(0, W1, H1), tm(L1, W1, H1), tm(L1, 0, H1), tm(L1+L2, 0, H1), tm(L1+L2, W1, H1), M]

    canvas(width=1200, height=1200)

    joints_viz = [cylinder(axis=vector(0, 0, 0.0), radius=0.04, color=vector(0.1, 0.3, 0.4), visible=False) for _ in range(7)]
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