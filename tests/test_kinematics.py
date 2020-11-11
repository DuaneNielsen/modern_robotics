import torch
from math import sin, cos, pi, radians, degrees
from se3 import *
from matplotlib import pyplot as plt
from vpython import cylinder, rate, vector


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

    def _v(tensor):
        return vector(tensor[0], tensor[1], tensor[2])

    center = torch.tensor([
        [0, 0, 0, 1],
        [0, 0, 0.3, 0],
    ], dtype=torch.float).T

    def p(center):
        return _v(body[0:3, 0])

    def axis(center):
        return _v(body[0:3, 1])

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
        j2.pos = p(body)
        j2.axis = axis(body)

        V2 = matrix_exp_screw(S2, t)
        T3 = V1.matmul(V2).matmul(J2)
        plot.joint(T3, 'r')
        assert torch.allclose(translation(T3_dh), translation(T3), atol=1e-6)
        body = T3.matmul(center)
        j3.pos = p(body)
        j3.axis = axis(body)

        V3 = matrix_exp_screw(S3, t)
        T4 = V1.matmul(V2).matmul(V3).matmul(M)
        plot.joint(T4, 'b')
        assert torch.allclose(translation(T4_dh), translation(T4), atol=1e-6)
        body = T4.matmul(center)
        j4.pos = p(body)
        j4.axis = axis(body)

        plot.update()
        t += 0.05


def test_plot_sphere():
    s = cylinder()
    s.pos = vector(-10, 0, 0)
    s.velocity = vector(10, 5, 0)

    t = 0
    dt = 0.01

    while t < 10:

        s.pos = s.pos + s.velocity * dt
        t = t + dt
        rate(abs(1.0 / dt))
        if not -10.0 < s.pos.x < 10.0:
            s.velocity.x = - s.velocity.x
        if not 0.0 < s.pos.y < 19.0:
            s.velocity.y = - s.velocity.y