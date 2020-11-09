from se3 import *
from math import sin, cos, radians, pi, degrees
from matplotlib import pyplot as plt
import pytest


I = torch.eye(4, 4)

x_axis = torch.tensor((
    [0, 1],
    [0, 0],
    [0, 0],
    [1, 1]
), dtype=torch.float)

y_axis = torch.tensor((
    [0, 0],
    [0, 1],
    [0, 0],
    [1, 1]
), dtype=torch.float)


def test_transform():
    Tsb = torch.tensor([
        [-1, 0, 0, 4],
        [0, 1, 0, 0.4],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float)

    Tbs = torch.tensor([
        [-1, 0, 0, 4],
        [0, 1, 0, -0.4],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float)

    Tsr = torch.tensor([
        [1, 0, 0, 2],
        [0, 1, 0, -1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=torch.float)

    ws = vector(0, 0, 2)
    rs = vector(2, -1, 0)

    wb = vector(0, 0, -2)
    rb = vector(2, -1.4, 0)

    wr = vector(0, 0, 2)
    rr = vector(0, 0, 0)

    vs = ws.cross(-rs)
    vb = wb.cross(-rb)
    vr = wr.cross(-rr)

    Vs = torch.cat((ws, vs))
    Vb = torch.cat((wb, vb))
    Vr = torch.cat((wr, vr))

    rs_hat = premul(Tsb, rb)
    assert torch.allclose(rs_hat, rs)

    origin_b_in_frame_s = premul(Tsb, vector(0, 0, 0))
    assert torch.allclose(origin_b_in_frame_s, vector(4.0, 0.4, 0))

    origin_r_in_frame_s = premul(Tsr, vector(0, 0, 0))
    assert torch.allclose(origin_r_in_frame_s, vector(2, -1, 0))

    _Tsr = transform(torch.eye(3), vector(2, -1, 0))
    assert torch.allclose(_Tsr, Tsr)

    Trs = inv_transform(torch.eye(3), vector(2, -1, 0))
    assert torch.allclose(_Tsr.matmul(Trs), I)
    assert torch.allclose(Trs.matmul(Tsr), I)

    vecA_in_b_frame = vector(3, 3, 0)
    vecA_in_s_frame = vector(1, 3.4, 0)

    assert torch.allclose(Tsb.matmul(Tbs), I)
    assert torch.allclose(premul(Tsb, vecA_in_b_frame), vecA_in_s_frame)
    assert torch.allclose(premul(Tbs, vecA_in_s_frame), vecA_in_b_frame)


def test_construct_extract():
    R = torch.rand(3, 3)
    p = torch.rand(3)
    T = transform(R, p)
    assert torch.allclose(rotation(T), R)
    assert torch.allclose(translation(T), p)


def test_rotation_log():

    omega = torch.rand(3)
    R = matrix_exp_rotation(omega, torch.ones(1) * 2.0)
    w, theta = matrix_log_rotation(R)

    def z_rotation(degrees):
        return torch.tensor([
            [cos(radians(degrees)), -sin(radians(degrees)), 0],
            [sin(radians(degrees)), cos(radians(degrees)), 0],
            [0, 0, 1],
        ])

    Rsb = z_rotation(30)
    Rsc = z_rotation(60)
    R = torch.matmul(Rsc, Rsb.inverse())
    w, theta = matrix_log_rotation(R)
    w = uncross_matrix(w)
    assert torch.allclose(w, torch.tensor([0, 0, 1.0]))
    R_exp = matrix_exp_rotation(w, theta)
    assert torch.allclose(R, R_exp)
    assert -0.0001 < theta - (radians(60) - radians(30)) < 0.0001

    def plot_angle(from_angle, to_angle, theta, expected):
        plt.ion()
        plt.plot([0, cos(radians(from_angle))], [0, sin(radians(from_angle))], label=f'from_angle {from_angle}')
        plt.plot([0, cos(radians(to_angle))], [0, sin(radians(to_angle))], label=f'to_angle {to_angle}')
        plt.plot([0, cos(theta)], [0, sin(theta)], label=f'theta {degrees(theta)}')
        plt.plot([0, cos(expected)], [0, sin(expected)], label=f'expected {degrees(expected)}')
        plt.legend()
        plt.xlim(left=-1, right=1)
        plt.ylim(bottom=-1, top=1)
        plt.pause(3)
        plt.cla()

    def normalize_angle(radians):
        return (radians + pi) % (2 * pi) - pi

    def difference(from_degrees, to_degrees, omega_hat, atol):
        Rsb = z_rotation(from_degrees)
        Rsc = z_rotation(to_degrees)
        R = torch.matmul(Rsc, Rsb.inverse())
        w, theta = matrix_log_rotation(R)
        w = uncross_matrix(w)
        expected = normalize_angle(radians(to_degrees) - radians(from_degrees))
        plot_angle(from_degrees, to_degrees, normalize_angle(w[2] * theta), expected)
        assert torch.allclose(w, omega_hat)
        assert -0.0001 < normalize_angle(w[2] * theta) - expected < 0.0001
        R_exp = matrix_exp_rotation(w, theta)
        assert torch.allclose(R, R_exp, atol=atol)


    difference(35, 65, omega_hat=torch.tensor([0, 0, 1.0]), atol=1e-8)
    difference(65, 35, omega_hat=torch.tensor([0, 0, -1.0]), atol=1e-7)
    """ exp(log(R)) of a 180 degree rotation loses precision"""
    difference(0, 180, omega_hat=torch.tensor([0, 0, 1.0]), atol=1e-7)
    difference(0, 270, omega_hat=torch.tensor([0, 0, -1.0]), atol=1e-7)

    """ verify exception is raised when identity"""
    Rsb = z_rotation(0)
    Rsc = z_rotation(360)
    R = torch.matmul(Rsc, Rsb.inverse())
    assert torch.allclose(R, torch.eye(3))
    with pytest.raises(Exception):
        w, theta = matrix_log_rotation(R)

    def x_rotation(degrees):
        return torch.tensor([
            [1, 0, 0],
            [0, cos(radians(degrees)), -sin(radians(degrees))],
            [0, sin(radians(degrees)), cos(radians(degrees))],

        ])

    def difference_x(from_degrees, to_degrees, omega_hat, atol):
        Rsb = x_rotation(from_degrees)
        Rsc = x_rotation(to_degrees)
        R = torch.matmul(Rsc, Rsb.inverse())
        w, theta = matrix_log_rotation(R)
        w = uncross_matrix(w)
        expected = normalize_angle(radians(to_degrees) - radians(from_degrees))
        plot_angle(from_degrees, to_degrees, normalize_angle(w[0] * theta), expected)
        assert torch.allclose(w, omega_hat)
        assert -0.0001 < normalize_angle(w[0] * theta) - expected < 0.0001
        R_exp = matrix_exp_rotation(w, theta)
        assert torch.allclose(R, R_exp, atol=atol)

    difference_x(35, 65, omega_hat=torch.tensor([1.0, 0, 0.0]), atol=1e-8)
    difference_x(65, 35, omega_hat=torch.tensor([-1.0, 0, 0]), atol=1e-7)
    """ exp(log(R)) of a 180 degree rotation loses precision"""
    difference_x(0, 180, omega_hat=torch.tensor([1.0, 0, 0.0]), atol=1e-7)
    difference_x(0, 270, omega_hat=torch.tensor([-1.0, 0, 0.0]), atol=1e-7)

    """ verify exception is raised when identity"""
    Rsb = x_rotation(0)
    Rsc = x_rotation(360)
    R = torch.matmul(Rsc, Rsb.inverse())
    assert torch.allclose(R, torch.eye(3))
    with pytest.raises(Exception):
        w, theta = matrix_log_rotation(R)


def test_uncross():
    omega = torch.tensor([1.0, 2.0, 3.0])
    R = cross_matrix(omega)
    omega_uncross = uncross_matrix(R)
    assert torch.allclose(omega, omega_uncross)


def test_matrix_log():

    Tsb = torch.tensor([
        [cos(radians(30)), -sin(radians(30)), 0, 1.0],
        [sin(radians(30)), cos(radians(30)), 0, 2.0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    Tsc = torch.tensor([
        [cos(radians(60)), -sin(radians(60)), 0, 2],
        [sin(radians(60)), cos(radians(60)), 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    Tbc = torch.matmul(Tsc, Tsb.inverse())
    s, theta = matrix_log_transform(Tbc)

    x_axis_b = Tsb.matmul(x_axis)
    y_axis_b = Tsb.matmul(y_axis)
    plt.plot(x_axis_b[0], x_axis_b[1])
    plt.plot(y_axis_b[0], y_axis_b[1])

    x_axis_c = Tsc.matmul(x_axis)
    y_axis_c = Tsc.matmul(y_axis)
    plt.plot(x_axis_c[0], x_axis_c[1])
    plt.plot(y_axis_c[0], y_axis_c[1])

    plt.xlim(left=0, right=10)
    plt.ylim(bottom=0, top=10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.pause(2.0)

    assert torch.allclose(uncross_matrix(rotation(s)), torch.tensor([0, 0, 1.0]))
    assert torch.allclose(translation(s), torch.tensor([3.3660, -3.3660, 0]))
    assert torch.allclose(theta, torch.tensor([pi/6]))

    Tsb = torch.tensor([
        [1, 0, 0, 3.0],
        [0, 1, 0, 4.0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    Tsc = torch.tensor([
        [cos(radians(60)), -sin(radians(60)), 0, 4-sin(radians(30))],
        [sin(radians(60)), cos(radians(60)), 0, 4-cos(radians(30))],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    Tbc = torch.matmul(Tsc, Tsb.inverse())
    s, theta = matrix_log_transform(Tbc)

    assert torch.allclose(uncross_matrix(rotation(s)), torch.tensor([0, 0, 1.0]))
    assert torch.allclose(translation(s), torch.tensor([4.0, -4.0, 0]))
    assert torch.allclose(theta, torch.tensor([pi/3]))


def test_twist_qsh():
    q = torch.tensor([3.37, 3.37, 0])
    s = torch.tensor([0, 0, 1.0])
    h = 100.0
    thetadot = 1.0
    twist = twist_qsh(q, s, h, thetadot)
    S, thetadot = screw(twist)

    t = 0.0
    plt.ion()

    while t < 5.0:

        T = matrix_exp_screw(S, thetadot * t)

        Tsb = torch.tensor([
            [cos(radians(30)), -sin(radians(30)), 0, 1.0],
            [sin(radians(30)), cos(radians(30)), 0, 2.0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        plt.scatter(q[0], q[1])

        x_axis_b = Tsb.matmul(x_axis)
        y_axis_b = Tsb.matmul(y_axis)
        plt.plot(x_axis_b[0], x_axis_b[1])
        plt.plot(y_axis_b[0], y_axis_b[1])

        Tsc = torch.tensor([
            [cos(radians(60)), -sin(radians(60)), 0, 2],
            [sin(radians(60)), cos(radians(60)), 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        x_axis_c = Tsc.matmul(x_axis)
        y_axis_c = Tsc.matmul(y_axis)
        plt.plot(x_axis_c[0], x_axis_c[1])
        plt.plot(y_axis_c[0], y_axis_c[1])

        Tsq = T.matmul(Tsb)
        x_axis_q = Tsq.matmul(x_axis)
        y_axis_q = Tsq.matmul(y_axis)
        plt.plot(x_axis_q[0], x_axis_q[1])
        plt.plot(y_axis_q[0], y_axis_q[1])

        plt.xlim(left=0, right=10)
        plt.ylim(bottom=0, top=10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.pause(0.1)
        plt.cla()

        t += 0.1


def test_3d_plot():
    q = torch.tensor([3.37, 3.37, 0])
    s = torch.tensor([0, 0, 1.0])
    h = 10.0
    thetadot = 1.0
    twist = twist_qsh(q, s, h, thetadot)
    S, thetadot = screw(twist)
    t = 0.0

    Tsb = torch.tensor([
        [cos(radians(30)), -sin(radians(30)), 0, 1.0],
        [sin(radians(30)), cos(radians(30)), 0, 2.0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    trajectory = []

    while t < 10.0:
        T = matrix_exp_screw(S, thetadot * t)
        T = T.matmul(Tsb)
        p = premul(T, torch.zeros(3))
        trajectory += [p]
        t += 0.1

    trajectory = torch.stack(trajectory)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Data for a three-dimensional line
    xline = trajectory[:, 0].numpy()
    yline = trajectory[:, 1].numpy()
    zline = trajectory[:, 2].numpy()
    ax.plot3D(xline, yline, zline, 'gray')
    plt.pause(2.0)


def test_adjoint():
    """
    test inverse adjoint works
    """

    twist = torch.rand(6)

    Thf = torch.tensor([
        [1, 0, 0, -.10],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    twist_hat = inv_adjoint(Thf).matmul(twist)
    assert not torch.allclose(twist, twist_hat)
    twist_hat = adjoint(Thf).matmul(twist_hat)
    assert torch.allclose(twist, twist_hat)

    Tsc = torch.tensor([
        [cos(radians(60)), -sin(radians(60)), 0, 2],
        [sin(radians(60)), cos(radians(60)), 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    twist_hat = inv_adjoint(Tsc).matmul(twist)
    assert not torch.allclose(twist, twist_hat)
    twist_hat = adjoint(Tsc).matmul(twist_hat)
    assert torch.allclose(twist, twist_hat)


def test_robot_hand():
    """ modern robotics Example 3.28"""
    F_h = torch.tensor([0, 0, 0, 0, -5.0, 0])
    F_a = torch.tensor([0, 0, 0, 0, 0, 1.0])

    Thf = torch.tensor([
        [1, 0, 0, -.10],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    Taf = torch.tensor([
        [1, 0, 0, -0.25],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])

    F_hand_f = adjoint(Thf).T.matmul(F_h)
    F_apple_f = adjoint(Taf).T.matmul(F_a)
    assert torch.allclose(F_hand_f + F_apple_f, torch.tensor([0, 0, -0.75, 0, -6, 0]))


def test_car_twists():
    """ modern robotics example 3.23"""

    Tsb = torch.tensor([
        [-1, 0, 0, 4.0],
        [0, 1, 0, 0.4],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    Vs = torch.tensor([0, 0, 2.0, -2.0, -4.0, 0])
    Vb = torch.tensor([0, 0, -2, 2.8, 4.0, 0])

    assert torch.allclose(Vs, adjoint(Tsb).matmul(Vb))
    assert torch.allclose(Vb, adjoint(Tsb.inverse()).matmul(Vs))
    assert torch.allclose(Vs, inv_adjoint(Tsb.inverse()).matmul(Vb))
    assert torch.allclose(Vb, inv_adjoint(Tsb).matmul(Vs))