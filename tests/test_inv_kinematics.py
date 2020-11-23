import torch
from torch import allclose, tensor
from se3 import matrix_log_transform, inv_adjoint, decompose_transform, inv_transform, matrix_exp_screw, screw, twist_vector

from math import pi, radians, cos, sin, degrees
from liegroups.torch.se3 import SE3Matrix as SE3
from robot import fkin_body, jacobian_body, ikin_body


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

def rad2degrees(theta_list):
    return theta_list * 180 / pi


def test_inv_kinematics_liegroups():
    """modern robotics Example 6.1"""

    Tsd = tensor([
        [-0.5, -0.866, 0, 0.366],
        [0.866, -0.5, 0, 1.366],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Tsd = SE3.from_matrix(Tsd, normalize=True)

    xd = torch.tensor([0.366, 1.386, 0., 0., 0., cos(-0.5)])

    M = tensor([
        [1., 0, 0, 2.],
        [0, 1., 0, 0],
        [0, 0, 1., 0],
        [0, 0, 0, 1.]
    ])
    M = SE3.from_matrix(M)

    # B1 = tensor([0, 0, 1., 0, 2., 0.])
    # B2 = tensor([0, 0, 1., 0, 1., 0.])

    B1 = tensor([0, 2., 0., 0, 0, 1.])
    B2 = tensor([0, 1., 0., 0, 0, 1.])

    b_list = torch.stack((B1, B2))

    theta_list = tensor([0., radians(30.0)])

    def fkin_body_l(M, b_list, theta_list):
        """
        Computes position and pose of end effector
        :param M: SE3(4x4) home position of end effector in the space frame
        :param b_list: se3(J, 6) (vel, angular vel)
        :param theta_list: J joint angles
        :return: SE3Matrix end effector position in the space frame
        """

        theta_list = theta_list.unsqueeze(1)

        V = b_list * theta_list
        v = SE3.exp(V)

        for ex in v.as_matrix():
            M = M.dot(SE3.from_matrix(ex))
        return M

    def jacobian_body_l(b_list, theta_list):
        """
        Computes the body jacobian for a J jointed robot
        :param b_list: 6, J screws in the body frame, J is the number of joints
        :param theta_list: J joint positions (pose of the robot)
        :return: 6, J body jacobian
        """
        J, _ = b_list.shape

        A = SE3.from_matrix(torch.eye(4))
        jac = []

        for i in reversed(range(J)):
            jac += [A.adjoint().matmul(b_list[i])]
            A = A.dot(SE3.exp(-b_list[i] * theta_list[i]))

        return torch.stack(list(reversed(jac)), dim=1)

    delta = torch.ones(2)
    count = 0

    while delta.norm().sqrt() > 0.0001 and count < 10:
        print([degrees(theta) for theta in theta_list])
        Tsb = fkin_body_l(M, b_list, theta_list)
        jb = jacobian_body_l(b_list, theta_list)
        Tbd = Tsb.inv().dot(Tsd)
        Vb = Tbd.log()
        delta = jb.pinverse().matmul(Vb)
        theta_list += delta
        count += 1
        #print(degrees(delta[0]), degrees(delta[1]))


def test_inverse_kinematics_myway():

    Tsd = tensor([
        [-0.5, -0.866, 0, 0.366],
        [0.866, -0.5, 0, 1.366],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    xd = torch.tensor([0.366, 1.386, 0., 0., 0., cos(-0.5)])

    M = tensor([
        [1., 0, 0, 2.],
        [0, 1., 0, 0],
        [0, 0, 1., 0],
        [0, 0, 0, 1.]
    ])

    B1 = tensor([0, 0, 1., 0, 2., 0.])
    B2 = tensor([0, 0, 1., 0, 1., 0.])

    b_list = torch.stack((B1, B2), dim=1)

    theta_list = tensor([0., radians(30.0)])

    delta = torch.ones(2)
    count = 0

    while delta.norm().sqrt() > 0.0001 and count < 10:
        print([degrees(theta) for theta in theta_list])
        Tsb = fkin_body(M, b_list, theta_list)
        jb = jacobian_body(b_list, theta_list)
        Tbd = Tsb.inverse().matmul(Tsd)
        Vb, mag = matrix_log_transform(Tbd)
        Vb = twist_vector(Vb * mag)
        delta = jb.pinverse().matmul(Vb)
        theta_list += delta
        count += 1

def test_inv_kinematics_from_robot():

    Tsd = tensor([
        [-0.5, -0.866, 0, 0.366],
        [0.866, -0.5, 0, 1.366],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    M = tensor([
        [1., 0, 0, 2.],
        [0, 1., 0, 0],
        [0, 0, 1., 0],
        [0, 0, 0, 1.]
    ])

    B1 = tensor([0, 0, 1., 0, 2., 0.])
    B2 = tensor([0, 0, 1., 0, 1., 0.])

    b_list = torch.stack((B1, B2), dim=1)

    theta_list_initial = tensor([0., radians(30.0)])

    theta_list_solution = ikin_body(b_list, M, Tsd, theta_list_initial, 0.0001, 0.0001, maxcount=4)

    assert allclose(rad2degrees(theta_list_solution), tensor([30.0, 90.0]), atol=0.01)