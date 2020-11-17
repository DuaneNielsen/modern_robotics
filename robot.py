import torch

from se3 import matrix_exp_screw, adjoint


def fkin_space(M, s_list, theta_list):
    """
    Computes the position end effector given end_effector home position, list of screws in the space (inertial) frame
    and joint angles
    :param s_list: 6, J screws in the space frame
    :param theta_list: J joint angles (pose of the robot)
    :return: (4x4) transformation matrix for the pose of the end effector in the body frame
    """
    _, J = s_list.shape

    A = torch.eye(4)
    for i in range(J):
        A = A.matmul(matrix_exp_screw(s_list[:, i], theta_list[i]))

    return A.matmul(M)


def jacobian_space(s_list, theta_list):
    """
    Computes the jacobian in the space frame for J jointed robot
    :param s_list: 6, J screws in the space frame, J is the number of joints
    :param theta_list: J joint positions (pose of the robot)
    :return: 6, J space jacobian
    """

    _, J = s_list.shape

    A = torch.eye(4)
    jac = []

    for i in range(J):
        jac += [adjoint(A).matmul(s_list[:, i])]
        A = A.matmul(matrix_exp_screw(s_list[:, i], theta_list[i]))

    return torch.stack(jac, dim=1)


def jacobian_body(b_list, theta_list):
    """
    Computes the body jacobian for a J jointed robot
    :param b_list: 6, J screws in the body frame, J is the number of joints
    :param theta_list: J joint positions (pose of the robot)
    :return: 6, J body jacobian
    """
    _, J = b_list.shape

    A = torch.eye(4)
    jac = []

    for i in reversed(range(J)):
        jac += [adjoint(A).matmul(b_list[:, i])]
        A = A.matmul(matrix_exp_screw(-b_list[:, i], theta_list[i]))

    return torch.stack(list(reversed(jac)), dim=1)


