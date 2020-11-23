import torch

from se3 import matrix_exp_screw, adjoint, inv_adjoint, matrix_log_transform, twist_vector, rotation, translation

from math import degrees

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


def space_to_body(M, s_list):
    """
    Converts screws in the space frame to screws in the body frame
    :param M: (4x4) transform to the end_effector in the space frame
    :param s_list: 6, J screws in the space frame
    :return: b_list: 6, J screws in the body frame
    """
    return inv_adjoint(M).matmul(s_list)


def body_to_space(M, b_list):
    """
    Converts screws in the body frame to screws in the space frame
    :param M: (4x4) transform to end effector in the space frame
    :param b_list: 6, J screws in the space frame
    :return: s_list: 6, J screws in the body frame
    """
    return adjoint(M).matmul(b_list)


def fkin_body(M, b_list, theta_list):
    """
    Computes position and pose of end effector
    :param M: home position of end effector in the space frame
    :param b_list: 6, J screws in the end effector frame
    :param theta_list: J joint angles
    :return: end effector position in the space frame
    """
    _, J = b_list.shape

    E = M.clone()
    for i in range(J):
        E = E.matmul(matrix_exp_screw(b_list[:, i], theta_list[i]))

    return E


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


class Singularity(Exception):
    pass


def angular_manip_ellipsoid(jac):
    return manip_ellipsoid(jac[0:3])


def translational_manip_ellipsoid(jac):
    return manip_ellipsoid(jac[3:6])


def manip_ellipsoid(jac):
    try:
        lam, v = torch.symeig(jac.matmul(jac.T).inverse(), eigenvectors=True)
    except Exception:
        """ your code should handle singularities, modern robotics 5.3 """
        raise Singularity('Robot pose is at singularity')
    lengths = lam.sqrt()
    axis = v[:, 0]

    return lengths, axis


def ikin_body(b_list, M, Tsd, theta_list_0, eomg, ev, maxcount=10):

    count = 0

    while count < maxcount:
        print([degrees(theta) for theta in theta_list_0])
        Tsb = fkin_body(M, b_list, theta_list_0)
        jb = jacobian_body(b_list, theta_list_0)

        # compute the twist from end effector Tsb to desired position Tsd
        Tbd = Tsb.inverse().matmul(Tsd)
        Vb, mag = matrix_log_transform(Tbd)
        Vb = twist_vector(Vb * mag)

        # if twist is small, we are done
        if Vb[0:3].norm() < eomg and Vb[3:6].norm() < ev:
            return theta_list_0

        # take a full gradient descent step
        delta = jb.pinverse().matmul(Vb)
        theta_list_0 += delta
        count += 1

    return theta_list_0