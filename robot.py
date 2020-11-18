import torch

from se3 import matrix_exp_screw, adjoint, inv_adjoint


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


def singular(jac):
    """ checks if a 6 x 6 jacobian is singular (ie: linearly independent) """
    with torch.no_grad():
        try:
            A = jac.matmul(jac.T).inverse()
            torch.cholesky(A)
        except Exception:
            return False
        return True


def manipualibility_ellipsoid(jac):
    A = jac.matmul(jac.T).inverse()
    try:
        A = torch.cholesky(A)
        lam, v = torch.eig(A, eigenvectors=True)
    except Exception:
        """ your code should handle singularities, modern robotics 5.3 """
        raise Exception('Robot pose is at singularity')
    return A, lam, v
