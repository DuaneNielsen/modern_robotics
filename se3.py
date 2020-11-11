from math import pi

import torch


def homo(v):
    return torch.cat((v, torch.ones(1)))


def unhomo(v):
    return v[0:3]


def twist(w, v):
    return torch.cat((w, v))


def cross_matrix(omega):
    """
    Returns the skew symetric matrix given a vector omega
    :param omega: 3D axis of rotation
    :return: skew symmetric matrix for axis
    given x, y, z returns
    [0, -z, y]
    [z, 0, -x]
    [-y, x, 0]
    """

    cross = torch.zeros(9).scatter(0, torch.tensor([5, 2, 1]), omega).reshape(3, 3)
    cross = cross + cross.T
    return cross * torch.tensor([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ], dtype=omega.dtype)


def uncross_matrix(cross_matrix):
    return cross_matrix.reshape(9).gather(0, torch.tensor([7, 2, 3]))


def matrix_exp_rotation(axis, theta):
    """
    rotation matrix around axis by angle theta
    :param axis: vector that points along the axis
    :param theta: the angle of rotation around the axis
    :return: rotation matrix
    """
    axis = axis / axis.norm()
    cross = cross_matrix(axis)
    return torch.eye(3) + torch.sin(theta) * cross + (1 - torch.cos(theta)) * cross.matmul(cross)


def rotation(T):
    """
    returns the rotation matrix from a Transform matrix
    :param T: (4x4) transform matrix
    :return: (3x3) rotation matrix
    """
    return T[0:3, 0:3]


def translation(T):
    return T[0:3, 3]


def adjoint(T):
    """
    returns Adjoint matrix to convert a twist from frame b to frame s
    Vs = [Adjoint(Tsb)] Vb
    :param T: (4x4) transform Tsb
    :return:
    """
    adj = torch.zeros(6, 6)
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    adj[0:3, 0:3] = R
    adj[3:6, 3:6] = R
    p = cross_matrix(p)
    adj[3:6, 0:3] = p.matmul(R)
    return adj


def inv_adjoint(T):
    """
    returns Adjoint matrix to convert a twist from frame s to frame b given Tsb
    Vb = [Adjoint(Tbs)] Vs
    :param T: (4x4) transform Tsb
    :return:
    """
    inv_adj = torch.zeros(6, 6)
    RT = T[0:3, 0:3].T
    p = T[0:3, 3]
    inv_adj[0:3, 0:3] = RT
    inv_adj[3:6, 3:6] = RT
    p = cross_matrix(p)
    inv_adj[3:6, 0:3] = (-RT).matmul(p)
    return inv_adj


def transform(R, p, dtype=torch.float):
    """
    returns Transform matrix Tsb
    vs = [Tsb] vb
    :param R: Rotation matrix s -> b
    :param p: vector from s -> b
    :param dtype:
    :return:
    """
    T = torch.eye(4, 4, dtype=dtype)
    T[0:3, 0:3] = R
    T[0:3, 3] = p
    return T


def inv_transform(R, p, dtype=torch.float):
    """
    vb = [Tbs] vs
    The inverse transform Tbs, given the transform parameters from s -> b
    :param R: Rotation matrix s -> b
    :param p: vector from s -> b
    :param dtype:
    :return:
    """
    inv_T = torch.eye(4, 4, dtype=dtype)
    inv_T[0:3, 0:3] = R.T
    p = (-R.T).matmul(p)
    inv_T[0:3, 3] = p
    return inv_T


def decompose_transform(T):
    R = rotation(T)
    p = T[0:3, 3]
    return R, p


def premul(T, v):
    """
    vs = [Tsb] vb
    :param T: 4x4 transformation matrix
    :param v: 3, N vector
    :return: 3, N vector
    """
    return unhomo(T.matmul(homo(v)))


def matrix_log_rotation(R):
    """
    Computes the derivative matrix so(3) given a rotation matrix SO(3)
    :param R: 3x3 rotation matrix in SO(3)
    :return: 3x3 cross product matrix in se3 and angular speed theta

    note: do not try to take the log of an identity matrix, you will get an exception
    as log(I) is not defined (same as log(0) is undefined)

    before calling this function check the Rotation matrix with torch.allclose(R, torch.eye(3))

    in the case of an identity rotation matrix, forget about rotation and just compute assuming pure translation
    see the code of matrix_log_transform
    """
    def logfactor(r):
        return 1.0 / torch.sqrt(2.0 * (1.0 + r))

    if torch.allclose(R, torch.eye(3)):
        """ it's the identity matrix, no rotation here, equivalent to taking log(0) """
        raise Exception("Rotation matrix is the identity, rotation matrix logarithm not defined, check the docs")
    else:
        trace = R.trace()
        if trace == -1:
            """ if the trace is -1, then theta must be pi"""
            theta = torch.tensor([pi])
            """ 
            2 * (1 + r) must be > 0 
            ie: r > -1
            """
            w = torch.zeros(3)
            if R[2, 2] > -1:
                r = logfactor(R[2, 2])
                w[0] = R[0, 2]
                w[1] = R[1, 2]
                w[2] = 1.0 + R[2, 2]
            elif R[1, 1] > -1:
                r = logfactor(R[1, 1])
                w[0] = R[0, 1]
                w[1] = 1.0 + R[1, 1]
                w[2] = R[2, 1]
            elif R[0, 0] > -1:
                r = logfactor(R[0, 0])
                w[0] = 1.0 + R[0, 0]
                w[1] = R[1, 0]
                w[2] = R[2, 0]
            else:
                raise Exception('no root found')
            w = w * r
            w = cross_matrix(w)
        else:
            """ main case """
            theta = torch.acos(0.5 * (trace - 1.0))
            w = (R - R.T) / (2.0 * torch.sin(theta))
    return w, theta


def twist_matrix(w, v):
    """

    :param w: 3x3 cross matrix
    :param v:
    :return:
    """
    S = torch.zeros(4, 4)
    S[0:3, 0:3] = w
    S[0:3, 3] = v
    return S


def twist_qsh(q, s, h, thetadot):
    w = s * thetadot
    v = (-s * thetadot).cross(q) + h * s * thetadot
    return torch.cat((w, v))


def screw(twist):
    """
    Decompose a twist into a screw axis and a speed (thetadot)

    :param twist: angular x, angular, y, angular z, x vel, y vel, z vel
    :return: 6 dimensional screw axis, magnitude theta
    """

    if torch.allclose(twist[0:3], torch.zeros(3)):
        thetadot = twist[3:6].norm()
    else:
        thetadot = twist[0:3].norm()
    return twist / thetadot, thetadot


def matrix_exp_screw(screw, thetadot):
    """
    given a screw and thetadot, compute the matrix exponential
    modern robotics eqn 3.88
    :param screw: normalized screw axis such that either || angular velocity || == 1 or || velocity || == 1
    :param thetadot:
    :return:
    """
    if not isinstance(thetadot, torch.Tensor):
        thetadot = torch.tensor([thetadot], dtype=torch.float)
    T = torch.eye(4, 4)
    v = screw[3:6]
    R = matrix_exp_rotation(screw[0:3], thetadot)
    w = cross_matrix(screw[0:3])
    G = thetadot * torch.eye(3, 3) + (1.0 - torch.cos(thetadot)) * w + (thetadot - torch.sin(thetadot)) * torch.matmul(w, w)
    v = G.matmul(v)
    T[0:3, 0:3] = R
    T[0:3, 3] = v
    return T


def matrix_log_transform(T):
    """
    returns the screw matrix in se(3) given the transform matrix SE(3) by taking the matrix logarithm
    :param T: SE(3) transform matrix (4x4)
    :return: normal se(3) screw matrix (4x4), speed theta
    """
    def inv_g(w, theta):
        """ modern robotics eqn 3.92 """
        squared_factor = (1/theta) - (0.5 * (1/torch.tan(theta/2)))
        return torch.eye(3) / theta - w / 2 + squared_factor * torch.matmul(w, w)

    if torch.allclose(rotation(T), torch.eye(3)):
        """ if no rotation, there is no angular velocity, so return normalized translation """
        w = torch.zeros(3)
        theta = translation(T).norm()
        v = translation(T) / theta

    else:
        w, theta = matrix_log_rotation(rotation(T))
        v = torch.matmul(inv_g(w, theta), translation(T))
    return twist_matrix(w, v), theta