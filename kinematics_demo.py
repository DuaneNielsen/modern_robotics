import torch
from se3 import *
from vpython import cylinder, vector, rate

if __name__ == '__main__':

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

    S1 = torch.tensor([0, 0, 1, 0, 0, 0], dtype=torch.float)
    S2 = torch.tensor([0, 0, 1, 0, -1, 0], dtype=torch.float)
    S3 = torch.tensor([0, 0, 1, 0, -2, 0], dtype=torch.float)

    def _v(tensor):
        return vector(tensor[0], tensor[1], tensor[2])

    center = torch.tensor([
        [0, 0, 0, 1],
        [0, 0, 0.3, 0],
    ], dtype=torch.float).T


    def p(body):
        return _v(body[0:3, 0])

    def axis(body):
        return _v(body[0:3, 1])

    j1 = cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(0, 0, 1))
    j2 = cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(0, 1, 1))
    j3 = cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(0, 1, 0))
    j4 = cylinder(axis=vector(0, 0, 0.2), radius=0.2, color=vector(1, 1, 0))

    t = 0.0
    i = 0
    while t < 4 * 3.1417:

        V1 = matrix_exp_screw(S1, t)
        T2 = V1.matmul(J1)
        body = T2.matmul(center)
        j2.pos = p(body)
        j2.axis = axis(body)

        V2 = matrix_exp_screw(S2, t)
        T3 = V1.matmul(V2).matmul(J2)
        body = T3.matmul(center)
        j3.pos = p(body)
        j3.axis = axis(body)

        V3 = matrix_exp_screw(S3, t)
        T4 = V1.matmul(V2).matmul(V3).matmul(M)
        body = T4.matmul(center)
        j4.pos = p(body)
        j4.axis = axis(body)

        rate(12)
        t += 0.05
        i += 1
