import torch

from se3 import matrix_exp_screw, adjoint


def jacobian_space(s_list, theta_list):

    es_t = []

    for screw, theta in zip(s_list, theta_list):
        es_t += [matrix_exp_screw(screw, theta)]

    chain = []
    T = torch.eye(4)
    for p in es_t[:-1]:
        T = T.matmul(p)
        chain += [adjoint(T)]

    j = [s_list[0]]
    for i, adj in enumerate(chain):
        j += [adj.matmul(s_list[i + 1])]

    return torch.stack(j).T