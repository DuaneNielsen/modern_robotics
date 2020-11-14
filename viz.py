import torch
from vpython import arrow, vector


def _v(tensor):
    return vector(tensor[0], tensor[1], tensor[2])


""" 
origin and 3 principle axis of a frame, homogenous co-ords,
such that F = T.matmul(F) where T is a transformation matrix and F is the frame
"""

frame = torch.tensor([
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
], dtype=torch.float).T


def origin(frame):
    return _v(frame[0:3, 0])


def x_axis(frame):
    return _v(frame[0:3, 1])


def y_axis(frame):
    return _v(frame[0:3, 2])


def z_axis(frame):
    return _v(frame[0:3, 3])


class FrameViz:
    def __init__(self):
        self.x_arrow = arrow(thickness=0.1, color=vector(0.1, 0.1, 0.5))
        self.y_arrow = arrow(thickness=0.1, color=vector(0.5, 0.5, 0.1))
        self.z_arrow = arrow(thickness=0.1, color=vector(0.5, 0.1, 0.1))

    def update(self, frame, length=0.3):
        """

        :param frame: frame to input
        :param length:
        :return:
        """
        self.x_arrow.pos = origin(frame)
        self.x_arrow.axis = (x_axis(frame) - origin(frame)) * length
        self.y_arrow.pos = origin(frame)
        self.y_arrow.axis = (y_axis(frame) - origin(frame)) * length
        self.z_arrow.pos = origin(frame)
        self.z_arrow.axis = (z_axis(frame) - origin(frame)) * length