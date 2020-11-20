import torch
from vpython import arrow, vector, canvas, cylinder, rate, sphere
from robot import jacobian_space
from se3 import matrix_exp_screw
from math import pi

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


def tm(x=0.0, y=0.0, z=0.0, R=None):
    """ returns a translation matrix """
    t = torch.tensor([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ], dtype=torch.float)
    if R is not None:
        t[0:3, 0:3] = R
    return t


class FrameViz:
    def __init__(self, thickness=0.1):
        self.x_arrow = arrow(thickness=thickness, color=vector(1.0, 0.1, 0.1), visible=False)
        self.y_arrow = arrow(thickness=thickness, color=vector(0.1, 1.0, 0.1), visible=False)
        self.z_arrow = arrow(thickness=thickness, color=vector(0.1, 0.1, 1.0), visible=False)

    def update(self, frame, length=None):
        """

        :param frame: frame to input
        :param length: vector of lengths of the axis to display
        :return:
        """
        if length is None:
            length = vector(0.3, 0.3, 0.3)
        self.x_arrow.pos = origin(frame)
        self.x_arrow.axis = (x_axis(frame) - origin(frame)) * length.x
        self.x_arrow.visible = True

        self.y_arrow.pos = origin(frame)
        self.y_arrow.axis = (y_axis(frame) - origin(frame)) * length.y
        self.y_arrow.visible = True

        self.z_arrow.pos = origin(frame)
        self.z_arrow.axis = (z_axis(frame) - origin(frame)) * length.z
        self.z_arrow.visible = True


class RobotViz:
    def __init__(self, joint_home_list, s_list):

        self.joints = joint_home_list
        self.n_joints = len(joint_home_list)
        self.screws = s_list

        self.scene = canvas(width=1200, height=1200)
        self.scene.camera.pos = vector(0, -2.4, -0.2)
        self.scene.camera.axis = vector(0, 2.4, 0.27)
        self.scene.camera.up = vector(0, -1, 0)

        self.joints_viz = [cylinder(axis=vector(0, 0, 0.0), radius=0.04, color=vector(0.1, 0.3, 0.4), visible=False)
                           for _ in range(self.n_joints)]
        self.links_viz = [None]
        self.links_viz += [cylinder(radius=0.03, color=vector(0.6, 0.6, 0.6), visible=False) for _ in range(self.n_joints-1)]

    def update(self, theta, fps=24):

        ts = [matrix_exp_screw(self.screws[:, i], theta[i]) for i in range(self.n_joints)]

        tb = tm()
        prev_joint_pos = vector(0, 0, 0)
        end_frame = None

        for tf, j, jv, lv in zip(ts, self.joints, self.joints_viz, self.links_viz):
            tb = tb.matmul(tf)
            joint_frame = tb.matmul(j).matmul(frame)

            jv.visible = True
            jv.pos = origin(joint_frame)
            jv.axis = (z_axis(joint_frame) - origin(joint_frame)) * 0.05
            if lv:
                lv.visible = True
                lv.pos = prev_joint_pos
                axis = jv.pos - prev_joint_pos
                if axis.mag == 0:
                    lv.visible = False
                else:
                    lv.axis = axis

            prev_joint_pos = origin(joint_frame)
            end_frame = joint_frame

        rate(fps)
        return end_frame