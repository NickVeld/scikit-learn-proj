#Made by Yaroslav
import numpy as np
from random import random
import math

class Generator():
    def __init__(self, n_components=2,
                 border_array=np.array([[0, 1], [0, 1]]),
                 function_array=np.array([(lambda x: x[0]), (lambda x: x[1]), (lambda x: math.cos(x[0]) + math.cos(x[1]))])):
        self.n_components = n_components
        self.border_array = border_array
        self.function_array = function_array

    def generate_net(self, n_points):
        return np.array(sorted(list(
            (
                list(
                    self.border_array[j][0] + random() * (self.border_array[j][1] - self.border_array[j][0])
                    for j in range(self.n_components)
                )
            ) for i in range(n_points)
        )))

    def process_net(self, net):
        return np.array(list(
            np.array(list(
                function(net[i])
                for function in self.function_array
            ))
            for i in range(net.shape[0])
        ))

    def generate_colors(self, n_points, color_data):
        q = (color_data[1] - color_data[0]) / (max(n_points - 1, 1))
        return np.array(
            list(
                color_data[0] + q * i
                for i in range(n_points)
            )
        )

    def generate_manifold(self, n_points, color_data=np.array([np.array([0,1,0]), np.array([0,0,1])])):
        return self.process_net(self.generate_net(n_points)), self.generate_colors(n_points, color_data)

class Mobius(Generator):
    def __init__(self, width=2, radius=1):
        super().__init__(
            n_components=2,
            border_array=np.array([[0, 2 * math.pi], [-width / 2, width / 2]]),
            function_array=np.array(
                [
                    (lambda x: (radius + x[1] / 2 * math.cos(x[0] / 2)) * math.cos(x[0])),
                    (lambda x: (radius + x[1] / 2 * math.cos(x[0] / 2)) * math.sin(x[0])),
                    (lambda x: x[1] / 2 * math.sin(x[0] / 2))
                ]))

class Ring(Generator):
    def __init__(self, width=2, radius=1):
        super().__init__(n_components=2,
              border_array=np.array([[0, 2 * math.pi], [-width / 2, width / 2]]),
              function_array=np.array(
                  [
                      (lambda x: radius * math.cos(x[0])),
                      (lambda x: radius * math.sin(x[0])),
                      (lambda x: x[1])
                  ]))

class Helix(Generator):
    def __init__(self, step=1, twists=1, width=1, offset=0):
        super().__init__(n_components=2,
              border_array=np.array([[0, twists * 2 * math.pi], [offset - width / 2, offset + width / 2]]),
              function_array=np.array(
                  [
                      (lambda x: x[1] * math.cos(x[0])),
                      (lambda x: x[1] * math.sin(x[0])),
                      (lambda x: step * x[0] / 2 / math.pi)
                  ]))

class S_curve(Generator):
    def __init__(self):
        super().__init__(n_components=2,
                         border_array=np.array([[-1/3 * math.pi, 4/3 * math.pi], [0, 1]]),
                         function_array=np.array(
                             [
                                 (lambda x: x[1]),
                                 (lambda x: math.sin(2 * x[0])),
                                 (lambda x: math.cos(x[0])),
                             ]))

class Spiral(Generator):
    def __init__(self):
        super().__init__(n_components=2,
                         border_array=np.array([[0, 10],[0, 10]]),
                         function_array=np.array(
                             [
                                 (lambda x: x[1]),
                                 (lambda x: x[0] * math.sin(x[0])),
                                 (lambda x: x[0] * math.cos(x[0])),
                             ]))
