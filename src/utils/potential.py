import numpy as np

class MorsePotentialDW():
    """
    A double-well potential based on Morse potentials
    used by Igeh, Jonny A. in his thesis work.
    Morse potential based on: https://doi.org/10.1103/PhysRev.34.57, 
    but the zero-point has been moved to prevent multiple (unecessary) evaluations of the exponential function.

    params:
    D_a: Depth of the left well
    D_b: Depth of the right well
    a: Width of the left well
    b: Width of the right well
    d: Distance between center of the wells
    """
    def __init__(self, D_a=20, D_b=20, k_a=1.0, k_b=1.0, d=4):
        self.D_a = D_a 
        self.D_b = D_b
        self.a = np.sqrt(k_a / (2 * D_a))
        self.b = np.sqrt(k_b / (2 * D_b))
        self.d = d
        self.x_a = -d/2
        self.x_b = d/2
        self.left_pot = lambda x: self.D_a * (1 - np.exp(-self.a * (x - self.x_a)))**2
        self.right_pot = lambda x: + self.D_b * (1 - np.exp(self.b * (x - self.x_b)))**2

    def __call__(self, x):
        return (
            self.D_a * (1 - np.exp(-self.a * (x - self.x_a)))**2
             + self.D_b * (1 - np.exp(self.b * (x - self.x_b)))**2
        )
    
    def derivative(self, x):
        return (
            2 * self.D_a * self.a * (1 - np.exp(-self.a * (x - self.x_a))) * np.exp(-self.a * (x - self.x_a))
            - 2 * self.D_b * self.b * (1 - np.exp(self.b * (x - self.x_b))) * np.exp(self.b * (x - self.x_b))
        )
    
    @property
    def params(self):
        return self.D_a, self.D_b, self.a, self.b, self.d