import numpy as np

from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.core.errors import ProblemError
from pySDC.implementations.datatype_classes.particles import particles, acceleration

class harmonicoscillator(penningtrap):
    def __init__(self, omega, epsilon):
        pass