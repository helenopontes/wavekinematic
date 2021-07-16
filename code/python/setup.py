import sys
from cx_Freeze import setup, Executable
import PySimpleGUI as sg
import math
from math import pi, cos, sqrt, sin, cosh, sinh
import numpy as np
import pandas as pd
from scipy import integrate
import sympy as sp
import random
import matplotlib.pyplot as plt
from airy import Airy
from stokes import Stokes
from pierson_moskowitz import PM
from jonswap import Jonswap
import warnings
warnings.filterwarnings("ignore")

base = None
if sys.platform == "win32":
    base = "Win32GUI"

executables = [
        Executable("wave_kinematics.py", base=base)
]

buildOptions = dict(
        packages = [],
        includes = ['numpy','pandas','scipy','random', 'matplotlib','airy','stokes','pierson_moskowitz','jonswap','warnings','math', 'PySimpleGUI'],
        include_files = [],
        excludes = []
)




setup(
    name = "wave_kinematics",
    version = "1.1",
    description = "build teste",
    options = dict(build_exe = buildOptions),
    executables = executables
 )
