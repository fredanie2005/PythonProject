import main
import matplotlib.pyplot as plt
import renderer

X, Y, Z = main.pave_plein(1, 10, 1, 10, 1, 10)  
renderer.plot3D(X, Y, Z)