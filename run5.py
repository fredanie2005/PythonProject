import TP1
import matplotlib.pyplot as plt
import renderer

X, Y, Z = TP1.pave_plein(1, 10, 1, 10, 1, 10)  
renderer.plot3D(X, Y, Z)