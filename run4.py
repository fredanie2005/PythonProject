import main
import matplotlib.pyplot as plt
import renderer

X, Y, Z = main.carre_plein(3,3)  
renderer.plot3D(X, Y, Z)