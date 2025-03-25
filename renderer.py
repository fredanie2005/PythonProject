import matplotlib.pyplot as plt

def plot3D(X, Y, Z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, Z, c=Z, cmap='ocean')
    fig.suptitle("DX13") 
    plt.show()
    