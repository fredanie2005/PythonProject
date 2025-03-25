import matplotlib.pyplot as plt

# 1 - Transpose
def transpose(M):
  M_t = []
  for i in range(len(M[0])):
    M_t.append([])
    for j in range(len(M)):
      M_t[i].append(M[j][i])
  return M_t

# 2 - Tracé de ligne
def ligne(n,xmin,xmax):
    W = []
    dx=(xmax-xmin)/(n-1)
    for x in range (n):
      W.append([x*dx,0,0])
    return(transpose(W))

# 3 - Carre Vide

def carre_vide(n,a):
  X = []
  Y = []
  Z = []
  
  X.append(ligne(20,-5,5))
  Y.append(ligne(20,-5,5))
  Z.append(ligne(0,0,0))
  return(X,Y,Z)

# 4 - Carre Plein

def carre_plein(n,a):
  X = []
  Y = []
  Z = []
  for x in range(n):
      for y in range(n):
          X.append(x*a)
          Y.append(y*a)
          Z.append(0)
  return(X,Y,Z)

# 5 - Cube 3D

def pave_plein(xmin, xmax, ymin, ymax, zmin, zmax):
    X = []
    Y = []
    Z = []
    for x in range(xmin, xmax + 1):  
        for y in range(ymin, ymax + 1): 
            for z in range(zmin, zmax + 1):
                X.append(x)
                Y.append(y)
                Z.append(z)
    return X, Y, Z




