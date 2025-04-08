import matplotlib.pyplot as plt
pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679

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

def factoriel(n):
    return 1 if n == 0 else n * factoriel(n - 1)


def cosinus(x, terms=10):
    result = 0
    for k in range(terms):
        result += (-1)**k * (x**(2 * k)) / factoriel(2 * k)
    return result


def sinus(x, terms=10):
    result = 0
    for k in range(terms):
        result += (-1)**k * (x**(2 * k + 1)) / factoriel(2 * k + 1)
    return result

def cercle_plein(n, R):
    points = []
    step_r = R / (n**0.5)
    step_teta = (2 * pi) / (n**0.5)

    for i in range(int(n**0.5)):
        r = i * step_r
        for j in range(int(n**0.5)):
            theta = j * step_teta
            x, y = r * cosinus(theta), r * sinus(theta)
            points.append([x, y, 0])

    return transpose(points)



def cylindre_plein(n, R, h):
    points = []

    step_r = R / (n**0.5)
    step_theta = 2 * pi / (n**0.5)
    step_z = h / (n**0.5)

    for k in range(int(n**0.5)):
        z = k * step_z
        for i in range(int(n**0.5)):
            r = i * step_r
            for j in range(int(n**0.5)):
                theta = j * step_theta
                x = r * cosinus(theta)
                y = r * sinus(theta)
                points.append([x, y, z])

    return transpose(points)