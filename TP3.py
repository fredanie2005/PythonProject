import TP1
import renderer
import matplotlib.pyplot as plt
import TP2

# Question 1
X, Y, Z = TP1.cylindre_plein(100, 1, 5)

# Question 2

def somme_vecteurs(vecteurs):
    somme = [0, 0, 0]
    for vecteur in vecteurs:
        for i in range(3):
            somme[i] += vecteur[i]
    return somme

def somme_forces(F):
    total = [0, 0, 0]
    for force_point in F:
        force = force_point[0]
        for i in range(3):
            total[i] += force[i]
    return total

def multiplication_vecteur_scalaire(vecteur, scalaire):
    return [composante * scalaire for composante in vecteur]

def division_vecteur_scalaire(vecteur, scalaire):
    return [composante / scalaire for composante in vecteur]

def addition_vecteurs3(vecteur1, vecteur2):
    return [vecteur1[i] + vecteur2[i] for i in range(3)]

def translation(m, F, G, vG, h):

    somme_forces = somme_forces(F)

    acceleration = multiplication_vecteur_scalaire(somme_forces, 1 / m)
    
    nouvelle_vitesse = addition_vecteurs3(vG, multiplication_vecteur_scalaire(acceleration, h))

    nouvelle_position = addition_vecteurs3(G, multiplication_vecteur_scalaire(vG, h))
    
    return nouvelle_position, nouvelle_vitesse

# Question 3

# Quetion 4 

def produit_vectoriel(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]

def rotation(I_inv, F, G, teta, omega, h):
    return
# Question 5

# Question 6

def mouvement(W , m , I  , F , G , vG , omega , h ,  n): 
    return

def translater_cylindre(W, vecteur):
    W_new = []
    for point in W:
        W_new.append([point[i] + vecteur[i] for i in range(3)])
    return W_new

def tracer_positions():
    n = 100
    R = 2
    h = 2
    m = 1.0
    h_temps = 0.1  # pas de temps

    # Forces : 3 forces appliquées à différents points
    F = [
        [[1, 0, 0], [0, 0, 0]],
        [[0, 1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0]]
    ]

    G = [0, 0, 0]
    vG = [0, 0, 0]

    cylindre = TP1.cylindre_plein(n, R, h)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for step in range(10):
        cylindre_translated = translater_cylindre(cylindre, G)
        x = [p[0] for p in cylindre_translated]
        y = [p[1] for p in cylindre_translated]
        z = [p[2] for p in cylindre_translated]
        ax.scatter(x, y, z, alpha=0.4)
        G, vG = translation(m, F, G, vG, h_temps)

    ax.set_title("Translation du cylindre sur 10 positions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()

tracer_positions()