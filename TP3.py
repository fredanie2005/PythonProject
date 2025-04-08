import TP1
import renderer
import matplotlib.pyplot as plt
import TP2

# Question 1
X, Y, Z = TP1.cylindre_plein(100, 1, 5)
renderer.plot3D(X, Y, Z)

# Question 2

def somme_vecteurs(vecteurs):
    somme = [0, 0, 0]
    for vecteur in vecteurs:
        for i in range(3):
            somme[i] += vecteur[i]
    return somme

def multiplication_vecteur_scalaire(vecteur, scalaire):
    return [composante * scalaire for composante in vecteur]

def division_vecteur_scalaire(vecteur, scalaire):
    return [composante / scalaire for composante in vecteur]

def addition_vecteurs3(vecteur1, vecteur2):
    return [vecteur1[i] + vecteur2[i] for i in range(3)]

def translation(m, F, G, vG, h):

    somme_forces = somme_vecteurs(F)

    acceleration = multiplication_vecteur_scalaire(somme_forces, 1 / m)
    
    nouvelle_vitesse = addition_vecteurs3(vG, multiplication_vecteur_scalaire(acceleration, h))

    nouvelle_position = addition_vecteurs3(G, multiplication_vecteur_scalaire(vG, h))
    
    return nouvelle_position, nouvelle_vitesse

# Question 3

# Quetion 4 

def produit_vectoriel(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]

def rotation(I_inv, F, G, teta, omega, h):
    """
    Calcule la rotation d'un objet et retourne la matrice de rotation.
    
    Paramètres:
    - I_inv: matrice inverse du tenseur d'inertie
    - F: liste des moments de force appliqués
    - G: position du centre de gravité
    - teta: angles de rotation actuels [θx, θy, θz]
    - omega: vitesse angulaire actuelle [ωx, ωy, ωz]
    - h: pas de temps
    
    Retourne:
    - R: matrice de rotation
    - nouveau_omega: nouvelle vitesse angulaire
    """
    import math
    
    # Calculer les moments résultants des forces par rapport au centre de gravité
    moments = []
    for force in F:
        moment = produit_vectoriel(G, force)
        moments.append(moment)
    
    # Somme des moments
    somme_moments = somme_vecteurs(moments)
    
    # Calculer l'accélération angulaire α = I_inv * M
    alpha = [0, 0, 0]
    for i in range(3):
        for j in range(3):
            alpha[i] += I_inv[i][j] * somme_moments[j]
    
    # Calculer la nouvelle vitesse angulaire ω' = ω + α*h
    nouveau_omega = addition_vecteurs3(omega, multiplication_vecteur_scalaire(alpha, h))
    
    # Calculer les nouveaux angles de rotation θ' = θ + ω*h + (1/2)*α*h²
    terme_acceleration = multiplication_vecteur_scalaire(alpha, 0.5 * h * h)
    terme_vitesse = multiplication_vecteur_scalaire(omega, h)
    nouveau_teta = addition_vecteurs3(teta, addition_vecteurs3(terme_vitesse, terme_acceleration))
    
    # Extraire les angles d'Euler
    roll = nouveau_teta[0]   # rotation autour de X
    pitch = nouveau_teta[1]  # rotation autour de Y
    yaw = nouveau_teta[2]    # rotation autour de Z
    
    # Calculer les sinus et cosinus
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    
    # Construire la matrice de rotation directement
    R = [
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ]
    
    return R, nouveau_omega

# Question 5

# Question 6

def mouvement(W , m , I  , F , G , vG , omega , h ,  n): 
    return