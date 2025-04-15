import matplotlib.pyplot as plt # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore
import random


def transpose(matrix):
    result = []
    for i in range(len(matrix[0])):
        newMatrix = []
        for j in range(len(matrix)):
            newMatrix.append(matrix[j][i])
        result.append(newMatrix)
    return result


def plot_3d(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=Z, cmap='ocean')
    plt.show()


def ligne(n, xmin, xmax):
    W = []
    dx = (xmax - xmin) / (n - 1)
    for x in range(n):
        W.append([x * dx, 0, 0])
    return (transpose(W))


def carrevide(n, a):
    points = []
    step = a / (n // 4)
    for i in range(n // 4):
        points.append([i * step - a / 2, -a / 2, 0])
        points.append([a / 2, i * step - a / 2, 0])
        points.append([a / 2 - i * step, a / 2, 0])
        points.append([-a / 2, a / 2 - i * step, 0])
    return transpose(points)


def carreplein(n, a):
    points = []
    side = int(n**0.5)
    step = a / side
    for i in range(side):
        for j in range(side):
            points.append([i * step - a / 2, j * step - a / 2, 0])
    return transpose(points)


def pave_plein(n, a, b, c):
    points = []
    side = int(n**(1 / 3))
    dx, dy, dz = a / side, b / side, c / side
    for i in range(side):
        for j in range(side):
            for k in range(side):
                points.append([i * dx - a / 2, j * dy - b / 2, k * dz - c / 2])
    return transpose(points)


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


def cercle_pleinRANDOMISE(n, R):
    points = []
    for i in range(n):
        r = (random.uniform(0, R**2))**0.5
        theta = random.uniform(0, 2 * 3.141592653589793)
        x, y = r * cosinus(theta), r * sinus(theta)
        points.append([x, y, 0])
    return transpose(points)


def cylindre_pleinRANDOMISE(n, R, h):
    points = []
    for i in range(n):
        r = (random.uniform(0, R**2))**0.5
        theta = random.uniform(0, 2 * 3.141592653589793)
        z = random.uniform(-h / 2, h / 2)
        x, y = r * cosinus(theta), r * sinus(theta)
        points.append([x, y, z])
    return transpose(points)


def cercle_plein(n, R):
    points = []
    step_r = R / (n**0.5)
    step_theta = (2 * 3.141592653589793) / (n**0.5)

    for i in range(int(n**0.5)):
        r = i * step_r
        for j in range(int(n**0.5)):
            theta = j * step_theta
            x, y = r * cosinus(theta), r * sinus(theta)
            points.append([x, y, 0])

    return transpose(points)



def cylindre_plein(n, R, h):
    points = []

    step_r = R / (n**0.5)
    step_theta = 2 * 3.141592653589793 / (n**0.5)
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

    return points


def prodmat(A, B):
    if len(A[0]) != len(B):
        return "Erreur : les matrices ne sont pas compatibles"
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result


def det(A):
    n = len(A)
    if any(len(row) != n for row in A):
        return "La matrice doit être carrée."
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    determinant = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j + 1:] for row in A[1:]]
        determinant += ((-1)**j) * A[0][j] * det(sub_matrix)
    return determinant


def com(A):
    n = len(A)
    if any(len(row) != n for row in A):
        return "La matrice doit être carrée."

    comatrix = []
    for i in range(n):
        row = []
        for j in range(n):
            sub_matrix = []
            for k in range(n):
                if k != i:
                    sub_row = A[k][:j] + A[k][j + 1:]
                    sub_matrix.append(sub_row)
            cofactor = ((-1)**(i + j)) * det(sub_matrix)
            row.append(cofactor)
        comatrix.append(row)
    return comatrix


def inv(A):
    n = len(A)
    if any(len(row) != n for row in A):
        return "La matrice doit être carrée."
    determinant = det(A)
    if determinant == 0:
        return "La matrice n'est pas inversible."
    comatrix = com(A)
    adjugate = transpose(comatrix)
    inverse = [[element // determinant for element in row] for row in adjugate]
    return inverse


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

def produit_vectoriel(u, v):
    return [
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    ] 

def somme_forces(F):
    total = [0, 0, 0]
    for force_point in F:
        force = force_point[0]
        for i in range(3):
            total[i] += force[i]
    return total

def translation(m, F, G, vG, h):

    sommeforces = somme_forces(F)

    acceleration = multiplication_vecteur_scalaire(sommeforces, 1 / m)

    nouvelle_vitesse = addition_vecteurs3(vG, multiplication_vecteur_scalaire(acceleration, h))

    nouvelle_position = addition_vecteurs3(G, multiplication_vecteur_scalaire(vG, h))

    return nouvelle_position, nouvelle_vitesse


def translater_cylindre(W, vecteur):
    W_new = []
    for point in W:
        W_new.append([point[i] + vecteur[i] for i in range(3)])
    return W_new
# Fonctions nécessaires pour la rotation
def produit_vectoriel(u, v):
    return [
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    ]

def rotation(I_inv, F, G, teta, omega, h):
    # Calcul du torque total
    torque = [0.0, 0.0, 0.0]
    for force, point in F:
        r = [point[i] - G[i] for i in range(3)]  # Vecteur de G au point d'application
        t = produit_vectoriel(r, force)
        for i in range(3):
            torque[i] += t[i]
    
    # Calcul de l'accélération angulaire alpha = I⁻¹ * torque
    alpha = [
        sum(I_inv[i][j] * torque[j] for j in range(3))
        for i in range(3)
    ]
    
    # Mise à jour de la vitesse angulaire
    new_omega = [omega[i] + alpha[i] * h for i in range(3)]
    
    # Mise à jour de l'orientation (méthode d'Euler)
    new_teta = [teta[i] + new_omega[i] * h for i in range(3)]
    
    return new_teta, new_omega

def rotation_matrix(teta):
    # Crée une matrice de rotation à partir des angles d'Euler (ZYX)
    cx = cosinus(teta[0])
    sx = sinus(teta[0])
    cy = cosinus(teta[1])
    sy = sinus(teta[1])
    cz = cosinus(teta[2])
    sz = sinus(teta[2])
    
    return [
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy, cy*sx, cy*cx]
    ]

def rotate_points(points, R):
    # Applique la matrice de rotation à tous les points
    rotated = []
    for p in points:
        new_p = [
            R[0][0]*p[0] + R[0][1]*p[1] + R[0][2]*p[2],
            R[1][0]*p[0] + R[1][1]*p[1] + R[1][2]*p[2],
            R[2][0]*p[0] + R[2][1]*p[1] + R[2][2]*p[2]
        ]
        rotated.append(new_p)
    return rotated

# Fonction de simulation mise à jour
def tracer_positions():
    n = 50
    R = 2
    h = 5  # hauteur du cylindre
    m = 1.0
    h_temps = 0.1
    
    # Tenseur d'inertie inverse pour un cylindre (axe Z)
    I_zz = 0.5 * m * R**2
    I_xx = (1/12) * m * (3*R**2 + h**2)
    I_inv = [
        [1/I_xx, 0, 0],
        [0, 1/I_xx, 0],
        [0, 0, 1/I_zz]
    ]
    
    # Configuration initiale
    F = [
        [[0, 2, 0], [R, 0, 0]],  # Force créant un couple
        [[0, -2, 0], [-R, 0, 0]] # Force opposée
    ]
    G = [0, 0, 0]
    vG = [0, 0, 0]
    teta = [0.0, 0.0, 0.0]
    omega = [0.0, 0.0, 0.0]
    
    cylindre = cylindre_plein(n, R, h)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for step in range(20):
        # Mise à jour physique
        G, vG = translation(m, F, G, vG, h_temps)
        teta, omega = rotation(I_inv, F, G, teta, omega, h_temps)
        
        # Rotation des points
        R = rotation_matrix(teta)
        cylindre_rot = rotate_points(cylindre, R)
        
        # Translation des points
        cylindre_trans = translater_cylindre(cylindre_rot, G)
        
        # Tracé
        x = [p[0] for p in cylindre_trans]
        y = [p[1] for p in cylindre_trans]
        z = [p[2] for p in cylindre_trans]
        ax.scatter(x, y, z, alpha=0.4)
    
    ax.set_title("Rotation du cylindre sous couple")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    plt.show()
def haltere(n, longueur_barre, rayon_barre, rayon_poids, hauteur_poids):

    points = []
    
    n_barre = n // 3
    n_poids = n // 3
    
    step_r_barre = rayon_barre / (n_barre**0.25)
    step_theta = 2 * 3.141592653589793 / (n_barre**0.25)
    step_z_barre = longueur_barre / (n_barre**0.25)
    
    for k in range(int(n_barre**0.25)):
        z = -longueur_barre/2 + k * step_z_barre
        for i in range(int(n_barre**0.25)):
            r = i * step_r_barre
            for j in range(int(n_barre**0.25)):
                theta = j * step_theta
                x = r * cosinus(theta)
                y = r * sinus(theta)
                points.append([x, y, z])
    
    step_r_poids = rayon_poids / (n_poids**0.25)
    step_z_poids = hauteur_poids / (n_poids**0.25)
    
    for k in range(int(n_poids**0.25)):
        z = -longueur_barre/2 - hauteur_poids + k * step_z_poids
        for i in range(int(n_poids**0.25)):
            r = i * step_r_poids
            for j in range(int(n_poids**0.25)):
                theta = j * step_theta
                x = r * cosinus(theta)
                y = r * sinus(theta)
                points.append([x, y, z])
    
    # Création du deuxième poids (cylindre droit)
    for k in range(int(n_poids**0.25)):
        z = longueur_barre/2 + k * step_z_poids
        for i in range(int(n_poids**0.25)):
            r = i * step_r_poids
            for j in range(int(n_poids**0.25)):
                theta = j * step_theta
                x = r * cosinus(theta)
                y = r * sinus(theta)
                points.append([x, y, z])
    
    return transpose(points)

def draw_haltere(n=30000, longueur_barre=6, rayon_barre=0.5, rayon_poids=2, hauteur_poids=1):

    points = haltere(n, longueur_barre, rayon_barre, rayon_poids, hauteur_poids)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y, Z = points[0], points[1], points[2]
    
    ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Haltère 3D')
    
    # Équilibrer les axes pour une meilleure visualisation
    max_range = max([
        max(X) - min(X),
        max(Y) - min(Y),
        max(Z) - min(Z)
    ]) / 2.0
    
    mid_x = (max(X) + min(X)) / 2
    mid_y = (max(Y) + min(Y)) / 2
    mid_z = (max(Z) + min(Z)) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()

def haltere_rotation_translation(n=1000, longueur_barre=6, rayon_barre=0.5, rayon_poids=2, hauteur_poids=1, 
                                 position=[0, 0, 0], angles=[0, 0, 0]):

    # Créer l'haltère centré à l'origine
    points_haltere = haltere(n, longueur_barre, rayon_barre, rayon_poids, hauteur_poids)
    
    # Convertir les points en format liste de listes pour la rotation
    points_list = []
    for i in range(len(points_haltere[0])):
        points_list.append([points_haltere[0][i], points_haltere[1][i], points_haltere[2][i]])
    
    # Appliquer la rotation
    R = rotation_matrix(angles)
    points_rotated = rotate_points(points_list, R)
    
    # Appliquer la translation
    points_final = translater_cylindre(points_rotated, position)
    
    # Convertir en format attendu par plot_3d
    X = [p[0] for p in points_final]
    Y = [p[1] for p in points_final]
    Z = [p[2] for p in points_final]
    
    return [X, Y, Z]

def draw_haltere_with_motion(n=20000, longueur_barre=6, rayon_barre=0.5, rayon_poids=2, hauteur_poids=1):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Définir les limites de l'axe pour une visualisation cohérente
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Haltère en mouvement')
    
    # Simuler différentes positions et rotations
    positions = [
        [0, 0, 0],
        [2, 1, 3],
        [4, -2, 1],
        [-3, 2, -2],
        [-1, -3, 4]
    ]
    
    rotations = [
        [0, 0, 0],
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 0.5],
        [0.3, 0.3, 0.3]
    ]
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i in range(len(positions)):
        points = haltere_rotation_translation(
            n=n, 
            longueur_barre=longueur_barre, 
            rayon_barre=rayon_barre, 
            rayon_poids=rayon_poids, 
            hauteur_poids=hauteur_poids,
            position=positions[i],
            angles=rotations[i]
        )
        
        X, Y, Z = points
        ax.scatter(X, Y, Z, color=colors[i], alpha=0.6, s=5, label=f'Position {i+1}')
    
    ax.legend()
    plt.show()

def animate_haltere(n=1000, longueur_barre=6, rayon_barre=0.5, rayon_poids=2, hauteur_poids=1, 
                   frames=50, save_animation=False):

    import matplotlib.animation as animation
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Définir les limites de l'axe pour une visualisation cohérente
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Animation d\'un haltère')
    
    # Créer un scatter plot vide
    scatter = ax.scatter([], [], [], c='blue', s=5)
    
    def update(frame):
        position = [5 * cosinus(frame/10), 5 * sinus(frame/10), 2 * sinus(frame/15)]
        angles = [frame/20, frame/25, frame/30]
        
        points = haltere_rotation_translation(
            n=n, 
            longueur_barre=longueur_barre, 
            rayon_barre=rayon_barre, 
            rayon_poids=rayon_poids, 
            hauteur_poids=hauteur_poids,
            position=position,
            angles=angles
        )
        
        X, Y, Z = points
        
        scatter._offsets3d = (X, Y, Z)
        
        return scatter,

    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=50, blit=True
    )
    
    if save_animation:
        anim.save('haltere_animation.gif', writer='pillow', fps=60)
    
    plt.show()

animate_haltere(save_animation=True)