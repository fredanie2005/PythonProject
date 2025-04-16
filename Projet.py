import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import tahaAPI
import TP1 
import TP2
import TP3

def solide(n):
    longueur_barre = 6   
    rayon_barre = 0.5    
    rayon_poids = 2     
    hauteur_poids = 1   
    
    points = []
    
    n_barre = n // 3     
    n_poids = n // 3      
    
    step_r_barre = rayon_barre / (n_barre**0.25)
    step_theta = 2 * tahaAPI.pi() / (n_barre**0.25)
    step_z_barre = longueur_barre / (n_barre**0.25)
    
    # Barre
    for k in range(int(n_barre**0.25)):
        z = -longueur_barre/2 + k * step_z_barre
        for i in range(int(n_barre**0.25)):
            r = i * step_r_barre
            for j in range(int(n_barre**0.25)):
                theta = j * step_theta
                x = r * tahaAPI.cosinus(theta)
                y = r * tahaAPI.sinus(theta)
                points.append([x, y, z])
    
    step_r_poids = rayon_poids / (n_poids**0.25)
    step_z_poids = hauteur_poids / (n_poids**0.25)
    
    # Poids Gauche
    for k in range(int(n_poids**0.25)):
        z = -longueur_barre/2 - hauteur_poids + k * step_z_poids
        for i in range(int(n_poids**0.25)):
            r = i * step_r_poids
            for j in range(int(n_poids**0.25)):
                theta = j * step_theta
                x = r * tahaAPI.cosinus(theta)
                y = r * tahaAPI.sinus(theta)
                points.append([x, y, z])
    
    # Poids Droit
    for k in range(int(n_poids**0.25)):
        z = longueur_barre/2 + k * step_z_poids
        for i in range(int(n_poids**0.25)):
            r = i * step_r_poids
            for j in range(int(n_poids**0.25)):
                theta = j * step_theta
                x = r * tahaAPI.cosinus(theta)
                y = r * tahaAPI.sinus(theta)
                points.append([x, y, z])
    
    return points

# Question 9: Plot the solid in 3D
def tracer_solide():
    """
    Plot the dumbbell (haltere) in 3D.
    """
    points = solide(1000)
    X = [p[0] for p in points]
    Y = [p[1] for p in points]
    Z = [p[2] for p in points]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Haltère 3D')
    
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

# Question 10: Function to move the inertia matrix
def deplace_mat(I, m, G, A):

    d = [G[i] - A[i] for i in range(3)]
    
    I_A = []
    for i in range(3):
        row = []
        for j in range(3):
            delta_ij = 1 if i == j else 0
            
            I_Aij = I[i][j] + m * (sum(d[k]**2 for k in range(3)) * delta_ij - d[i] * d[j])
            row.append(I_Aij)
        I_A.append(row)
    
    return I_A

def rotation_matrix(teta):

    cx = tahaAPI.cosinus(teta[0])
    sx = tahaAPI.sinus(teta[0])
    cy = tahaAPI.cosinus(teta[1])
    sy = tahaAPI.sinus(teta[1])
    cz = tahaAPI.cosinus(teta[2])
    sz = tahaAPI.sinus(teta[2])
    
    return [
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy, cy*sx, cy*cx]
    ]

def rotate_points(points, R):

    rotated = []
    for p in points:
        new_p = [
            R[0][0]*p[0] + R[0][1]*p[1] + R[0][2]*p[2],
            R[1][0]*p[0] + R[1][1]*p[1] + R[1][2]*p[2],
            R[2][0]*p[0] + R[2][1]*p[1] + R[2][2]*p[2]
        ]
        rotated.append(new_p)
    return rotated

def rotation(I_inv, F, G, teta, omega, h):

    torque = [0.0, 0.0, 0.0]
    for force, point in F:
        r = [point[i] - G[i] for i in range(3)] 
        t = tahaAPI.produit_vectoriel(r, force)
        for i in range(3):
            torque[i] += t[i]
    
    alpha = [
        sum(I_inv[i][j] * torque[j] for j in range(3))
        for i in range(3)
    ]
    
    new_omega = [omega[i] + alpha[i] * h for i in range(3)]
    
    new_teta = [teta[i] + new_omega[i] * h for i in range(3)]
    
    return new_teta, new_omega

def translater_cylindre(W, vecteur):
    W_new = []
    for point in W:
        W_new.append([point[i] + vecteur[i] for i in range(3)])
    return W_new

def mouvement(W, m, I, F, G, vG, omega, h, n):
    positions = []
    
    I_inv = TP2.inverse(I)
    
    current_G = G.copy()
    current_vG = vG.copy()
    current_teta = [0, 0, 0]
    current_omega = omega.copy()
    
    for i in range(n):
        R = rotation_matrix(current_teta)
        rotated_points = rotate_points(W, R)
        translated_points = translater_cylindre(rotated_points, current_G)
        positions.append(translated_points)
        
        current_G, current_vG = tahaAPI.translation(m, F, current_G, current_vG, h)
        current_teta, current_omega = rotation(I_inv, F, current_G, current_teta, current_omega, h)
    
    return positions

# Question 11: Translate the solid with forces
def tracer_positions_translation():
    n = 1000
    m = 10.0  # Mass of the haltere
    h_temps = 0.5  # Time step
    
    F = [
        [[0, 0, 0], [0, 0, 0]],  # Force in X direction
        [[0, 10, 0], [0, 0, 0]],  # Force in Y direction
        [[0, 0, 0], [0, 0, 0]]  # Force in Z direction
    ]
    
    G = [0, 0, 0]  # Initial position
    vG = [0, 0, 0]  # Initial velocity
    
    haltere_points = solide(n)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for step in range(10):
        G, vG = tahaAPI.translation(m, F, G, vG, h_temps)

        translated_points = translater_cylindre(haltere_points, G)
        
        X = [p[0] for p in translated_points]
        Y = [p[1] for p in translated_points]
        Z = [p[2] for p in translated_points]

        opacity = 0.3 + 0.7 * (step / 9)
        ax.scatter(X, Y, Z, alpha=opacity, s=5, label=f'Step {step+1}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Translation of Haltere')
    
    ax.set_xlim(-10, 30)
    ax.set_ylim(-10, 20)
    ax.set_zlim(-10, 10)
    
    plt.legend()
    plt.show()

def tracer_positions_rotation():
    n = 1000
    m = 10.0 
    h_temps = 0.2  

    I = [
        [m * 8, 0, 0],
        [0, m * 8, 0],
        [0, 0, m * 2]
    ]
    
    I_inv = TP2.inverse(I)
    
    F = [
        [[0, 1, 0], [3, 0, 0]], 
        [[0, -1, 0], [-3, 0, 0]] 
    ]
    
    G = [0, 0, 0] 
    teta = [0, 0, 0]  
    omega = [0, 0, 0]  
    
    haltere_points = solide(n)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for step in range(10):
        teta, omega = rotation(I_inv, F, G, teta, omega, h_temps)
        
        R = rotation_matrix(teta)
        rotated_points = rotate_points(haltere_points, R)
        
        X = [p[0] for p in rotated_points]
        Y = [p[1] for p in rotated_points]
        Z = [p[2] for p in rotated_points]
        
        opacity = 0.3 + 0.7 * (step / 9)
        ax.scatter(X, Y, Z, alpha=opacity, s=5, label=f'Step {step+1}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rotation of Haltere')
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    
    plt.legend()
    plt.show()

def tracer_positions_mouvement():
    n = 1000
    m = 10.0 
    h_temps = 0.2  
    
    I = [
        [m * 8, 0, 0],
        [0, m * 8, 0],
        [0, 0, m * 2]
    ]
    
    F = [
        [[1, 1, 0], [3, 0, 0]], 
        [[1, -1, 0], [-3, 0, 0]]
    ]
    
    G = [0, 0, 0] 
    vG = [0, 0, 0] 
    omega = [0, 0, 0]  
    
    haltere_points = solide(n)
    
    positions = mouvement(haltere_points, m, I, F, G, vG, omega, h_temps, 10)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for step, pos in enumerate(positions):
        X = [p[0] for p in pos]
        Y = [p[1] for p in pos]
        Z = [p[2] for p in pos]
        
        opacity = 0.3 + 0.7 * (step / (len(positions) - 1))
        ax.scatter(X, Y, Z, alpha=opacity, s=5, label=f'Step {step+1}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Combined Movement of Haltere')
    
    ax.set_xlim(-10, 30)
    ax.set_ylim(-10, 30)
    ax.set_zlim(-10, 10)
    
    plt.legend()
    plt.show()

def calculer_inertie_haltere(m_barre, m_poids, longueur_barre, rayon_barre, rayon_poids, hauteur_poids):

    I_barre = [
        [m_barre * (3*rayon_barre**2 + longueur_barre**2) / 12, 0, 0],
        [0, m_barre * (3*rayon_barre**2 + longueur_barre**2) / 12, 0],
        [0, 0, m_barre * rayon_barre**2 / 2]
    ]
    
    I_poids_local = [
        [m_poids * (3*rayon_poids**2 + hauteur_poids**2) / 12, 0, 0],
        [0, m_poids * (3*rayon_poids**2 + hauteur_poids**2) / 12, 0],
        [0, 0, m_poids * rayon_poids**2 / 2]
    ]
    
    G_poids_gauche = [-longueur_barre/2 - hauteur_poids/2, 0, 0]

    G_poids_droit = [longueur_barre/2 + hauteur_poids/2, 0, 0]
    
    G = [0, 0, 0]  
    
    I_poids_gauche = deplace_mat(I_poids_local, m_poids, G_poids_gauche, G)
    I_poids_droit = deplace_mat(I_poids_local, m_poids, G_poids_droit, G)
    
    I_total = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(I_barre[i][j] + I_poids_gauche[i][j] + I_poids_droit[i][j])
        I_total.append(row)
    
    return I_total

def tracer_positions_mouvement_anime(n=1000, steps=50, h_temps=0.1, export_gif=True, filename="haltere_movement.gif"):
    import numpy as np
    import matplotlib.animation as animation
    from matplotlib.colors import Normalize
    m = 10.0  
    I = [
        [m * 8, 0, 0],
        [0, m * 8, 0],
        [0, 0, m * 2]
    ]

    F = [
        [[1, 1, 0.5], [3, 0, 0]], 
        [[1, -1, 0], [-3, 0, 0]]   
    ]
    
    G = [0, 0, 0] 
    vG = [0, 0, 0]  
    omega = [0, 0, 0]  
    
    haltere_points = solide(n)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(-10, 30)
    ax.set_ylim(-10, 30)
    ax.set_zlim(-10, 10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Animated Movement of Haltere')
    
    scatter = ax.scatter([], [], [], c=[], cmap='viridis', s=5)

    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    
    all_positions = []
    current_G = G.copy()
    current_vG = vG.copy()
    current_teta = [0, 0, 0]
    current_omega = omega.copy()
    
    I_inv = [
        [1/I[0][0], 0, 0],
        [0, 1/I[1][1], 0],
        [0, 0, 1/I[2][2]]
    ]
    
    print("Precomputing positions...")
    for i in range(steps):
        current_G, current_vG = tahaAPI.translation(m, F, current_G, current_vG, h_temps)
        current_teta, current_omega = rotation(I_inv, F, current_G, current_teta, current_omega, h_temps)
        
        R = rotation_matrix(current_teta)
        rotated_points = rotate_points(haltere_points, R)
        translated_points = translater_cylindre(rotated_points, current_G)
        
        all_positions.append(translated_points)
    
    all_z_values = [p[2] for pos in all_positions for p in pos]
    z_min, z_max = min(all_z_values), max(all_z_values)
    norm = Normalize(z_min, z_max)
    
    def update(frame):
        pos = all_positions[frame]
        
        X = [p[0] for p in pos]
        Y = [p[1] for p in pos]
        Z = [p[2] for p in pos]
        
        scatter._offsets3d = (X, Y, Z)

        scatter.set_array(np.array(Z))
        
        time_text.set_text(f'Time: {frame * h_temps:.1f}s')
        
        return scatter, time_text

    print("Creating animation...")
    anim = animation.FuncAnimation(
        fig, update, frames=steps, interval=50, blit=True
    )
    
    if export_gif:
        print(f"Exporting animation to {filename}...")
        anim.save(filename, writer='pillow', fps=20)
        print(f"Animation saved to {filename}")
    
    plt.show()
    return anim

def main():
    print("Démonstration du projet - Haltère")
    
    # Question 9: Plot the solid
    print("\nQuestion 9: Tracer le solide")
    tracer_solide()
    
    # Question 11: Translation
    print("\nQuestion 11: Translation du solide")
    tracer_positions_translation()
    
    # Question 12: Rotation
    print("\nQuestion 12: Rotation du solide")
    tracer_positions_rotation()
    
    # Question 13: Combined movement
    print("\nQuestion 13: Mouvement combiné du solide")
    tracer_positions_mouvement()
    
    # Animated version with GIF export
    print("\nQuestion 13 (Animé): Mouvement combiné du solide avec animation")
    tracer_positions_mouvement_anime(n=1000, steps=50, h_temps=0.1, export_gif=True)

if __name__ == "__main__":
    main()

