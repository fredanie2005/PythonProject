import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import tahaAPI
import TP1 
import TP2

# Helper functions
def cosinus(x, terms=10):
    result = 0
    for k in range(terms):
        result += (-1)**k * (x**(2 * k)) / factorial(2 * k)
    return result

def sinus(x, terms=10):
    result = 0
    for k in range(terms):
        result += (-1)**k * (x**(2 * k + 1)) / factorial(2 * k + 1)
    return result

def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)

def transpose(matrix):
    result = []
    for i in range(len(matrix[0])):
        newMatrix = []
        for j in range(len(matrix)):
            newMatrix.append(matrix[j][i])
        result.append(newMatrix)
    return result

# Question 8: Create a function to generate a dumbbell (haltere)
def solide(n):
    """
    Generate a dumbbell (haltere) with n points.
    Returns a list of [x, y, z] coordinates.
    """
    # Parameters for the haltere
    longueur_barre = 6      # Length of the bar
    rayon_barre = 0.5       # Radius of the bar
    rayon_poids = 2         # Radius of the weights
    hauteur_poids = 1       # Height of the weights
    
    points = []
    
    # Distribute points among components
    n_barre = n // 3        # Points for the bar
    n_poids = n // 3        # Points for each weight
    
    # Calculate step sizes
    step_r_barre = rayon_barre / (n_barre**0.25)
    step_theta = 2 * 3.14159 / (n_barre**0.25)
    step_z_barre = longueur_barre / (n_barre**0.25)
    
    # Create the bar (cylinder)
    for k in range(int(n_barre**0.25)):
        z = -longueur_barre/2 + k * step_z_barre
        for i in range(int(n_barre**0.25)):
            r = i * step_r_barre
            for j in range(int(n_barre**0.25)):
                theta = j * step_theta
                x = r * cosinus(theta)
                y = r * sinus(theta)
                points.append([x, y, z])
    
    # Calculate step sizes for weights
    step_r_poids = rayon_poids / (n_poids**0.25)
    step_z_poids = hauteur_poids / (n_poids**0.25)
    
    # Create the left weight (cylinder)
    for k in range(int(n_poids**0.25)):
        z = -longueur_barre/2 - hauteur_poids + k * step_z_poids
        for i in range(int(n_poids**0.25)):
            r = i * step_r_poids
            for j in range(int(n_poids**0.25)):
                theta = j * step_theta
                x = r * cosinus(theta)
                y = r * sinus(theta)
                points.append([x, y, z])
    
    # Create the right weight (cylinder)
    for k in range(int(n_poids**0.25)):
        z = longueur_barre/2 + k * step_z_poids
        for i in range(int(n_poids**0.25)):
            r = i * step_r_poids
            for j in range(int(n_poids**0.25)):
                theta = j * step_theta
                x = r * cosinus(theta)
                y = r * sinus(theta)
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
    
    # Set equal aspect ratio
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
    """
    Move the inertia matrix I from point G to point A.
    I: inertia matrix at point G
    m: mass
    G: coordinates of point G
    A: coordinates of point A
    """
    # Calculate the displacement vector
    d = [G[i] - A[i] for i in range(3)]
    
    # Apply the parallel axis theorem
    I_A = []
    for i in range(3):
        row = []
        for j in range(3):
            # Kronecker delta
            delta_ij = 1 if i == j else 0
            
            # Calculate the new component
            I_Aij = I[i][j] + m * (sum(d[k]**2 for k in range(3)) * delta_ij - d[i] * d[j])
            row.append(I_Aij)
        I_A.append(row)
    
    return I_A

# Helper functions for movement
def somme_forces(F):
    """Calculate the sum of all forces."""
    total = [0, 0, 0]
    for force_point in F:
        force = force_point[0]
        for i in range(3):
            total[i] += force[i]
    return total

def multiplication_vecteur_scalaire(vecteur, scalaire):
    """Multiply a vector by a scalar."""
    return [composante * scalaire for composante in vecteur]

def addition_vecteurs3(vecteur1, vecteur2):
    """Add two 3D vectors."""
    return [vecteur1[i] + vecteur2[i] for i in range(3)]

def produit_vectoriel(u, v):
    """Calculate the cross product of two vectors."""
    return [
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    ]

def translation(m, F, G, vG, h):
    """
    Calculate translation of an object.
    m: mass
    F: forces
    G: position
    vG: velocity
    h: time step
    """
    # Calculate total force
    total_force = somme_forces(F)

    # Calculate acceleration
    acceleration = multiplication_vecteur_scalaire(total_force, 1 / m)
    
    # Update velocity
    nouvelle_vitesse = addition_vecteurs3(vG, multiplication_vecteur_scalaire(acceleration, h))

    # Update position
    nouvelle_position = addition_vecteurs3(G, multiplication_vecteur_scalaire(vG, h))
    
    return nouvelle_position, nouvelle_vitesse

def rotation_matrix(teta):
    """
    Create a rotation matrix from Euler angles (ZYX).
    teta: [theta_x, theta_y, theta_z]
    """
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
    """
    Apply rotation matrix R to all points.
    points: list of [x, y, z] coordinates
    R: 3x3 rotation matrix
    """
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
    """
    Calculate rotation of an object.
    I_inv: inverse of inertia tensor
    F: forces
    G: center of mass
    teta: orientation
    omega: angular velocity
    h: time step
    """
    # Calculate total torque
    torque = [0.0, 0.0, 0.0]
    for force, point in F:
        r = [point[i] - G[i] for i in range(3)]  # Vector from G to application point
        t = produit_vectoriel(r, force)
        for i in range(3):
            torque[i] += t[i]
    
    # Calculate angular acceleration alpha = I⁻¹ * torque
    alpha = [
        sum(I_inv[i][j] * torque[j] for j in range(3))
        for i in range(3)
    ]
    
    # Update angular velocity
    new_omega = [omega[i] + alpha[i] * h for i in range(3)]
    
    # Update orientation (Euler method)
    new_teta = [teta[i] + new_omega[i] * h for i in range(3)]
    
    return new_teta, new_omega

def translater_cylindre(W, vecteur):
    """
    Translate all points in W by vector.
    W: list of [x, y, z] coordinates
    vecteur: [dx, dy, dz] translation vector
    """
    W_new = []
    for point in W:
        W_new.append([point[i] + vecteur[i] for i in range(3)])
    return W_new

def mouvement(W, m, I, F, G, vG, omega, h, n):
    """
    Simulate movement of an object.
    W: points of the object
    m: mass
    I: inertia tensor
    F: forces
    G: initial position
    vG: initial velocity
    omega: initial angular velocity
    h: time step
    n: number of steps
    """
    positions = []
    
    # Calculate inverse of inertia tensor
    # Note: This is a simplified inversion for a diagonal matrix
    I_inv = [
        [1/I[0][0], 0, 0],
        [0, 1/I[1][1], 0],
        [0, 0, 1/I[2][2]]
    ]
    
    current_G = G.copy()
    current_vG = vG.copy()
    current_teta = [0, 0, 0]
    current_omega = omega.copy()
    
    for i in range(n):
        # Store current position
        R = rotation_matrix(current_teta)
        rotated_points = rotate_points(W, R)
        translated_points = translater_cylindre(rotated_points, current_G)
        positions.append(translated_points)
        
        # Update position and orientation
        current_G, current_vG = translation(m, F, current_G, current_vG, h)
        current_teta, current_omega = rotation(I_inv, F, current_G, current_teta, current_omega, h)
    
    return positions

# Question 11: Translate the solid with forces
def tracer_positions_translation():
    """
    Plot the dumbbell (haltere) at different positions during translation.
    """
    n = 1000
    m = 10.0  # Mass of the haltere
    h_temps = 0.5  # Time step
    
    # Define forces
    F = [
        [[2, 0, 0], [0, 0, 0]],  # Force in X direction
        [[0, 1, 0], [0, 0, 0]],  # Force in Y direction
        [[0, 0, 0.5], [0, 0, 0]]  # Force in Z direction
    ]
    
    G = [0, 0, 0]  # Initial position
    vG = [0, 0, 0]  # Initial velocity
    
    # Create the solid
    haltere_points = solide(n)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 10 positions
    for step in range(10):
        # Update position
        G, vG = translation(m, F, G, vG, h_temps)
        
        # Translate the solid
        translated_points = translater_cylindre(haltere_points, G)
        
        # Extract coordinates
        X = [p[0] for p in translated_points]
        Y = [p[1] for p in translated_points]
        Z = [p[2] for p in translated_points]
        
        # Plot with decreasing opacity for older positions
        opacity = 0.3 + 0.7 * (step / 9)
        ax.scatter(X, Y, Z, alpha=opacity, s=5, label=f'Step {step+1}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Translation of Haltere')
    
    # Set equal aspect ratio
    ax.set_xlim(-10, 30)
    ax.set_ylim(-10, 20)
    ax    # Set equal aspect ratio (continued)
    ax.set_zlim(-10, 10)
    
    plt.legend()
    plt.show()

# Question 12: Rotate the solid around its center of inertia
def tracer_positions_rotation():
    """
    Plot the dumbbell (haltere) at different positions during rotation.
    """
    n = 1000
    m = 10.0  # Mass of the haltere
    h_temps = 0.2  # Time step
    
    # Inertia tensor for the haltere (approximation)
    # For a dumbbell with two masses at the ends of a rod
    I = [
        [m * 8, 0, 0],
        [0, m * 8, 0],
        [0, 0, m * 2]
    ]
    
    # Inverse of inertia tensor
    I_inv = [
        [1/(m * 8), 0, 0],
        [0, 1/(m * 8), 0],
        [0, 0, 1/(m * 2)]
    ]
    
    # Forces creating torque
    F = [
        [[0, 1, 0], [3, 0, 0]],  # Force at right end
        [[0, -1, 0], [-3, 0, 0]]  # Force at left end
    ]
    
    G = [0, 0, 0]  # Center of inertia
    teta = [0, 0, 0]  # Initial orientation
    omega = [0, 0, 0]  # Initial angular velocity
    
    # Create the solid
    haltere_points = solide(n)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 10 positions
    for step in range(10):
        # Update orientation
        teta, omega = rotation(I_inv, F, G, teta, omega, h_temps)
        
        # Rotate the solid
        R = rotation_matrix(teta)
        rotated_points = rotate_points(haltere_points, R)
        
        # Extract coordinates
        X = [p[0] for p in rotated_points]
        Y = [p[1] for p in rotated_points]
        Z = [p[2] for p in rotated_points]
        
        # Plot with decreasing opacity for older positions
        opacity = 0.3 + 0.7 * (step / 9)
        ax.scatter(X, Y, Z, alpha=opacity, s=5, label=f'Step {step+1}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rotation of Haltere')
    
    # Set equal aspect ratio
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    
    plt.legend()
    plt.show()

# Question 13: Combined movement (rotation and translation)
def tracer_positions_mouvement():
    """
    Plot the dumbbell (haltere) at different positions during combined movement.
    """
    n = 1000
    m = 10.0  # Mass of the haltere
    h_temps = 0.2  # Time step
    
    # Inertia tensor for the haltere
    I = [
        [m * 8, 0, 0],
        [0, m * 8, 0],
        [0, 0, m * 2]
    ]
    
    # Forces creating both translation and rotation
    F = [
        [[1, 1, 0], [3, 0, 0]],  # Force at right end
        [[1, -1, 0], [-3, 0, 0]]  # Force at left end
    ]
    
    G = [0, 0, 0]  # Initial position
    vG = [0, 0, 0]  # Initial velocity
    omega = [0, 0, 0]  # Initial angular velocity
    
    # Create the solid
    haltere_points = solide(n)
    
    # Simulate movement
    positions = mouvement(haltere_points, m, I, F, G, vG, omega, h_temps, 10)
    
    # Plot the positions
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for step, pos in enumerate(positions):
        # Extract coordinates
        X = [p[0] for p in pos]
        Y = [p[1] for p in pos]
        Z = [p[2] for p in pos]
        
        # Plot with decreasing opacity for older positions
        opacity = 0.3 + 0.7 * (step / (len(positions) - 1))
        ax.scatter(X, Y, Z, alpha=opacity, s=5, label=f'Step {step+1}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Combined Movement of Haltere')
    
    # Set equal aspect ratio
    ax.set_xlim(-10, 30)
    ax.set_ylim(-10, 30)
    ax.set_zlim(-10, 10)
    
    plt.legend()
    plt.show()

# Additional function to calculate inertia tensor for the haltere
def calculer_inertie_haltere(m_barre, m_poids, longueur_barre, rayon_barre, rayon_poids, hauteur_poids):
    """
    Calculate the inertia tensor for a dumbbell.
    
    Parameters:
    m_barre: mass of the bar
    m_poids: mass of each weight
    longueur_barre: length of the bar
    rayon_barre: radius of the bar
    rayon_poids: radius of the weights
    hauteur_poids: height of the weights
    
    Returns:
    Inertia tensor at the center of mass
    """
    # Inertia tensor for the bar (cylinder along z-axis)
    I_barre = [
        [m_barre * (3*rayon_barre**2 + longueur_barre**2) / 12, 0, 0],
        [0, m_barre * (3*rayon_barre**2 + longueur_barre**2) / 12, 0],
        [0, 0, m_barre * rayon_barre**2 / 2]
    ]
    
    # Inertia tensor for each weight (cylinder along z-axis)
    I_poids_local = [
        [m_poids * (3*rayon_poids**2 + hauteur_poids**2) / 12, 0, 0],
        [0, m_poids * (3*rayon_poids**2 + hauteur_poids**2) / 12, 0],
        [0, 0, m_poids * rayon_poids**2 / 2]
    ]
    
    # Position of the left weight
    G_poids_gauche = [-longueur_barre/2 - hauteur_poids/2, 0, 0]
    
    # Position of the right weight
    G_poids_droit = [longueur_barre/2 + hauteur_poids/2, 0, 0]
    
    # Center of mass of the whole system
    G = [0, 0, 0]  # Assuming symmetry
    
    # Move inertia tensors to the center of mass
    I_poids_gauche = deplace_mat(I_poids_local, m_poids, G_poids_gauche, G)
    I_poids_droit = deplace_mat(I_poids_local, m_poids, G_poids_droit, G)
    
    # Total inertia tensor
    I_total = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(I_barre[i][j] + I_poids_gauche[i][j] + I_poids_droit[i][j])
        I_total.append(row)
    
    return I_total
import matplotlib.animation as animation
from matplotlib.colors import Normalize

def tracer_positions_mouvement_anime(n=1000, steps=50, h_temps=0.1, export_gif=True, filename="haltere_movement.gif"):

    m = 10.0  # Mass of the haltere
    
    # Inertia tensor for the haltere
    I = [
        [m * 8, 0, 0],
        [0, m * 8, 0],
        [0, 0, m * 2]
    ]
    
    # Forces creating both translation and rotation
    F = [
        [[1, 1, 0.5], [3, 0, 0]],   # Force at right end
        [[1, -1, 0], [-3, 0, 0]]    # Force at left end
    ]
    
    G = [0, 0, 0]  # Initial position
    vG = [0, 0, 0]  # Initial velocity
    omega = [0, 0, 0]  # Initial angular velocity
    
    # Create the solid
    haltere_points = solide(n)
    
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits
    ax.set_xlim(-10, 30)
    ax.set_ylim(-10, 30)
    ax.set_zlim(-10, 10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Animated Movement of Haltere')
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], [], c=[], cmap='viridis', s=5)
    
    # Initialize text for time display
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    
    # Precompute all positions for better performance
    all_positions = []
    current_G = G.copy()
    current_vG = vG.copy()
    current_teta = [0, 0, 0]
    current_omega = omega.copy()
    
    # Inverse of inertia tensor
    I_inv = [
        [1/I[0][0], 0, 0],
        [0, 1/I[1][1], 0],
        [0, 0, 1/I[2][2]]
    ]
    
    print("Precomputing positions...")
    for i in range(steps):
        # Update position and orientation
        current_G, current_vG = translation(m, F, current_G, current_vG, h_temps)
        current_teta, current_omega = rotation(I_inv, F, current_G, current_teta, current_omega, h_temps)
        
        # Apply rotation and translation
        R = rotation_matrix(current_teta)
        rotated_points = rotate_points(haltere_points, R)
        translated_points = translater_cylindre(rotated_points, current_G)
        
        all_positions.append(translated_points)
    
    # Normalize Z values for coloring
    all_z_values = [p[2] for pos in all_positions for p in pos]
    z_min, z_max = min(all_z_values), max(all_z_values)
    norm = Normalize(z_min, z_max)
    
    # Animation update function
    def update(frame):
        # Get the precomputed position for this frame
        pos = all_positions[frame]
        
        # Extract coordinates
        X = [p[0] for p in pos]
        Y = [p[1] for p in pos]
        Z = [p[2] for p in pos]
        
        # Update scatter plot
        scatter._offsets3d = (X, Y, Z)
        import numpy as np
        scatter.set_array(np.array(Z))
        
        # Update time text
        time_text.set_text(f'Time: {frame * h_temps:.1f}s')
        
        return scatter, time_text
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(
        fig, update, frames=steps, interval=50, blit=True
    )
    
    # Export as GIF if requested
    if export_gif:
        print(f"Exporting animation to {filename}...")
        anim.save(filename, writer='pillow', fps=20)
        print(f"Animation saved to {filename}")
    
    plt.show()
    return anim

# Main function to run all demonstrations
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

