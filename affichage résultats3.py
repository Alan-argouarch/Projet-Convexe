############################### AFFICHAGE DES RESULTATS DU PROJET ########################################

"""
Pour afficher les résultats, simplement copier coller le code dans le fichier python du projet dans la partie 
if __name__ == "__main__": puis lancer le code
Ne pas oublier de bien importer les librairies 
"""
########################################################################################################

# Librairies
import numpy as np 
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

##################### Partie 1.1.1



##################### Partie 1.1.2
vecteur_A = input("Entrez les éléments du vecteur A de dimension n+1, séparés par des espaces : ")
 A = np.array([float(x) for x in vecteur_A.split()])

vecteur_X = input("Entrez les éléments du vecteur X de dimension n, séparés par des espaces : ")
X = np.array([float(x) for x in vecteur_X.split()])
    
projection = proj_demi_espace(X, A)
print("La projection de X sur F est :", projection)

##################### Partie 1.1.3

# Contraintes
A1 = np.array([3, 2, 5])  # 3x + 2y <= 5
A2 = np.array([-2, 3, 1])  # -2x + 3y <= 1 (équivalent à 2x - 3y >= -1)

# Point initial
X = np.array([10, 10])

# Projections successives
X_proj = proj_demi_espace(X, A1)
print(X_proj)
X_proj = proj_demi_espace(X_proj, A2)

print("Projection de X sur P :", X_proj)

# Définir les frontières de l'ensemble P
x_vals = np.linspace(-2, 15, 400)
y_vals_1 = (5 - 3 * x_vals) / 2  # Frontière pour 3x + 2y = 5
y_vals_2 = (2 * x_vals + 1) / 3  # Frontière pour 2x - 3y = -1

# Créer le graphique
plt.figure(figsize=(10, 10))

# Zone de P (intersection des demi-espaces)
x = np.linspace(-2, 15, 400)
y = np.linspace(-2, 15, 400)
X_mesh, Y_mesh = np.meshgrid(x, y)
Z1 = 3 * X_mesh + 2 * Y_mesh <= 5
Z2 = 2 * X_mesh - 3 * Y_mesh >= -1
P_region = Z1 & Z2
plt.contourf(X_mesh, Y_mesh, P_region, levels=1, colors=["#d6e9f9"], alpha=0.5)

# Tracer les frontières des demi-espaces
plt.plot(x_vals, y_vals_1, 'r-', label="3x + 2y = 5")
plt.plot(x_vals, y_vals_2, 'b-', label="2x - 3y = -1")

# Tracer les points X et X_proj
plt.plot(X[0], X[1], 'go', label="X")
plt.plot(X_proj[0], X_proj[1], 'mo', label="Px")

# Tracer le vecteur X - X_proj
plt.quiver(X[0], X[1], X_proj[0] - X[0], X_proj[1] - X[1], 
           angles="xy", scale_units="xy", scale=1, color="black", linewidth=1.5, label="X - Px")

# Ajuster les limites
plt.xlim(-2, 15)
plt.ylim(-2, 15)

# Ajouter les labels et la légende
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
plt.axvline(0, color='black', linewidth=0.5, linestyle="--")
plt.legend()
plt.title("Projection de X=(10,10) sur P")
plt.grid()

# Afficher le graphique
plt.show()
##################### Partie 1.1.4

### Projection sur une boule en 2D
    X=np.array([4,6])
    X0=np.array([-0.5,2])
    r=0.5
    Y=X-X0
    projection = proj_boule(X,X0,r)
    print("La projection de X sur B est :", projection)

    # Création de la figure pour afficher les résultats
    fig, ax = plt.subplots()

    # Tracer la boule (cercle de rayon r autour de X0)
    cercle = plt.Circle(X0, r, color='blue', fill=False, linestyle='--')
    ax.add_artist(cercle)

    # Tracer le vecteur X
    ax.quiver(0, 0, X[0], X[1], angles='xy', scale_units='xy', scale=1, color='red', label="Vecteur X")

    # Tracer la projection du vecteur X sur la boule
    ax.quiver(0, 0, projection[0], projection[1], angles='xy', scale_units='xy', scale=1, color='green', label="Projection de X")

    # Tracer la projection du vecteur X sur la boule
    ax.quiver(X0[0], X0[1], Y[0], Y[1], angles='xy', scale_units='xy', scale=1, color='blue', label="X-X0")

    # Configuration du graphique
    ax.set_xlim(-1.5, 9)
    ax.set_ylim(-1.5, 9)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    # Afficher le graphique
    plt.grid(True)
    plt.title("Affichage 2D de la projection de X sur une boule de centre X0 et de rayon r=0.5")
    plt.show()




### Projection sur une boule en 3D

# Calcul de la projection de X sur la boule

    X = np.array([4,4,4])
    X0 = np.array([1, 2, 3])
    Y=X-X0
    r = 1
    projection = proj_boule(X, X0, r)
    print("La projection de X sur B est :", projection)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Tracer la boule (cercle de rayon r autour de X0)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    X_sphere = X0[0] + r * np.outer(np.cos(u), np.sin(v))
    Y_sphere = X0[1] + r * np.outer(np.sin(u), np.sin(v))
    Z_sphere = X0[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(X_sphere, Y_sphere, Z_sphere, color='b', alpha=0.3)
    # Tracer le vecteur X avec une flèche de type "cone"
    ax.quiver(0, 0, 0, X[0], X[1], X[2], color='r', label="Vecteur X", arrow_length_ratio=0.1, linewidth=2, edgecolor='red')

    # Tracer la projection de X sur la boule avec une flèche de type "cone"
    ax.quiver(0, 0, 0, projection[0], projection[1], projection[2], color='g', label="Projection de X", arrow_length_ratio=0.1, linewidth=2, edgecolor='green')

    # Tracer X-X0 sur la boule avec une flèche de type "cone"
    ax.quiver(X0[0], X0[1], X0[2], Y[0], Y[1], Y[2], color='b', label="X-X0", arrow_length_ratio=0.1, linewidth=2, edgecolor='blue')

    # Configuration du graphique
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_zlim([0, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Afficher le graphique
    plt.title("Affichage 3D de la projection du vecteur X sur la boule B de centre X0 et de rayon r")
    plt.show()
############################# Partie 1.2
############ Projection sur une ellipse :
    x0, y0 = 1,0.5  # Centre de l'ellipse
    a, b =3, 5  # Demi-grand axe (a) et demi-petit axe (b)

    # Vecteur X à projeter et point du centre de l'ellipse
    X = np.array([5, 5])

    # Critères de convergence
    max_iterations = 100  # Nombre maximum d'itérations

    # Initialisation
    t = 0  # Paramètre initial
    tmin = t  # Meilleure valeur de t trouvée
    d_init = float('inf') # Distance très grande 
    iteration = 0
    h = 0.01

    # Boucle d'optimisation

    while iteration < max_iterations:
        t += h  # Incrément de t
        d = distance_ellipse(t)
        # Si la distance calculé est inférieur à la distance calculée précédemment, on enregistre la valeur
        if d < d_init:
            d_init = d
            tmin = t

        iteration += 1

    # Résultat
    print("La valeur de t minimisant la distance est:", tmin)
    print("Distance minimale:", d_init)
    print("Nombre d'itérations:", iteration)
    
    ### On calcul ensuite par la suite le projeté à partir de tmin et de la paramétrisation de l'ellipse
    Pc=point_ellipse(tmin)
    print("le projeté de X sur C est", Pc)
    # On convertit en valeurs numériques
    Pc[0]= float(Pc[0])
    Pc[1] = float(Pc[1])


    ### Verification des résultats :
    # Vecteur tangent normalisé
    vecteur_tangent = tangente_ellipse(tmin)
    vecteur_tangent=np.array([float(vecteur_tangent[0]),float(vecteur_tangent[1])])
    # Produit scalaire entre le projeté et le vecteur tangent :
    X1=Pc[0] - X[0]
    X2=Pc[1] - X[1]
    Vecteur_projection=np.array([X1,X2])
    produit_scalaire = np.dot(vecteur_tangent, Vecteur_projection)
    print("le produit scalaire entre le vecteur tangent et le vecteur projeté est : ", produit_scalaire)



    # On générèe les points de la courbe
    t = np.linspace(0, 2 * np.pi, 500)  # Paramètre t de 0 à 2π
    x = x0 + a * np.cos(t)              # Coordonnée x
    y = y0 + b * np.sin(t)              # Coordonnée y

    # On trace l'ellipse
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label="Ellipse paramétrée")
    plt.scatter(x0, y0, color="red", label="Centre (x0, y0)")  # Centre de l'ellipse
    plt.title(f"Projection de X={X} sur l'ellipse centré en ({x0},{y0}), a={a}, b={b} : {iteration} itérations, pas = {h}, ⟨Vecteur projection│Vecteur tangent⟩={produit_scalaire}")

    # Affichage du vecteur X
    plt.quiver(0, 0, X[0], X[1], angles='xy', scale_units='xy', scale=1, color="green", linewidth=0.1, label="Vecteur X")

    # Affichage du vecteur Pc
    plt.quiver(0, 0, Pc[0] , Pc[1] , angles='xy', scale_units='xy', scale=1, color="blue", linewidth=0.1, label="Vecteur Projeté sur l'ellipse")

   # Tracé du vecteur projection
    plt.quiver(Pc[0], Pc[1], X[0] - Pc[0], X[1] - Pc[1], angles='xy', scale_units='xy', scale=1, color='red', label="Vecteur projection")

    # Affichage du vecteur tangent
    plt.quiver(Pc[0], Pc[1], vecteur_tangent[0], vecteur_tangent[1]  , angles='xy', scale_units='xy', scale=1, color="purple", linewidth=0.1, label="Vecteur tangent")


    # Personnalisation du tracé
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axis("equal")  # Échelle égale pour x et y
    plt.legend()
    plt.grid(True)

    # Augmenter l'échelle du graphique
    plt.xlim(min(Pc[0], X[0]) - 2, max(Pc[0], X[0]) + 2)  # Étendre les limites sur l'axe X
    plt.ylim(min(Pc[1], X[1]) - 2, max(Pc[1], X[1]) + 2)  # Étendre les limites sur l'axe Y

    # Fixer le rapport d'aspect pour que les axes soient uniformes
    plt.gca().set_aspect('equal', adjustable='box')
    # Affichage
    plt.show()

######### Projection sur une ellipse qqc

x0, y0 = 1,0.5  # Centre de l'ellipse
a, b =0.5, 1  # Demi-grand axe (a) et demi-petit axe (b)
teta=np.pi/7
# Vecteur X à projeter et point du centre de l'ellipse
X = np.array([2, 2])

    # Critères de convergence
max_iterations = 10000  # Nombre maximum d'itérations

    # Initialisation
t = 0  # Paramètre initial
tmin = t  # Meilleure valeur de t trouvée
d_init = float('inf') # Distance très grande 
iteration = 0
h = 0.0001

# Boucle d'optimisation

while iteration < max_iterations:
    t += h  # Incrément de t
    d = distance_ellipse_qqc(t)
        # Si la distance calculé est inférieur à la distance calculée précédemment, on enregistre la valeur
    if d < d_init:
        d_init = d
        tmin = t

    iteration += 1

    # Résultat
print("La valeur de t minimisant la distance est:", tmin)
print("Distance minimale:", d_init)
print("Nombre d'itérations:", iteration)
    
### On calcul ensuite par la suite le projeté à partir de tmin et de la paramétrisation de l'ellipse
Pc=point_ellipse_qqc(tmin)
print("le projeté de X sur C est", Pc)


    ### Verification des résultats :
    # Vecteur tangent normalisé
vecteur_tangent = tangente_ellipse_qqc(tmin)
vecteur_tangent=np.array([float(vecteur_tangent[0]),float(vecteur_tangent[1])])
    # Produit scalaire entre le projeté et le vecteur tangent :
X1=Pc[0] - X[0]
X2=Pc[1] - X[1]
Vecteur_projection=np.array([X1,X2])
produit_scalaire = np.dot(vecteur_tangent, Vecteur_projection)
print("le produit scalaire entre le vecteur tangent et le vecteur projeté est : ", produit_scalaire)


##############Projection sur une parabole 
# Test pour voir si X appartient au convexe
resultat=ineq_parabole(X)
if resultat="X appartient à C":
   print("X appartient à C")
   sys.exit()

# Initialisation
X = np.array([4, 3])
t = 0  # Paramètre initial
tmin = t  # Meilleure valeur de t trouvée
d_init = float('inf')  # Distance initiale (très grande)
iteration = 0
max_iterations = 1000  # Nombre maximal d'itérations
h = 0.01  # Pas initial
tolerance = 1e-8  # Critère de convergence sur la distance

while iteration < max_iterations:
    t += h  # Incrément de t
    d = distance_parabole1(t)
    
    # Si une meilleure distance est trouvée, on met à jour
    if d < d_init:
        d_init = d
        tmin = t
        iteration += 1
    else:
        # Si la distance n'améliore plus, on réduit le pas pour raffiner
        h =h/2
    # Critère d'arrêt si le pas devient trop petit
    if abs(h) < tolerance:
        break
    

# Résultat
print("La valeur de t minimisant la distance est:", tmin)
print("Distance minimale:", d_init)
print("Nombre d'itérations:", iteration)
    
### On calcul ensuite par la suite le projeté à partir de tmin et de la paramétrisation de l'ellipse
Pc=point_parabole1(tmin)
print("le projeté de X sur C est", Pc)

### Verification des résultats :
# Vecteur tangent normalisé
vecteur_tangent = tangente_parabole1(tmin)
# Produit scalaire entre le projeté et le vecteur tangent :
X1=Pc[0] - X[0]
X2=Pc[1] - X[1]
Vecteur_projection=np.array([X1,X2])
produit_scalaire = np.dot(vecteur_tangent, Vecteur_projection)
print("le produit scalaire entre le vecteur tangent et le vecteur projeté est : ", produit_scalaire)


# Affichage graphique
t_values = np.linspace(-5, 5, 500)
parabole_x = t_values
parabole_y = t_values**2

plt.figure(figsize=(8, 8))

# Tracé de la parabole
plt.plot(parabole_x, parabole_y, label="Parabole $y = x^2$", color="blue")

# Tracé du point à projeter
plt.scatter([X[0]], [X[1]], color="red", label="Point $X$", zorder=5)

# Tracé du projeté
plt.scatter([Pc[0]], [Pc[1]], color="green", label="Projeté $P_c$", zorder=5)

# Tracé du vecteur tangent au point projeté
plt.quiver(
    Pc[0], Pc[1],
    vecteur_tangent[0], vecteur_tangent[1],
    angles='xy', scale_units='xy', scale=0.5, color="orange", label="Vecteur tangent"
)

# Ligne entre X et son projeté
plt.plot([X[0], Pc[0]], [X[1], Pc[1]], "k--", label="Distance minimale")

# Annotations
plt.annotate("X", (X[0], X[1]), textcoords="offset points", xytext=(-10, -10), ha='center', color="red")
plt.annotate("$P_c$", (Pc[0], Pc[1]), textcoords="offset points", xytext=(-10, 10), ha='center', color="green")

# Configuration du graphique
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.title(
    f"Projection de X={X} sur le convexe de bord y=x²\n"
    f"{iteration} itérations, pas = {h}, ⟨Vecteur projection│Vecteur tangent⟩ = {produit_scalaire:.4f}"
)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis("equal")

# Limiter les axes à la plage souhaitée
plt.xlim(-6, 6)
plt.ylim(0, 12)
plt.show()


############ Projection sur une demi hyperbole
X = np.array([0.12, 0.3])

# On vérifie si X appartient au convexe
resultat=ineq_demi_hyperbole(X)
if resultat=="X appartient à C":
   print(resultat)
   sys.exit()

# Initialisation

t = 0  # Paramètre initial
tmin = t  # Meilleure valeur de t trouvée
d_init = float('inf')  # Distance initiale (très grande)
iteration = 0
max_iterations = 10000  # Nombre maximal d'itérations
h = 0.001  # Pas initial
tolerance = 1e-10  # Critère de convergence sur la distance

while iteration < max_iterations:
    t += h  # Incrément de t
    d = distance_demi_hyperbole(t)
    
    # Si une meilleure distance est trouvée, on met à jour
    if d < d_init:
        d_init = d
        tmin = t
        iteration += 1
    else:
        # Si la distance n'améliore plus, on réduit le pas pour raffiner
        h =h/2
    # Critère d'arrêt si le pas devient trop petit
    if abs(h) < tolerance:
        break
    

# Résultat
print("La valeur de t minimisant la distance est:", tmin)
print("Distance minimale:", d_init)
print("Nombre d'itérations:", iteration)
    
### On calcul ensuite par la suite le projeté à partir de tmin et de la paramétrisation de l'ellipse
Pc=point_demi_hyperbole(tmin)
print("le projeté de X sur C est", Pc)

# Vecteur tangent normalisé
vecteur_tangent = tangente_demi_hyperbole(tmin)
# Produit scalaire entre le projeté et le vecteur tangent :
X1=Pc[0] - X[0]
X2=Pc[1] - X[1]
Vecteur_projection=np.array([X1,X2])
produit_scalaire = np.dot(vecteur_tangent, Vecteur_projection)
print("le produit scalaire entre le vecteur tangent et le vecteur projeté est : ", produit_scalaire)


# Affichage graphique
t_values = np.linspace(0.001, 10, 500)
parabole_x = t_values
parabole_y = 1/t_values

plt.figure(figsize=(8, 8))

# Tracé de la parabole
plt.plot(parabole_x, parabole_y, label="demi hyperbole y = x^2$", color="blue")

# Tracé du point à projeter
plt.scatter([X[0]], [X[1]], color="red", label="Point $X$", zorder=5)

# Tracé du projeté
plt.scatter([Pc[0]], [Pc[1]], color="green", label="Projeté $P_c$", zorder=5)

# Tracé du vecteur tangent au point projeté
plt.quiver(
    Pc[0], Pc[1],
    vecteur_tangent[0], vecteur_tangent[1],
    angles='xy', scale_units='xy', scale=0.5, color="orange", label="Vecteur tangent"
)


# Ligne entre X et son projeté
plt.plot([X[0], Pc[0]], [X[1], Pc[1]], "k--", label="Distance minimale")

# Annotations
plt.annotate("X", (X[0], X[1]), textcoords="offset points", xytext=(-10, -10), ha='center', color="red")
plt.annotate("$P_c$", (Pc[0], Pc[1]), textcoords="offset points", xytext=(-10, 10), ha='center', color="green")

# Configuration du graphique
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend()
plt.title(
    f"Projection de X={X} sur la demi hyperbole 0<1/x<=y \n"
    f"{iteration} itérations, pas = {h}, ⟨Vecteur projection│Vecteur tangent⟩ = {produit_scalaire:.4f} "
)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis("equal")

# Limiter les axes à la plage souhaitée
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.show()
################# Projection sur un ellipsoide

 # Paramètres de l'ellipsoïde
    a, b, c = 1, 1, 1  # Demi-axes
    x0, y0, z0 = 0, 0, 0  # Centre de l'ellipsoïde

    X = np.array([1,1,1])  # Point à projeter
    somme = ineq_cartesienne_ellipsoide(X)
    if somme<=1 :
        print("X appartient au convexe, pas de projection (sortie du programme)")
        sys.exit()

    # Critères de convergence 
    max_iterations=100000
    tolerance = 0.000000001
    alpha = 0.01

    (u,v) = projection_ellipsoide(X)
    Pc = np.round(point_ellipse_3D(u,v),4)
    print("Projection de X sur l'ellipsoïde :", Pc)

    (t1,t2)= plan_tangent_ellipsoide(u,v)
    print("le plan tangent à l'ellipsoide est :", (t1,t2))
    x_proj = X[0] - Pc[0]
    y_proj = X[1] - Pc[1]
    z_proj = X[2] - Pc[2]
    Vect_proj=np.array([x_proj,y_proj,z_proj])

    # Verification de l'orthogonalité : 
    Produit_scalaire1=np.dot(t1,(Vect_proj/np.linalg.norm(Vect_proj)))
    Produit_scalaire2=np.dot(t2,(Vect_proj/np.linalg.norm(Vect_proj)))
    print("Le produit scalaire entre X-Pc et la première direction du plan tangent est :", Produit_scalaire1)
    print("le produit scalaire entre X-Pc et la deuxième direction du plan tangent est : ", Produit_scalaire2)
    # Création d'une grille paramétrique pour l'ellipsoïde
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(-np.pi / 2, np.pi / 2, 100)
    U, V = np.meshgrid(u, v)

    # Coordonnées de l'ellipsoïde
    x = x0 + a * np.cos(U) * np.cos(V)
    y = y0 + b * np.sin(U) * np.cos(V)
    z = z0 + c * np.sin(V)

    # Configuration de la figure 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
 
    # Tracé de l'ellipsoïde
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.6, edgecolor='none')

    # Tracé du vecteur X
    ax.quiver(0, 0, 0, X[0], X[1], X[2], color='red', label='Vecteur X', linewidth=1, arrow_length_ratio=0.2)

    # Tracé du vecteur Pc
    ax.quiver(0, 0, 0, Pc[0], Pc[1], Pc[2], color='green', label='Vecteur Pc', linewidth=2, arrow_length_ratio=0.2)

    # Tracé de X-Pc

    ax.quiver(Pc[0], Pc[1], Pc[2], Vect_proj[0], Vect_proj[1], Vect_proj[2], color='blue', label='x-Pc', linewidth=2, arrow_length_ratio=0.2)
    print(Vect_proj)
    # Points sur X et Pc
    ax.scatter(*X, color='red', label='Point X', s=50)
    ax.scatter(*Pc, color='green', label='Point Pc', s=50)
    
    # Tracé du plan tangent :
    ax.quiver(Pc[0], Pc[1], Pc[2], t1[0], t1[1], t1[2], color='purple', label='Direction 1 du plan tangent en Pc : t1', linewidth=2, arrow_length_ratio=0.2)
    print(Vect_proj)
    ax.quiver(Pc[0], Pc[1], Pc[2], t2[0], t2[1], t2[2], color='purple', label='Direction 2 du plan tangent en Pc : t2', linewidth=2, arrow_length_ratio=0.2)
    print(Vect_proj)

    
    # Ajustement des axes pour un espace 3D uniforme
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Configuration des axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title = (f"Projection d'un vecteur sur un ellipsoïde, "
         f"centre de l'ellipse: ({x0}, {y0}, {z0}), "
         f"demi-axes: a={a}, b={b}, c={c}.\n"
         f"Vecteur X: ({X[0]}, {X[1]}, {X[2]}), "
         f"Projeté de X : ({Pc[0]},{Pc[1]},{Pc[2]})\n"
         f"⟨X-Pc│t1⟩ : ({Produit_scalaire1}), "
         f"⟨X-Pc│t2⟩ : ({Produit_scalaire2})\n"
         f"maximum d'itérations : ({max_iterations}), tolérance : ({tolerance})")


    ax.set_title(title)

    # Légende
    ax.legend()

    # Affichage
    plt.show()

######## Autre exemple 
 # Paramètres de l'ellipsoïde
a, b, c = 0.5, 1.2, 0.8  # Demi-axes
x0, y0, z0 = 0.2, -0.4, -0.8  # Centre de l'ellipsoïde

X = np.array([1.7,0.17,-1.494])  # Point à projeter
somme = ineq_cartesienne_ellipsoide(X)
if somme<=1 :
    print("X appartient au convexe, pas de projection (sortie du programme)")
    sys.exit()

# Critères de convergence 
max_iterations=100000
tolerance = 0.000000001
alpha = 0.01

(u,v) = projection_ellipsoide(X)
Pc = np.round(point_ellipse_3D(u,v),4)
print("Projection de X sur l'ellipsoïde :", Pc)

(t1,t2)= plan_tangent_ellipsoide(u,v)
print("le plan tangent à l'ellipsoide est :", (t1,t2))
x_proj = X[0] - Pc[0]
y_proj = X[1] - Pc[1]
z_proj = X[2] - Pc[2]
Vect_proj=np.array([x_proj,y_proj,z_proj])

# Verification de l'orthogonalité : 
Produit_scalaire1=np.dot(t1,(Vect_proj/np.linalg.norm(Vect_proj)))
Produit_scalaire2=np.dot(t2,(Vect_proj/np.linalg.norm(Vect_proj)))
print("Le produit scalaire entre X-Pc et la première direction du plan tangent est :", Produit_scalaire1)
print("le produit scalaire entre X-Pc et la deuxième direction du plan tangent est : ", Produit_scalaire2)

######## Autre exemple 
 # Paramètres de l'ellipsoïde
a, b, c = 1, 2, 3  # Demi-axes
x0, y0, z0 = -1, 0.5, 1  # Centre de l'ellipsoïde

X = np.array([-2.5,2,1.6])  # Point à projeter
somme = ineq_cartesienne_ellipsoide(X)
if somme<=1 :
    print("X appartient au convexe, pas de projection (sortie du programme)")
    sys.exit()

# Critères de convergence 
max_iterations=100000
tolerance = 0.000000001
alpha = 0.01

(u,v) = projection_ellipsoide(X)
Pc = np.round(point_ellipse_3D(u,v),4)
print("Projection de X sur l'ellipsoïde :", Pc)

(t1,t2)= plan_tangent_ellipsoide(u,v)
print("le plan tangent à l'ellipsoide est :", (t1,t2))
x_proj = X[0] - Pc[0]
y_proj = X[1] - Pc[1]
z_proj = X[2] - Pc[2]
Vect_proj=np.array([x_proj,y_proj,z_proj])

# Verification de l'orthogonalité : 
Produit_scalaire1=np.dot(t1,(Vect_proj/np.linalg.norm(Vect_proj)))
Produit_scalaire2=np.dot(t2,(Vect_proj/np.linalg.norm(Vect_proj)))
print("Le produit scalaire entre X-Pc et la première direction du plan tangent est :", Produit_scalaire1)
print("le produit scalaire entre X-Pc et la deuxième direction du plan tangent est : ", Produit_scalaire2)

########### Projection sur un ensemble non convexe
###Fonction inverse 
### Cas 1
X = np.array([0, 6])

t = 2 # Paramètre initial
tmin = t  # Meilleure valeur de t trouvée
d_init = float('inf')  # Distance initiale (très grande)
iteration = 0
max_iterations = 10000  # Nombre maximal d'itérations
h = 0.001  # Pas initial
tolerance = 1e-8  # Critère de convergence sur la distance

while iteration < max_iterations:
    t += h  # Incrément de t
    d = distance_parabole1(t)
    
    # Si une meilleure distance est trouvée, on met à jour
    if d < d_init:
        d_init = d
        tmin = t
        iteration += 1
    else:
        # Si la distance n'améliore plus, on réduit le pas pour raffiner
        h =h/2
    # Critère d'arrêt si le pas devient trop petit
    if abs(h) < tolerance:
        break
    

# Résultat
print("La valeur de t minimisant la distance est:", tmin)
print("Distance minimale:", d_init)
print("Nombre d'itérations:", iteration)
    
### On calcul ensuite par la suite le projeté à partir de tmin et de la paramétrisation de l'ellipse
Pc=point_parabole1(tmin)
print("le projeté de X sur C est", Pc)

### Verification des résultats :
# Vecteur tangent normalisé
vecteur_tangent = tangente_parabole1(tmin)
# Produit scalaire entre le projeté et le vecteur tangent :
X1=Pc[0] - X[0]
X2=Pc[1] - X[1]
Vecteur_projection=np.array([X1,X2])
produit_scalaire = np.dot(vecteur_tangent, Vecteur_projection)
print("le produit scalaire entre le vecteur tangent et le vecteur projeté est : ", produit_scalaire)

### Cas 2
X = np.array([0, 6])

t = -3 # Paramètre initial
tmin = t  # Meilleure valeur de t trouvée
d_init = float('inf')  # Distance initiale (très grande)
iteration = 0
max_iterations = 10000  # Nombre maximal d'itérations
h = 0.001  # Pas initial
tolerance = 1e-8  # Critère de convergence sur la distance

while iteration < max_iterations:
    t += h  # Incrément de t
    d = distance_parabole1(t)
    
    # Si une meilleure distance est trouvée, on met à jour
    if d < d_init:
        d_init = d
        tmin = t
        iteration += 1
    else:
        # Si la distance n'améliore plus, on réduit le pas pour raffiner
        h =h/2
    # Critère d'arrêt si le pas devient trop petit
    if abs(h) < tolerance:
        break
    

# Résultat
print("La valeur de t minimisant la distance est:", tmin)
print("Distance minimale:", d_init)
print("Nombre d'itérations:", iteration)
    
### On calcul ensuite par la suite le projeté à partir de tmin et de la paramétrisation de l'ellipse
Pc=point_parabole1(tmin)
print("le projeté de X sur C est", Pc)

### Verification des résultats :
# Vecteur tangent normalisé
vecteur_tangent = tangente_parabole1(tmin)
# Produit scalaire entre le projeté et le vecteur tangent :
X1=Pc[0] - X[0]
X2=Pc[1] - X[1]
Vecteur_projection=np.array([X1,X2])
produit_scalaire = np.dot(vecteur_tangent, Vecteur_projection)
print("le produit scalaire entre le vecteur tangent et le vecteur projeté est : ", produit_scalaire)