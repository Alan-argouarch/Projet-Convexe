############################### AFFICHAGE DES RESULTATS DU PROJET ########################################

"""
Pour afficher les résultats, simplement copier coller le code dans le fichier python du projet dans la partie 
if __name__ == "__main__": puis lancer le code
Ne pas oublier de bien importer les librairies 
"""
########################################################################################################

##################### Partie 1.1.1

vecteur_A = input("Entrez les éléments du vecteur A de dimension n+1, séparés par des espaces : ")
 A = np.array([float(x) for x in vecteur_A.split()])

vecteur_X = input("Entrez les éléments du vecteur X de dimension n, séparés par des espaces : ")
X = np.array([float(x) for x in vecteur_X.split()])
    
projection = proj_hyperplan(X, A)
print("La projection de X sur H est :", projection)

##################### Partie 1.1.2
vecteur_A = input("Entrez les éléments du vecteur A de dimension n+1, séparés par des espaces : ")
 A = np.array([float(x) for x in vecteur_A.split()])

vecteur_X = input("Entrez les éléments du vecteur X de dimension n, séparés par des espaces : ")
X = np.array([float(x) for x in vecteur_X.split()])
    
projection = proj_demi_espace(X, A)
print("La projection de X sur F est :", projection)

##################### Partie 1.1.3
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
### Projection sur une ellipse :
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
