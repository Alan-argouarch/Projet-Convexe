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