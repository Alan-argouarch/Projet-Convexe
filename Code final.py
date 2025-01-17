############################################### Code du projet d'Analyse 5 ######################################################################
"""
Ce fichier comprends les fonctions ainsi que les algorithmes utilisés dans ce projet.

Comme rappelé dans le rapport, le code d'affichage des graphiques n'est pas présent dans ce fichier, afin d'améliorer la lisibilité du code, parce que 
le code d'affichage ne présente pas un grand intêret d'un point de vu théorique et algorithmique, et parce qu'ils sont déja affichés dans le rapport.

De plus, la numérotation ainsi que le nom des parties est la même que celle du rapport.

"""

######## Librairies utilisées :

import numpy as np #pour le calcul vectoriel
import sympy as sp #pour le calcul fonctionnel
import matplotlib.pyplot as plt #Pour l'affichage 2D
from mpl_toolkits.mplot3d import Axes3D # Pour l'affichage 3D
import os #Pour la vidéo de la partie 2.2
import sys #pour la fonction sys.exit()

####################        1.Sous-espaces vectoriels et boules	##################################################
########    1.1	Projection sur un hyperplan	

def proj_hyperplan(X, A):
    """
    Projette le vecteur X sur l' hyperplan H défini par le vecteur A.

    Arguments :
    - X : vecteur de dimension n, représentant le vecteur à projeter.
    - A : vecteur de dimension n+1, contenant le vecteur normal et le terme constant.

    """
    p = A.shape[0] - 1  # Détermine la longueur de la partie normale 
    n = A[:p] # Extrait tous les éléments de A sauf le dernier, correspondant à la partie normale.
    h = A[-1] # Extrait le dernier élément de A

    # Soit F 
    # Calcul de la projection
    proj = X - ((np.dot(X, n)-h)/(np.dot(n,n)))*n
    return proj

vecteur_A = input("Entrez les éléments du vecteur A de dimension n+1, séparés par des espaces : ")
A = np.array([float(x) for x in vecteur_A.split()])

vecteur_X = input("Entrez les éléments du vecteur X de dimension n, séparés par des espaces : ")
X = np.array([float(x) for x in vecteur_X.split()])
    
projection = proj_hyperplan(X, A)
print("La projection de X sur H est :", projection)


########    1.2	Projection sur un demi-espace	

def proj_demi_espace(X, A):
    """
    Projette le vecteur X dans le demi espace F définit par : 
    F : {X∈R^n,A∈R^(n+1)  | a1*x1+⋯+an*xn≤a(n+1)}

    Arguments :
    - X : vecteur de dimension n, représentant le vecteur à projeter.
    - A : vecteur de dimension n+1, contenant le vecteur normal et le terme constant.

    """
    p = A.shape[0] - 1
    n = A[:p]

    "Le vecteur X est-il déja dans F ?"
    #Calcul de la distance d au demi espace F
    d = np.dot(X, n)
    "Le dernier terme de A:a(n+1) est le terme constant c" 
    c = A[-1] 

    if d > c :
        """
        Si la distance d est supérieur à la constante c, alors X n'est pas dans le demi espace F.
        On projète donc X sur l'hyperplan H (car H appartient bien à F, il constitue son "contour")
        """
        #Calcul de la projection
        return(proj_hyperplan(X, A))
    else :
        """
        X est déja dans F car la distance d  est inférieur au terme constant 
        On retourne simplement le vecteur X 
        """
        return(X)
    
#Vecteurs A1 et A2 des deux demis espaces
A1 = np.array([3, 2, 5])  # 3x + 2y <= 5
A2 = np.array([-2, 3, 1])  # -2x + 3y <= 1 

#Vecteur à projeter
X = np.array([10, 10])

#Double projection
X_proj = proj_demi_espace(X, A1)
print(X_proj)
X_proj = proj_demi_espace(X_proj, A2)

print("Projection de X sur P :", X_proj)


########    1.3	Projection sur une boule

def proj_boule(X,X0,r):
    "On calcul la distance de X à la boule"
    #Calcul de la distance à la boule
    Y=X-X0
    distance = np.linalg.norm(Y)
    if distance <= r :
        "X est déja dans la boule, pas besoin de projeter"
        return X
    else: 
        "X n'est pas dans la boule, on projete"
        proj = X0+r*(Y/(np.linalg.norm(Y)))
        return(proj)
    
### Projection sur une boule en 2D
X=np.array([4,6])
X0=np.array([-0.5,2])
r=0.5
Y=X-X0
projection = proj_boule(X,X0,r)
print("La projection de X sur B est :", projection)

### Projection sur une boule en 3D
X = np.array([4,4,4])
X0 = np.array([1, 2, 3])
Y=X-X0
r = 1
projection = proj_boule(X, X0, r)
print("La projection de X sur B est :", projection)



###################         2.	Autres exemples en dimension 2 et 3	##################################################
####################    2.1	Projection sur une ellipse pleine	

def distance_ellipse(t):
    "Fonction distance au carré à minimiser"
    return ((X[0]-x0-a*sp.cos(t))**2) + ((X[1]-y0-b*sp.sin(t))**2) 

def point_ellipse(t):
     x_t=x0+a*sp.cos(t)
     y_t=y0+b*sp.sin(t)
     P=np.array([x_t,y_t])
     return P

def tangente_ellipse(t):
    "On convertit les valeurs au format float pour éviter les erreurs de formats lors des calculs"
    x_t=float(-a*sp.sin(t)) 
    y_t=float(b*sp.cos(t))
    P=np.array([x_t,y_t])
    "Normalisation"
    P=P/np.linalg.norm(P)
    return P

def equation_ellipse_pleine(X):
    x=X[0]
    y=X[1]
    eq = ((x-x0)**2/a**2) + ((y-y0)**2/b**2) 
    if eq<=1 :
        return("X appartient à C")
    else :
        return("X n'appartient pas à C")

x0, y0 = 1,0.5  # Centre de l'ellipse
a, b =3, 5  # Demi-axes

# Vecteur X 
X = np.array([5, 5])

#On test si X appartient au convexe
resultat=equation_ellipse_pleine(X)
if resultat=="X appartient à C":
    print(resultat)
    sys.exit

# Nombre maximum d'itérations
max_iterations = 100  
iteration = 0 #initialisation des itérations


# Initialisation
t = 0  # Paramètre initial
tmin = t  #initialisation de tmin

dinit = float('inf') # Distance initial 
h = 0.01 # pas

# Boucle d'optimisation
while iteration < max_iterations:
     t = t+h  # Incrémentation
     "calcul de la distance à l'ellipse"
     d = distance_ellipse(t)
     # Si la distance calculé est inférieur à la distance calculée précédemment, on enregistre la valeur
     if d < dinit:
         dinit = d
         tmin = t
     iteration = iteration + 1

# Résultat
print("La valeur de t minimisant la distance est:", tmin)
print("Distance minimale:", dinit)
print("Nombre d'itérations:", iteration)
    
### On calcul ensuite par la suite le projeté à partir de tmin et de la paramétrisation de l'ellipse
Pc=point_ellipse(tmin)
print("le projeté de X sur C est", Pc)
# On convertit en valeurs numériques
Pc[0]= float(Pc[0])
Pc[1] = float(Pc[1])


### Verification des résultats :
# Vecteur tangent 
vecteur_tangent = tangente_ellipse(tmin)
# Conversion de format
vecteur_tangent=np.array([float(vecteur_tangent[0]),float(vecteur_tangent[1])])
# Produit scalaire entre le projeté et le vecteur tangent :
X1=Pc[0] - X[0]
X2=Pc[1] - X[1]
Vecteur_projection=np.array([X1,X2])
produit_scalaire = np.dot(vecteur_tangent, Vecteur_projection)
print("le produit scalaire entre le vecteur tangent et le vecteur projeté est : ", produit_scalaire)


##########################    2.2	Projection sur une ellipse pleine quelconque

def distance_ellipse_qqc(t):
    return(((X[0]-x0-a*sp.cos(teta)*sp.cos(t) + b*sp.sin(teta)*sp.sin(t))**2) + (X[1]-y0-a*sp.sin(teta)*sp.cos(t) - b*sp.cos(teta)*sp.sin(t))**2)

def point_ellipse_qqc(t):
     x_t=x0 + a*sp.cos(teta)*sp.cos(t) - b*sp.sin(teta)*sp.sin(t)
     y_t=y0 + a*sp.sin(teta)*sp.cos(t) + b*sp.cos(teta)*sp.sin(t)
     P=np.array([x_t,y_t])
     return P

def tangente_ellipse_qqc(t):
    x_t=float(-a*sp.cos(teta)*sp.sin(t) - b*sp.sin(teta)*sp.cos(t))
    y_t=float(-a*sp.sin(teta)*sp.sin(t) + b*sp.cos(teta)*sp.cos(t))
    P=np.array([x_t,y_t])
    # Normalisation
    P=P/np.linalg.norm(P)
    return P

x0, y0 = 1,0.5  # Centre de l'ellipse
a, b =0.5, 1  # Demi-axes
teta=np.pi/7 
# Vecteur X à projeter 
X = np.array([2, 2])

max_iterations = 10000  # Nombre maximum d'itérations
iteration = 0 #Initialisation des itérations

#Initialisation de t et de tmin
t = 0  
tmin = t  
d_init = float('inf') # Distance initiale
h = 0.0001 #pas


# Boucle d'optimisation
while iteration < max_iterations:
    t = t+h  # Incrément de t
    d = distance_ellipse_qqc(t)
        # Si la distance calculé est inférieur à la distance calculée précédemment, on enregistre la valeur
    if d < d_init:
        d_init = d
        tmin = t

    iteration = iteration+1

    # Résultat
print("La valeur de t minimisant la distance est:", tmin)
print("Distance minimale:", d_init)
print("Nombre d'itérations:", iteration)
    
### On calcul ensuite par la suite le projeté à partir de tmin 
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


###################################    2.3	Projection sur des intérieurs de paraboles
def distance_parabole1(t):
    return ((X[0]-t)**2) + ((X[1]-t**2)**2)

def point_parabole1(t):
     x_t=float(t)
     y_t=float(t**2)
     P=np.array([x_t,y_t])
     return P

def tangente_parabole1(t):
    x_t=1
    y_t=float(2*t)
    P=np.array([x_t,y_t])
    # Normalisation
    P=P/np.linalg.norm(P)
    return P

def ineq_parabole(X):
    x=X[0]
    y=X[1]
    if (x**2)<=y:
        return("X appartient à C")

# Initialisation
X = np.array([4, 3])

# Test pour voir si X appartient au convexe
resultat=ineq_parabole(X)
if resultat=="X appartient à C":
   print("X appartient à C")
   sys.exit()


max_iterations = 10000  # Nombre maximum d'itérations
iteration = 0 #Initialisation des itérations

#Initialisation de t et de tmin
t = 0  
tmin = t  
d_init = float('inf') # Distance initiale
h = 0.0001 #pas
tolerance = 1e-8  # Critère de convergence sur la distance

while iteration < max_iterations:
    t=t+h  # Incrément de t
    d= distance_parabole1(t)
    if d < d_init:
        d_init = d
        tmin = t
        iteration = iteration+1
    else:
        # Si la distance ne se réduit pas, on réduit le pas pour améliorer la précision
        h=h/2
    # Critère d'arrêt si le pas devient trop petit
    if abs(h) < tolerance:
        break
    

# Résultat
print("La valeur de t minimisant la distance est:", tmin)
print("Distance minimale:", d_init)
print("Nombre d'itérations:", iteration)
    
### On calcul ensuite par la suite le projeté à partir de tmin
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

##############################    2.4	Projection sur des demi-hyperboles	
def distance_demi_hyperbole(t):
    return ((X[0] - t)**2) + ((X[1] - (1/t))**2)

def point_demi_hyperbole(t):
     x_t=float(t)
     y_t=float((1/t))
     P=np.array([x_t,y_t])
     return P

def tangente_demi_hyperbole(t):
    x_t=1
    y_t=float(-1/(t**2))
    P=np.array([x_t,y_t])
    # Normalisation
    P=P/np.linalg.norm(P)
    return P

def ineq_demi_hyperbole(X):
    x=X[0]
    y=X[1]
    if (1/x)<=y:
        return("X appartient à C")

X = np.array([0.12, 0.3])

# On vérifie si X appartient au convexe
resultat=ineq_demi_hyperbole(X)
if resultat=="X appartient à C":
   print(resultat)
   sys.exit()

# Initialisation

max_iterations = 10000  # Nombre maximum d'itérations
iteration = 0 #Initialisation des itérations

#Initialisation de t et de tmin
t = 0  
tmin = t  
d_init = float('inf') # Distance initiale
h = 0.0001 #pas
tolerance = 1e-8  # Critère de convergence sur la distance

while iteration < max_iterations:
    t = t+h  # Incrément de t
    d = distance_demi_hyperbole(t)

    if d < d_init:
        d_init = d
        tmin = t
        iteration = iteration + 1
    else:
        # Si la distance ne se réduit pas, on réduit le pas pour améliorer la précision
        h =h/2
    # Critère d'arrêt si le pas devient trop petit
    if abs(h) < tolerance:
        break
    

# Résultat
print("La valeur de t minimisant la distance est:", tmin)
print("Distance minimale:", d_init)
print("Nombre d'itérations:", iteration)
    
### On calcul ensuite par la suite le projeté à partir de tmin 
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

########    2.5	Projection sur des ensembles non convexes	
### On test deux projections pour la fonction y=x^2
### Cas 1
# Initialisation
X = np.array([0, 6])

max_iterations = 10000  # Nombre maximum d'itérations
iteration = 0 #Initialisation des itérations

#Initialisation de t et de tmin
t = 2 
tmin = t  
d_init = float('inf') # Distance initiale
h = 0.0001 #pas
tolerance = 1e-8  # Critère de convergence sur la distance

while iteration < max_iterations:
    t=t+h  # Incrément de t
    d= distance_parabole1(t)
    if d < d_init:
        d_init = d
        tmin = t
        iteration = iteration+1
    else:
        # Si la distance ne se réduit pas, on réduit le pas pour améliorer la précision
        h=h/2
    # Critère d'arrêt si le pas devient trop petit
    if abs(h) < tolerance:
        break
    

# Résultat
print("La valeur de t minimisant la distance est:", tmin)
print("Distance minimale:", d_init)
print("Nombre d'itérations:", iteration)
    
### On calcul ensuite par la suite le projeté à partir de tmin
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

max_iterations = 10000  # Nombre maximum d'itérations
iteration = 0 #Initialisation des itérations

#Initialisation de t et de tmin
t = -3
tmin = t  
d_init = float('inf') # Distance initiale
h = 0.0001 #pas
tolerance = 1e-8  # Critère de convergence sur la distance

while iteration < max_iterations:
    t=t+h  # Incrément de t
    d= distance_parabole1(t)
    if d < d_init:
        d_init = d
        tmin = t
        iteration = iteration+1
    else:
        # Si la distance ne se réduit pas, on réduit le pas pour améliorer la précision
        h=h/2
    # Critère d'arrêt si le pas devient trop petit
    if abs(h) < tolerance:
        break
    

# Résultat
print("La valeur de t minimisant la distance est:", tmin)
print("Distance minimale:", d_init)
print("Nombre d'itérations:", iteration)
    
### On calcul ensuite par la suite le projeté à partir de tmin
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

###################################    2.6	Projection sur un ellipsoïde plein en 3D	

def point_ellipse_3D(u, v):
    x = x0 + a*np.cos(u)*np.cos(v)
    y = y0 + b*np.sin(u)*np.cos(v)
    z = z0 + c*np.sin(v)
    return np.array([x, y, z])

# Fonction objectif : distance au carré
def distance_ellipse_3D(u, v):
    P = point_ellipse_3D(u, v)
    return np.sum((X-P)**2)

def gradient_distance(u, v):
    P = point_ellipse_3D(u, v)
    #On approxime les dérivées partielles à l'aide de la méthode des différences finies
    h = 0.00001
    grad_u = (distance_ellipse_3D(u+h, v) - distance_ellipse_3D(u-h, v))/(2*h)
    grad_v = (distance_ellipse_3D(u, v+h) - distance_ellipse_3D(u, v-h))/(2*h)
    return np.array([grad_u, grad_v])

def projection_ellipsoide(X):
    u, v = 0.0, 0.0  # Initialisation
    for i in range(max_iterations):
        grad = gradient_distance(u, v)
        norm_grad = np.linalg.norm(grad)
        if norm_grad < tolerance:  # Critère de convergence
            break
        u = u - (alpha * grad[0])
        v = v-  (alpha * grad[1])
    return (u,v) 

def vecteur_ecart(X,Pc):
    P = (X-Pc)
    return P

def plan_tangent_ellipsoide(u,v):
    x1 = -a*np.sin(u)*np.cos(v)
    y1 = b*np.cos(u)*np.cos(v)
    z1 = 0
    t1 = np.array([x1,y1,z1])
    x2 = -a*np.cos(u)*np.sin(v)
    y2 = -b*np.sin(u)*np.sin(v)
    z2 = c*np.cos(v)
    t2 = np.array([x2,y2,z2])
    t1=t1/np.linalg.norm(t1)
    t2=t2/np.linalg.norm(t2)
    return(t1,t2)

def ineq_cartesienne_ellipsoide(X) :
    A = ((X[0]-x0)/a)**2
    B = ((X[1]-y0)/b)**2
    C = ((X[2]-z0)/c)**2
    somme = A+B+C
    return(somme)

# Paramètres de l'ellipsoïde
a, b, c = 1, 1, 1  # Demi-axes
x0, y0, z0 = 0, 0, 0  # Centre de l'ellipsoïde

X = np.array([1,1,1])  #Vecteur à projeter 
somme = ineq_cartesienne_ellipsoide(X)
if somme<=1 :
    print("X appartient au convexe, pas de projection (sortie du programme)")
    sys.exit()

#Critères de convergence 
max_iterations=100000
tolerance = 0.000000001
alpha = 0.01

#Calcul du projeté
(u,v) = projection_ellipsoide(X)
Pc = np.round(point_ellipse_3D(u,v),4) #On arrondit à 4 chiffres après la virgule 
print("Projection de X sur l'ellipsoïde :", Pc)

(t1,t2)= plan_tangent_ellipsoide(u,v)
print("le plan tangent à l'ellipsoide est :", (t1,t2))
x_proj = X[0]-Pc[0]
y_proj = X[1]-Pc[1]
z_proj = X[2]-Pc[2]
Vect_proj=np.array([x_proj,y_proj,z_proj])

# Verification de l'orthogonalité : 
Produit_scalaire1=np.dot(t1,(Vect_proj/np.linalg.norm(Vect_proj)))
Produit_scalaire2=np.dot(t2,(Vect_proj/np.linalg.norm(Vect_proj)))
print("Le produit scalaire entre X-Pc et la première direction du plan tangent est :", Produit_scalaire1)
print("le produit scalaire entre X-Pc et la deuxième direction du plan tangent est : ", Produit_scalaire2)

##################          3.	Intersection de convexe	 ##################################################
# Croisement entre deux droites
import numpy as np
import matplotlib.pyplot as plt

def projection_orthogonale(x, y, a, b, c):
    # pour effectuer la projection orthogonale on effectue le produit scalaire
    # entre le vecteur (1,-a/b) et le vecteur (x_proj-x,y_proj-y).
    # On résoud ensuite cette équation. 
    x_proj=(x-a*y/b -a*c/b**2)/(1+a**2/b**2)
    y_proj=-a/b*x_proj-c/b

    return x_proj, y_proj

# point de coordonnées (x,y) et doite d'équation ax+by+c=0 et dx+ey+f=0
def croisement(x,y,a,b,c,d,e,f):
   
    tracer_droite(a,b,c)
    tracer_droite(d,e,f)
    plt.plot(x,y, marker="x",color="red")
    
    x1,y1= projection_orthogonale(x, y, a, b, c)
    x2,y2= projection_orthogonale(x1, y1, d, e, f)
    compteur=2
    
    plt.plot((x,x1),(y,y1))
    plt.plot((x1,x2),(y1,y2))
    
    while abs(x1-x2)>0.01 or abs(y1-y2)>0.01:
        x1,y1= projection_orthogonale(x2, y2, a, b, c)
        plt.plot((x2,x1),(y2,y1))
        x2,y2= projection_orthogonale(x1, y1, d, e, f)
        plt.plot((x1,x2),(y1,y2))
        compteur+=2
 
    plt.grid()
    plt.axhline(0, color='black',linewidth=0.5, linestyle='--')  # Axe horizontal
    plt.axvline(0, color='black',linewidth=0.5, linestyle='--')  # Axe vertical
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()    
 
    return x2,y2 , compteur

def tracer_droite(a,b,c):
    plt.axis("equal")
    plt.axis([-3, 6, -5, 4])      # L'intervalle dépend de vers où sera situé l'intersection
    x = np.linspace(-3, 6, 500)
    y = -a/b * x - c/b
    plt.plot(x, y,color="black")
	
########    3.4  	En dimension 2 et 3.  

### Projection sur un hyperplan 
import numpy as np 

def proj_hyperplan(X, A):
    # Extraire la partie normale de A
        n =np.array(A[:-1])
        h=A[-1]
    # Soit F 
    # Calcul de la projection
        proj = X - ((np.dot(X, n)-h)/(np.dot(n,n)))*n
        return proj

def croisement(X,A,B):
    
        projA= proj_hyperplan(X,A)
        projB= proj_hyperplan(projA,B)
        compteur=2
    
        while abs(projA[0]-projB[0])>0.01 or abs(projA[1]-projB[1])>0.01 or abs(projA[2]-projB[2])>0.01:
            projA= proj_hyperplan(projB,A)
            projB= proj_hyperplan(projA,B)
            compteur+=2
        
        return projB , compteur

##### Représentation graphique de la projection sur 2 plans 
    # Librairies
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# X=(x,y,z) pour une représentations en 3 dimensions
def proj_hyperplan(X, A):

    # Extraire la partie normale de A
        n =np.array(A[:-1])
        h=A[-1]
    # Soit F 
    # Calcul de la projection
        proj = X - ((np.dot(X, n)-h)/(np.dot(n,n)))*n
        return proj

def croisement(T,A,B):
     
     plt.scatter(T[0],T[1],T[2], color='red')
     Z= -A[0]/A[2]*X -A[1]/A[2]*Y -A[3]/A[2]
     plt.plot_surface(X, Y,Z, alpha=0.5, color='red', label='Plan Z=X+Y')
    
     Z= -B[0]/B[2]*X -B[1]/B[2]*Y -B[3]/B[2]
     plt.plot_surface(X, Y,Z, alpha=0.5, color='green', label='Plan Z=X+Y')
     
     projA= proj_hyperplan(T,A)
     plt.plot((T[0],projA[0]), (T[1],projA[1]), (T[2],projA[2]), color='blue', linewidth=2, label='Ligne 3D')
     projB= proj_hyperplan(projA,B)
     plt.plot((projA[0],projB[0]), (projA[1],projB[1]), (projA[2],projB[2]), color='blue', linewidth=2, label='Ligne 3D')
     compteur=2
    
     while abs(projA[0]-projB[0])>0.01 or abs(projA[1]-projB[1])>0.01 or abs(projA[2]-projB[2])>0.01:
         projA= proj_hyperplan(projB,A)
         plt.plot((projB[0],projA[0]), (projB[1],projA[1]), (projB[2],projA[2]), color='blue', linewidth=2, label='Ligne 3D')
         projB= proj_hyperplan(projA,B)
         plt.plot((projA[0],projB[0]), (projA[1],projB[1]), (projA[2],projB[2]), color='blue', linewidth=2, label='Ligne 3D')

         compteur+=2
     return projB , compteur

fig = plt.figure() 
plt = fig.add_subplot(111, projection='3d')

     # Définition de l'espace 3D
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

print(croisement((-4,-1,-2),(-4,1,1,1),(4,-6,8,2)))

### Intersection de 2 boules
import numpy as np 

def proj_boule(X,X0,r):
    "On calcul la distance de X à la boule"
    #Calcul de la distance à la boule
    X=np.array(X)
    X0=np.array(X0)
    Y=X-X0
    distance = np.linalg.norm(Y)
    if distance <= r :
        "X est déja dans la boule, pas besoin de projeter"
        return X
    else: 
        "X n'est pas dans la boule, on projete"
        proj = X0+r*(Y/(np.linalg.norm(Y)))
        return(proj)
    
# X=[x,y,z]: point, A=[x,y,z]: coordonnées centre de la boule, a=r:rayon boule
def croisement(X,A,a,B,b):
    A=np.array(A)
    B=np.array(B) 
   # on vérifie qu'il y a une intersection entre les 2 boules
    dist = np.linalg.norm(B-A)
    if dist > a+b:
        return "il n'y a pas d'intersection"
        
    projA= proj_boule(X,A,a)
    projB= proj_boule(projA,B,b)
    compteur=2
    
    while abs(projA[0]-projB[0])>0.001 or abs(projA[1]-projB[1])>0.001 or abs(projA[2]-projB[2])>0.001:
        projA= proj_boule(projB,A,a)
        projB= proj_boule(projA,B,b)    
        compteur+=2
        
    return projB , compteur

### Intersection 2 boules, représentation graphique
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def proj_boule(X,X0,r):
    "On calcul la distance de X à la boule"
    #Calcul de la distance à la boule
    X=np.array(X)
    X0=np.array(X0)
    
    Y=X-X0
    distance = np.linalg.norm(Y)
    if distance <= r :
        "X est déja dans la boule, pas besoin de projeter"
        return X
    else: 
        "X n'est pas dans la boule, on projete"
        proj = X0+r*(Y/(np.linalg.norm(Y)))
        return(proj)
    
# X=[x,y,z]: point, A=[x,y,z]: coordonnées centre de la boule, a=r:rayon boule
def croisement(X,A,a,B,b):
    plt.scatter(X[0],X[1],X[2], color='red')
    A=np.array(A)
    B=np.array(B)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    AX_sphere = A[0] + a * np.outer(np.cos(u), np.sin(v))
    AY_sphere = A[1] + a * np.outer(np.sin(u), np.sin(v))
    AZ_sphere = A[2] + a * np.outer(np.ones(np.size(u)), np.cos(v))
    plt.plot_surface(AX_sphere, AY_sphere, AZ_sphere, color='b', alpha=0.3)
    BX_sphere = B[0] + b * np.outer(np.cos(u), np.sin(v))
    BY_sphere = B[1] + b * np.outer(np.sin(u), np.sin(v))
    BZ_sphere = B[2] + b * np.outer(np.ones(np.size(u)), np.cos(v))
    plt.plot_surface(BX_sphere, BY_sphere, BZ_sphere, color='g', alpha=0.3)
     
   # on vérifie qu'il y a une intersection entre les 2 boules
    dist = np.linalg.norm(B-A)
    if dist > a+b:
        return "il n'y a pas d'intersection"
    
    
    projA= proj_boule(X,A,a)
    plt.plot((X[0],projA[0]), (X[1],projA[1]), (X[2],projA[2]), color='red', linewidth=2, label='Ligne 3D')
    projB= proj_boule(projA,B,b)
    plt.plot((projA[0],projB[0]), (projA[1],projB[1]), (projA[2],projB[2]), color='red', linewidth=2, label='Ligne 3D')
    compteur=2
    
    while abs(projA[0]-projB[0])>0.001 or abs(projA[1]-projB[1])>0.001 or abs(projA[2]-projB[2])>0.001:
        projA= proj_boule(projB,A,a)
        plt.plot((projB[0],projA[0]), (projB[1],projA[1]), (projB[2],projA[2]), color='red', linewidth=2, label='Ligne 3D')
        projB= proj_boule(projA,B,b)
        plt.plot((projA[0],projB[0]), (projA[1],projB[1]), (projA[2],projB[2]), color='red', linewidth=2, label='Ligne 3D')
        compteur+=2
        
    return projB , compteur

fig = plt.figure() 
plt = fig.add_subplot(111, projection='3d')

     # Définition de l'espace 3D
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

croisement([5,5,5],[0,0,0],2,[3.9,0,0],2) 
   
##################          4.	Applications #####################################	
#---- problème 1 :  ----
X=proj_boule([1,1,1,1],[1,0,-3,2],2)   
for k in range(100):
    X=proj_hyperplan(X,[2,-1,3,1,-3])    
    X=proj_boule(X,[1,0,-3,2],2)
print(X)



#---- problème 2 :  ----

projA =proj_demi_espace([0,0,0,0,0,0], [1,1,1,1,1,1,-4])  
for k in range (100):
      projB =proj_demi_espace(projA, [-2,1,3,-2,1,-1,-7]) 
      projC =proj_demi_espace(projB, [-6,5,-7,0,9,-5,12]) 
      projD =proj_demi_espace(projC, [1,0,8,3,5,8,-1]) 
      projA =proj_demi_espace(projD, [1,1,1,1,1,1,-4] ) 

print(projA)
print(projB)
print(projC)
print(projD)

# produit scalaire de 2 listes pour vérifié inéquation
def p(liste1, liste2):
    # Vérifier que les listes ont la même longueur
    if len(liste1) != len(liste2):
        raise ValueError("Les deux listes doivent avoir la même longueur.")
    
    # Calculer le produit scalaire
    return sum(x * y for x, y in zip(liste1, liste2))

x= projA.tolist()
a= [1,1,1,1,1,1]
b=[-2,1,3,-2,1,-1]
c=[-6,5,-7,0,9,-5]
d=[1,0,8,3,5,8]

print(p(x,a),p(x,b),p(x,c),p(x,d)) # il faut vérifie que les équations sont inférieures à -4 , -7 , 12 et -1
print("la norme est :" , p(x,x)**0.5)



#---- problème 3 :  ----
for a in np.arange(3.742,3.746,0.00001):   # on réduit l'intervalle au fur et à mesure des essais
    max=a
    X=proj_boule([2,2,2],[0,0,0],1)   
    for k in range(1000):   # on augmente la précision de la méthode avec plus d'itérations
        Y=proj_hyperplan(X,[1,2,3,a])    
        X=proj_boule(Y,[0,0,0],1)
    if abs(X[0]-Y[0])>0.0001 or abs(X[1]-Y[1])>0.0001 or abs(X[2]-Y[2])>0.0001: # on augmente la précision en réduisant l'écart entre les 2 projections 
        print(X,Y)
        break
print(max)



#---- problème 4 :  ---

def distance_demi_hyperbole(t):
    h = t
    v = 10 / np.sqrt(2 * h)
    X = np.array([10, 0])  
    P = np.array([h, v])  
    return np.sum((X - P)**2)


def point_demi_hyperbole(t):
    h = t
    v = 10 / np.sqrt(2 * h)
    return np.array([h, v])


def tangente_demi_hyperbole(t):
    h = t
    # Dérivée de v = 10 / sqrt(2h)
    dv_dh = -10 / (2 * (2 * h)**(3/2))
    return np.array([1, dv_dh]) / np.linalg.norm([1, dv_dh])  # On normalise

def minimiser_energie():
    """
    Projette le point X=(10, 0) sur la demi-hyperbole définie par v*sqrt(2h) = 10
    """
    # Initialisation
    X = np.array([10, 0])
    t = 1.0
    max_iterations = 10000
    tolerance = 1e-8
    h = t

    for iteration in range(max_iterations): 
        P = point_demi_hyperbole(h) 
        vecteur_proj = X - P   
        vecteur_tangent = tangente_demi_hyperbole(h)  
        produit_scalaire = np.dot(vecteur_proj, vecteur_tangent)
        if abs(produit_scalaire) < tolerance: # On arrête si vecteur_proj est orthogonal au vecteur tangent
            break
        h = h + 0.01 * produit_scalaire
    P_opt = point_demi_hyperbole(h)  # Point projeté optimal
    v = P_opt[1]
    energie_min = 10 * h + 0.5 * v**2
    return h, v, energie_min, iteration


h, v, energie_min, iteration = minimiser_energie()
print(f"Hauteur optimale (h) : {h:.6f}")
print(f"Vitesse optimale (v) : {v:.6f}")
print(f"Énergie minimale : {energie_min:.6f}")
print(f"Nombre d'itérations : {iteration}")



