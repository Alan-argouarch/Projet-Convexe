# Librairies
import numpy as np 
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

### Projection sur un hyperplan 
def proj_hyperplan(X, A):
    """
    Projette le vecteur X sur l' hyperplan H défini par le vecteur A.

    Arguments :
    - X : vecteur de dimension n, représentant le vecteur à projeter.
    - A : vecteur de dimension n+1, contenant le vecteur normal et le terme constant.

    """
    # Extraire la partie normale de A
    p = A.shape[0] - 1
    n = A[:p]
    h = A[-1]
    # Soit F 
    # Calcul de la projection
    proj = X - ((np.dot(X, n)-h)/(np.dot(n,n)))*n
    return proj


### Projection sur un demi espace 
def proj_demi_espace(X, A):
    """
    Projette le vecteur X dans le demi espace F définit par : 
    F : {X∈R^n,A∈R^(n+1)  | a1*x1+⋯+an*xn≤a(n+1)}

    Arguments :
    - X : vecteur de dimension n, représentant le vecteur à projeter.
    - A : vecteur de dimension n+1, contenant le vecteur normal et le terme constant.

    """
    # Extraire la partie normale de A
    p = A.shape[0] - 1
    n = A[:p]

    "Le vecteur X est-il déja dans F ?"
    # Calcul de la distance d au demi espace F
    d = np.dot(X, n)
    "Le dernier terme de A:a(n+1) est le terme constant c" 
    # Calcul de c 
    c = A[-1] 

    if d > c :
        """
        Si la distance d est supérieur à la constante c, alors X n'est pas dans le demi espace F.
        On projète donc X sur l'hyperplan H (car H appartient bien à F, il constitue son "contour")
        """
        # Calcul de la projection
        return(proj_hyperplan(X, A))
    else :
        """
        X est déja dans F car la distance d  est inférieur au terme constant 
        On retourne simplement le vecteur X 
        """
        return(X)
    



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
    
def distance_ellipse(t):
    return ((X[0] - x0 - a * sp.cos(t))**2) + ((X[1] - y0 - b * sp.sin(t))**2)

def point_ellipse(t):
     x_t=x0+a*sp.cos(t)
     y_t=y0+b*sp.sin(t)
     P=np.array([x_t,y_t])
     return P

def tangente_ellipse(t):
    x_t=float(-a*sp.sin(t))
    y_t=float(b*sp.cos(t))
    P=np.array([x_t,y_t])
    # Normalisation
    P=P/np.linalg.norm(P)
    return P

def distance_ellipse_qqc(t):
    return(((X[0] - x0 - a*sp.cos(teta)*sp.cos(t) + b*sp.sin(teta)*sp.sin(t))**2) + (X[1] - y0 - a*sp.sin(teta)*sp.cos(t) - b*sp.cos(teta)*sp.sin(t))**2)

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

def point_ellipse_3D(u, v):
    x = x0 + a * np.cos(u) * np.cos(v)
    y = y0 + b * np.sin(u) * np.cos(v)
    z = z0 + c * np.sin(v)
    return np.array([x, y, z])

# Fonction objectif : distance au carré
def distance_ellipse_3D(u, v):
    P = point_ellipse_3D(u, v)
    return np.sum((X - P) ** 2)

def gradient_distance(u, v):
    P = point_ellipse_3D(u, v)
    #On approxime les dérivées partielles à l'aide de la méthode des différences finies
    h = 0.00001
    grad_u = (distance_ellipse_3D(u + h, v) - distance_ellipse_3D(u - h, v)) / (2 * h)
    grad_v = (distance_ellipse_3D(u, v + h) - distance_ellipse_3D(u, v - h)) / (2 * h)
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

def vecteur_normal_ellipsoide(u,v):
    x = b*c*np.cos(u)*(np.cos(v)**2)
    y = -a*c*np.sin(u)*(np.cos(v)**2)
    z = a*b*np.cos(v)*np.sin(v)
    P = np.array([x,y,z])
    return P/np.linalg.norm(P)

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




if __name__ == "__main__": 
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
    


    
  


        



    
