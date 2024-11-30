# Librairies
import numpy as np 
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    # Soit F 
    # Calcul de la projection
    proj = X - (np.dot(X, n)/(np.linalg.norm(n)**2))*n
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
    c = A[p] 

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


if __name__ == "__main__":
    "Coller ici le code à éxécuter"
    



    

    