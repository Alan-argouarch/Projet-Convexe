# Librairies
import numpy as np 

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



if __name__ == "__main__":
    vecteur_A = input("Entrez les éléments du vecteur A de dimension n+1, séparés par des espaces : ")
    A = np.array([float(x) for x in vecteur_A.split()])

    vecteur_X = input("Entrez les éléments du vecteur X de dimension n, séparés par des espaces : ")
    X = np.array([float(x) for x in vecteur_X.split()])

    
    projection = proj_demi_espace(X, A)
    print("La projection de X sur F est :", projection)
    
