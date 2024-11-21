import numpy as np

vecteur = input("Entrez les éléments du vecteur A de dimension n+1, séparés par des espaces : ")
# Convertir l'entrée en un tableau NumPy
A = np.array([float(x) for x in vecteur.split()])
print("vecteur A :", A)

vecteur = input("Entrez les éléments du vecteur X de dimension n, séparés par des espaces : ")
# Convertir l'entrée en un tableau NumPy
X = np.array([float(x) for x in vecteur.split()])
print("vecteur X :", X)

p = A.shape[0]-1
n=A[:p]
print("Vecteur n :", n)
proj=X-((np.dot(X, n))/(np.linalg.norm(n))**2)*n
print("la projection de X sur H est :", proj)

    
