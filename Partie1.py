import torch

# 1. Tenseur A (4x3), valeurs aléatoires loi normale
A = torch.randn(4, 3)

# 2. Tenseur B (3x2), rempli de 1
B = torch.ones(3, 2)

# 3. Affichage des dimensions
print("Shape de A :", A.shape)
print("Shape de B :", B.shape)

# 4. Produit matriciel
C = torch.matmul(A, B)
print("C =", C)
print("Shape de C :", C.shape)
"""  view() permet de modifier la forme du tenseur sans changer ses valeurs.
PyTorch calcule automatiquement la taille nécessaire. """

# 5. Changement de forme
C2 = C.view(-1)
print("C2 =", C2)
print("Shape de C2 :", C2.shape)