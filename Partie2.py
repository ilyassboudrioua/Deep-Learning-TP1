import torch

# 1. Données
x = torch.tensor(2.0)
y = torch.tensor(8.0)

# 2. Paramètre à apprendre (requires_grad=True = PyTorch trace les calculs)
w = torch.tensor(1.0, requires_grad=True)

# 3. Forward
y_pred = w * x

# 4. Perte
loss = (y_pred - y) ** 2

# 5. Backward (rétropropagation)
loss.backward()

# 6. Affichage du gradient
print("w.grad =", w.grad)