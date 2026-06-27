import torch

# 1 et 2. Données : on veut apprendre y = 3x
X = torch.tensor([1.0, 2.0, 3.0, 4.0])
Y = torch.tensor([3.0, 6.0, 9.0, 12.0])

# 3. Initialisation du poids
w = torch.tensor(0.0, requires_grad=True)

# 4. Taux d'apprentissage
lr = 0.01

# 5. Boucle d'entraînement
for epoch in range(100):
    y_pred = w * X
    loss = ((y_pred - Y) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad

    w.grad.zero_()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss={loss.item():.4f}, w={w.item():.4f}")

# 6. Valeur finale
print("w final =", w.item())