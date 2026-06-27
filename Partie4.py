import torch


# Données (reprises de la Partie 3)
X = torch.tensor([1.0, 2.0, 3.0, 4.0])
Y = torch.tensor([3.0, 6.0, 9.0, 12.0])

# 2. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilise :", device)

# 3. Mise en forme 2D : (batch_size, features) -> (4, 1)
X = X.view(-1, 1)
Y = Y.view(-1, 1)

# 4. Envoi sur le device
X = X.to(device)
Y = Y.to(device)

# 5. Modèle de régression linéaire (y = wx + b)
model = torch.Linear(in_features=1, out_features=1).to(device)

# 6. Fonction de perte
criterion = torch.MSELoss()

# 7. Optimiseur (descente de gradient stochastique)
optimizer = torch.SGD(model.parameters(), lr=0.01)

# 8 et 9. Boucle d'entraînement sur 200 époques
for epoch in range(200):
    y_pred = model(X)
    loss = criterion(y_pred, Y)

    optimizer.zero_grad()  # 1. on vide les anciens gradients
    loss.backward()        # 2. on calcule les nouveaux gradients
    optimizer.step()       # 3. l'optimiseur met à jour les paramètres

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss={loss.item():.4f}")

# 10. Paramètres appris
for name, param in model.named_parameters():
    print(name, "=", param.item())

# 11. Inférence pour x = 10
test_input = torch.tensor([[10.0]]).to(device)
prediction = model(test_input)
print("Prediction pour x = 10 :", prediction.item())