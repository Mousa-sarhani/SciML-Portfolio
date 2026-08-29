import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim

# 1. Experimental Data exportation and Cleaning
file_path = '/Users/moussaserhani/SciML-Portfolio/data/brass(1).csv'
df = pd.read_csv(file_path , skiprows=list(range(16))+[17])

# 2. Define physical constants for conversion
GAUGE_LENGTH = 25.25 # mm
CROSS_SECTIONAL_AREA = 19.63 # mm^2

# 3. Calculate Engineering Stress and Strain
df['Strain'] = df['Extension'] / GAUGE_LENGTH 
df['Stress'] = df['Load'] / CROSS_SECTIONAL_AREA 

# 4. Extract as numpy arrays
strain_data = df['Strain'].values.reshape(-1, 1)
stress_data = df['Stress'].values.reshape(-1, 1)

# 5. Normalize data (Min-Max Scaling)
strain_min, strain_max = strain_data.min(), strain_data.max()
stress_min, stress_max = stress_data.min(), stress_data.max()

strain_data_norm = (strain_data - strain_min) / (strain_max - strain_min)
stress_data_norm = (stress_data - stress_min) / (stress_max - stress_min) 

strain_tensor_norm = torch.tensor(strain_data_norm, dtype=torch.float32)
stress_tensor_norm = torch.tensor(stress_data_norm, dtype=torch.float32)

# 6. Define Neural Network
class ST_MLP(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

model = ST_MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 7. Training Loop
print("Training MLP...")
for epoch in range(2000):
    optimizer.zero_grad()
    Stress_pred = model(strain_tensor_norm)
    loss = loss_fn(Stress_pred, stress_tensor_norm)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch: {epoch:4d} | Loss: {loss.item():.6f}")

# 8. Evaluation and Plotting
with torch.no_grad():
    stress_pred_norm = model(strain_tensor_norm).numpy()

# Inverse Scaling (Back to physical MPa)
stress_pred_physical = stress_pred_norm * (stress_max - stress_min) + stress_min

plt.figure(figsize=(10, 5))
plt.scatter(strain_data, stress_data, label="True Experimental Data", color="blue", alpha=0.3)
plt.plot(strain_data, stress_pred_physical, label="MLP Approximation", color="red", linewidth=2)
plt.legend()
plt.title("Stress-Strain Curve Approximation using MLP")
plt.xlabel("Strain (mm/mm)")
plt.ylabel("Stress (MPa)")
plt.show()