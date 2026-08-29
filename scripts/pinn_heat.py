import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Training Data Setup
x_ic = torch.rand(100, 1)
t_ic = torch.zeros(100, 1)
T_ic = torch.sin(torch.pi * x_ic)

x_bc_left = torch.zeros(50, 1)
x_bc_right = torch.ones(50, 1)
x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
t_bc = torch.rand(100, 1)
T_bc = torch.zeros(100, 1)

x_phys = torch.rand(10000, 1).requires_grad_(True)
t_phys = torch.rand(10000, 1).requires_grad_(True)

# 2. PINN Architecture
class HEAT_MLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=1, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1) 
        return self.net(inputs)

model = HEAT_MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
alpha = 0.01

# 3. Training Loop
print("Training PINN...")
for epoch in range(5000):
    optimizer.zero_grad()
    
    loss_ic = loss_fn(model(x_ic, t_ic), T_ic)
    loss_bc = loss_fn(model(x_bc, t_bc), T_bc)
    
    T_pred_phys = model(x_phys, t_phys)
    
    T_t = torch.autograd.grad(T_pred_phys, t_phys, grad_outputs=torch.ones_like(T_pred_phys), create_graph=True)[0]
    T_x = torch.autograd.grad(T_pred_phys, x_phys, grad_outputs=torch.ones_like(T_pred_phys), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x_phys, grad_outputs=torch.ones_like(T_x), create_graph=True)[0]
    
    residual = T_t - alpha * T_xx
    loss_phys = torch.mean(residual**2)
    
    loss_total = loss_ic + loss_bc + loss_phys
    loss_total.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Total: {loss_total.item():.5f} | Phys: {loss_phys.item():.5f}")

# 4. Evaluation at t=5.0
x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
t_test = torch.tensor([5.0]).repeat(100, 1)

with torch.no_grad():
    T_test = model(x_test, t_test)
    
T_exact = torch.sin(torch.pi * x_test) * torch.exp(-alpha * (torch.pi**2) * t_test)

plt.figure(figsize=(10, 5))
plt.scatter(x_ic.detach().numpy(), T_ic.detach().numpy(), label='Initial Condition (t=0)', color='blue', s=10, alpha=0.3)
plt.scatter(x_bc.detach().numpy(), T_bc.detach().numpy(), label='Boundary Condition', color='red', s=10, alpha=0.3)
plt.plot(x_test.detach().numpy(), T_exact.detach().numpy(), label='Exact Temperature at t=5', color='orange', linewidth=2)
plt.plot(x_test.detach().numpy(), T_test.detach().numpy(), label='PINN Prediction at t=5', color='green', linewidth=2, linestyle='--')

plt.title("1D Heat Equation: PINN Prediction vs Exact Analytical Solution")
plt.xlabel("Position (x)")
plt.ylabel("Temperature (T)")
plt.legend() 
plt.show()