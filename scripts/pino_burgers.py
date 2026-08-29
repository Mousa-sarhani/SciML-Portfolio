import torch
import torch.nn as nn

# 1. Setup the Finite Difference Stencil
dx = 0.01 
fd_filter = nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)

# Central difference kernel: [-1, 0, 1] / (2*dx)
fd_kernel = torch.tensor([[[-1.0, 0.0, 1.0]]]) / (2 * dx)

# Inject stencil into the layer and freeze weights
fd_filter.weight.data = fd_kernel
fd_filter.weight.requires_grad = False 

loss_fn = nn.MSELoss()

# 2. The PINO Loss Function
def pino_loss(fno_prediction_grid, true_target_grid):
    
    # Step A: Data Loss (MSE between prediction and true target)
    data_loss = loss_fn(fno_prediction_grid, true_target_grid)
    
    # Step B: Spatial derivative (du/dx) across the predicted grid
    du_dx = fd_filter(fno_prediction_grid)
    
    # Step C: Physics Loss (Enforcing incompressibility du/dx = 0)
    physics_loss = torch.mean(du_dx**2)
    
    # Step D: Total loss combination
    total_loss = data_loss + 0.1 * physics_loss
    
    return total_loss

# 3. Test execution block
if __name__ == "__main__":
    print("Testing PINO Loss formulation...")
    mock_prediction = torch.randn(10, 1, 100) # 10 simulations, 1 channel, 100 pixels
    mock_target = torch.randn(10, 1, 100)

    loss = pino_loss(mock_prediction, mock_target)
    print(f"Calculated Total PINO Loss: {loss.item():.5f}")