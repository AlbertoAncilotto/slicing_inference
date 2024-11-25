import torch
import torch.nn as nn
from get_receptive_field import compute_receptive_field
from slicing import sliced_forward, learn_slices


class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.neck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), 
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  
            nn.Conv2d(16, 3, kernel_size=9, padding=4), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.neck(x)
        x = self.decoder(x)
        return x
      
                
        

if __name__ == "__main__":
    model = SimpleAutoencoder()
    
    # Compute receptive field - choose target layer based on receptive field and input resolution
    compute_receptive_field(model, None)
    module = model.encoder
    input_shape = (1, 3, 160, 160)
    
    slices = learn_slices(module, input_shape, num_slices=4, threshold=1e-6)
        
    # Full model forward pass
    example_input = torch.rand(input_shape)
    example_output = module(example_input)
    
    # Sliced forward pass
    full_output_from_slices = sliced_forward(module, slices, example_input)
    
    # Compare outputs
    print(f"Full output from slices: {full_output_from_slices.shape}")
    print(f"Error: {torch.abs(full_output_from_slices - example_output).max()}")
    print(f"MSE: {torch.nn.functional.mse_loss(full_output_from_slices, example_output)}")
    
