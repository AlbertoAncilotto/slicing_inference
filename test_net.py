import torch
import torch.nn as nn
import torch.nn.functional as F
from get_receptive_field import compute_receptive_field
from slicing import learn_slice_flip

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
    
    def forward_shapes(self, x):
        print('Encoder shapes:')
        for layer in self.encoder:
            x = layer(x)
            print(x.shape)
            
        # print('Neck shapes:')
        # for layer in self.neck:
        #     x = layer(x)
        #     print(x.shape)
            
        # print('Decoder shapes:')
        # for layer in self.decoder:
        #     x = layer(x)
        #     print(x.shape)
        return x        
                
        

if __name__ == "__main__":
    model = SimpleAutoencoder()
    
    # Compute receptive field - choose target layer based on receptive field and input resolution
    compute_receptive_field(model, None)
    module = model
    input_shape = (1, 3, 160, 160)
    
    slices = []
    for slice_id in range(4):
        # compute the slicing input/ output based on the set accuracy threshold
        input_lines, output_lines = learn_slice_flip(module, num_slices=4, slice_id=slice_id, input_shape=input_shape, threshold=1e-6)
        print('////////////////////////////////////////')
        print(f"Slice {slice_id}: {input_lines} -> {output_lines}")
        print('////////////////////////////////////////')
        slices.append((input_lines, output_lines))
        
    # Full model forward pass
    example_input = torch.rand(input_shape)
    example_output = module(example_input)
    
    # Sliced model forward pass
    slices_output = []
    for slice in slices:
        input_lines, output_lines = slice
        input_tensor = example_input[:,:,input_lines[0]:input_lines[1],:]
        output_tensor = module(input_tensor)[:,:,output_lines[0]:output_lines[1],:]
        slices_output.append(output_tensor)
    full_output_from_slices = torch.cat(slices_output, dim=2)
    
    # Compare outputs
    print(f"Full output from slices: {full_output_from_slices.shape}")
    print(f"Error: {torch.abs(full_output_from_slices - example_output).max()}")
    print(f"MSE: {torch.nn.functional.mse_loss(full_output_from_slices, example_output)}")
    
