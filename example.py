"""
Quick example script demonstrating basic usage.
"""

import torch
from models import Flex_Model
from data_utils import PatternFitting, get_mgrid
from torch.utils.data import DataLoader
from train import train_siren
from plotting import plot_out_of_bounds_grid
import matplotlib.pyplot as plt


def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check if pattern file exists
    pattern_file = 'images/checker1.png'
    try:
        # Create dataset
        print(f"\nLoading pattern: {pattern_file}")
        dataset = PatternFitting(pattern_file, sidelen=128, selected_channel=0)
        dataloader = DataLoader(dataset, batch_size=1)
        
        # Create SIREN model
        print("Creating SIREN model...")
        model = Flex_Model(
            input_dim=2,
            output_dim=1,
            hidden_dim=256,
            nhidden=3,
            activation='sin'
        ).to(device)
        
        # Train
        print("Training model for 300 steps...")
        train_siren(model, dataloader, total_steps=300, img_dim=128, 
                   steps_til_summary=100, lr=1e-4, device=device)
        
        # Visualize out-of-bounds behavior
        print("\nGenerating out-of-bounds visualization...")
        plot_out_of_bounds_grid(model, img_dim=128, grid_size=3, 
                               save_path='example_output.png', device=device)
        
        print("\nDone! Check 'example_output.png' for results.")
        
    except FileNotFoundError:
        print(f"\nError: Pattern file '{pattern_file}' not found.")
        print("Please make sure you have pattern image files in the 'images/' directory.")
        print("\nYou can create a simple test pattern by running:")
        print("  mkdir -p images")
        print("  python -c \"from PIL import Image; import numpy as np; img = np.zeros((128, 128)); img[::16, :] = 255; img[:, ::16] = 255; Image.fromarray(img.astype(np.uint8)).save('images/checker1.png')\"")


if __name__ == '__main__':
    main()
