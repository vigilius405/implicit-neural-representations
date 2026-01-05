"""
Simple visualization script to plot saved model outputs.

Usage:
    python visualize.py --model-path <path_to_model> --pattern <pattern_file>
"""

import torch
import argparse
from models import Flex_Model
from data_utils import PatternFitting
from torch.utils.data import DataLoader
from plotting import plot_out_of_bounds_grid, plot_single_output


def main():
    parser = argparse.ArgumentParser(
        description='Visualize trained implicit neural representation models'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['relu', 'tanh', 'rbf', 'sin', 'ffn', 'fkan'],
        help='Model activation type'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        required=True,
        help='Pattern image file'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to saved model weights (optional, will create new model if not provided)'
    )
    parser.add_argument(
        '--sidelen',
        type=int,
        default=128,
        help='Image side length'
    )
    parser.add_argument(
        '--grid-size',
        type=int,
        default=5,
        help='Grid size for out-of-bounds visualization'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}")
    
    # Create model
    indim = 256 if args.model_type == 'ffn' else 2
    sidelen = args.sidelen // 2 if args.model_type == 'fkan' else args.sidelen
    
    model = Flex_Model(
        indim, 1,
        hidden_dim=256,
        nhidden=3,
        freq=30,
        activation=args.model_type
    ).to(device)
    
    # Load weights if provided
    if args.model_path:
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print("No model path provided. Using randomly initialized model.")
    
    # Generate visualizations
    print(f"\nGenerating out-of-bounds grid visualization...")
    plot_out_of_bounds_grid(
        model, sidelen,
        grid_size=args.grid_size,
        magnif=1,
        save_path=f'{args.model_type}_{args.pattern}_grid.png',
        device=device
    )
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
