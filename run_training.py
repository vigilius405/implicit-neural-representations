"""
Main training script for implicit neural representations.

This script trains models on image patterns and evaluates their ability to
conserve patterns out of bounds.
"""

import torch
from torch.utils.data import DataLoader
import argparse
from models import Flex_Model
from data_utils import PatternFitting
from train import train_siren, eval_model
from plotting import plot_out_of_bounds_grid, plot_loss_comparison


def main():
    parser = argparse.ArgumentParser(
        description='Train implicit neural representation models on image patterns'
    )
    parser.add_argument(
        '--patterns', 
        nargs='+', 
        default=['images/checker1.png', 'images/checker2.png', 'images/checker3.png', 
                 'images/tartan1.png', 'images/tartan2.png', 'images/tartan3.png'],
        help='List of pattern image files to train on (can be relative or absolute paths)'
    )
    parser.add_argument(
        '--model-types', 
        nargs='+', 
        default=['relu', 'tanh', 'rbf', 'sin', 'ffn', 'fkan'],
        help='Model activation types to train'
    )
    parser.add_argument(
        '--sidelen', 
        type=int, 
        default=128,
        help='Image side length (resolution)'
    )
    parser.add_argument(
        '--hidden-dim', 
        type=int, 
        default=256,
        help='Hidden layer dimension'
    )
    parser.add_argument(
        '--nhidden', 
        type=int, 
        default=3,
        help='Number of hidden layers'
    )
    parser.add_argument(
        '--freq', 
        type=int, 
        default=30,
        help='Frequency for sinusoidal models'
    )
    parser.add_argument(
        '--steps', 
        type=int, 
        default=301,
        help='Training steps (simpler patterns)'
    )
    parser.add_argument(
        '--steps-complex', 
        type=int, 
        default=601,
        help='Training steps (complex patterns)'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--eval-only', 
        action='store_true',
        help='Only run evaluation, skip training'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    
    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}")
    
    # Load data
    print("Loading image patterns...")
    img_pattern_fits = []
    img_dataloaders = []
    for pattern_file in args.patterns:
        img_pattern_fit = PatternFitting(pattern_file, args.sidelen, selected_channel=0)
        img_dataloader = DataLoader(img_pattern_fit, batch_size=1, pin_memory=True, num_workers=0)
        img_pattern_fits.append(img_pattern_fit)
        img_dataloaders.append(img_dataloader)
    
    # For FKAN models, use smaller resolution
    fkan_pattern_fits = []
    fkan_dataloaders = []
    for pattern_file in args.patterns:
        fkan_pattern_fit = PatternFitting(pattern_file, args.sidelen // 2, selected_channel=0)
        fkan_dataloader = DataLoader(fkan_pattern_fit, batch_size=1, pin_memory=True, num_workers=0)
        fkan_pattern_fits.append(fkan_pattern_fit)
        fkan_dataloaders.append(fkan_dataloader)
    
    # Training loop
    final_losses = []
    models = []
    
    for mtype in args.model_types:
        for i, img in enumerate(img_dataloaders):
            torch.cuda.empty_cache()
            
            print(f'\n{"="*60}')
            print(f'Training model {mtype} on data {args.patterns[i]}')
            print(f'{"="*60}\n')
            
            # Adjust parameters based on model type
            indim = 256 if mtype == 'ffn' else 2
            model = Flex_Model(
                indim, 1, 
                hidden_dim=args.hidden_dim, 
                nhidden=args.nhidden, 
                freq=args.freq, 
                activation=mtype
            ).to(device)
            
            # Use appropriate dataloader and sidelen for FKAN
            current_dataloader = fkan_dataloaders[i] if mtype == 'fkan' else img
            sidelen = args.sidelen // 2 if mtype == 'fkan' else args.sidelen
            
            # More steps for complex patterns (last 3)
            nsteps = args.steps_complex if i >= 3 else args.steps
            steps_til_summary = (nsteps - 1) // 3
            
            # Train the model
            train_siren(
                model, current_dataloader, 
                total_steps=nsteps, 
                img_dim=sidelen, 
                steps_til_summary=steps_til_summary, 
                lr=args.lr,
                device=device
            )
            
            # Save model
            models.append((f'{mtype} / {args.patterns[i]}', model.to('cpu')))
            
            # Evaluate model
            _, ground_truth = next(iter(current_dataloader))
            ground_truth = ground_truth.to(device)
            ground_truth = torch.squeeze(ground_truth)
            
            losses = eval_model(
                model.to(device), sidelen, ground_truth, 
                f'{mtype} / {args.patterns[i]}', 
                near_dist=1, far_dist=2, 
                verbose=True,
                device=device
            )
            final_losses.append(losses)
            
            # Visualize out-of-bounds behavior
            print(f"\nGenerating out-of-bounds visualization for {mtype} / {args.patterns[i]}...")
            plot_out_of_bounds_grid(
                model.to(device), sidelen, grid_size=3, magnif=1,
                save_path=f'outputs/{mtype}_{args.patterns[i]}_oob_grid.png',
                device=device
            )
    
    # Plot final loss comparison
    print("\n" + "="*60)
    print("Generating loss comparison plot...")
    print("="*60)
    plot_loss_comparison(final_losses, save_path='outputs/loss_comparison.png')
    
    print("\nTraining complete!")
    print(f"Total models trained: {len(models)}")


if __name__ == '__main__':
    main()
