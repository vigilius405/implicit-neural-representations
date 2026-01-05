"""
Visualization and plotting utilities for implicit neural representations.
"""

import torch
import matplotlib.pyplot as plt
from data_utils import get_mgrid


def visualize_training_progress(model_output, near_cell, far_cell, wide_area, img_dim, save_path=None):
    """
    Visualize training progress with multiple views.
    
    Args:
        model_output: Model output on training data
        near_cell: Output on nearby cell
        far_cell: Output on far cell
        wide_area: Output on wide area
        img_dim: Image dimension
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(model_output.cpu().view(img_dim, img_dim).detach().numpy())
    axes[0].set_title('Training Region')
    axes[1].imshow(near_cell.cpu().view(img_dim, img_dim).detach().numpy())
    axes[1].set_title('Near Cell')
    axes[2].imshow(far_cell.cpu().view(img_dim, img_dim).detach().numpy())
    axes[2].set_title('Far Cell')
    axes[3].imshow(wide_area.cpu().view(img_dim, img_dim).detach().numpy())
    axes[3].set_title('Wide Area')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_single_output(model, coords, img_dim, title='Model Output', save_path=None, device='cuda'):
    """
    Plot a single model output.
    
    Args:
        model: Trained model
        coords: Input coordinates
        img_dim: Image dimension
        title: Plot title
        save_path: Optional path to save the figure
        device: Device to run on
    """
    with torch.no_grad():
        if torch.cuda.is_available() and device == 'cuda':
            coords = coords.cuda()
        coords = torch.unsqueeze(coords, 0)
        model_out, _ = model(coords)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(model_out.cpu().view(img_dim, img_dim).numpy())
        ax.set_title(title)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def plot_out_of_bounds_grid(model, img_dim, grid_size=3, magnif=1, save_path=None, device='cuda'):
    """
    Plot a grid of out-of-bounds predictions.
    
    Args:
        model: Trained model
        img_dim: Image dimension
        grid_size: Size of the grid (e.g., 3 means 3x3 grid)
        magnif: Magnification factor
        save_path: Optional path to save the figure
        device: Device to run on
    """
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    offset = grid_size // 2
    
    for i in range(grid_size):
        for j in range(grid_size):
            xcell_idx = i - offset
            ycell_idx = j - offset
            
            with torch.no_grad():
                out_of_range_coords = get_mgrid(img_dim, 2) * magnif
                out_of_range_coords[:, 0] = out_of_range_coords[:, 0] + int(xcell_idx * img_dim)
                out_of_range_coords[:, 1] = out_of_range_coords[:, 1] + int(ycell_idx * img_dim)
                if torch.cuda.is_available() and device == 'cuda':
                    out_of_range_coords = out_of_range_coords.cuda()
                out_of_range_coords = torch.unsqueeze(out_of_range_coords, 0)
                model_out, _ = model(out_of_range_coords)
                
                axes[i, j].imshow(model_out.cpu().view(img_dim, img_dim).detach().numpy())
                axes[i, j].set_title(f'({xcell_idx}, {ycell_idx})')
                axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_loss_comparison(results, save_path=None):
    """
    Plot loss comparison across different models.
    
    Args:
        results: List of evaluation results [modinfo, regular_loss, near_loss, far_loss, ...]
        save_path: Optional path to save the figure
    """
    if not results:
        print("No results to plot")
        return
    
    model_names = [r[0] for r in results]
    regular_losses = [r[1].item() if torch.is_tensor(r[1]) else r[1] for r in results]
    near_losses = [r[2].item() if torch.is_tensor(r[2]) else r[2] for r in results]
    far_losses = [r[3].item() if torch.is_tensor(r[3]) else r[3] for r in results]
    
    x = range(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar([i - width for i in x], regular_losses, width, label='Regular Loss')
    ax.bar(x, near_losses, width, label='Near Loss')
    ax.bar([i + width for i in x], far_losses, width, label='Far Loss')
    
    ax.set_ylabel('MSE Loss')
    ax.set_title('Model Loss Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
