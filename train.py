"""
Training utilities for implicit neural representation models.
"""

import torch
import torch.nn.functional as F
from data_utils import get_mgrid


def train_siren(img_siren, dataloader, total_steps, img_dim, steps_til_summary=10, lr=1e-4, device='cuda'):
    """
    Train an implicit neural representation model.
    
    Args:
        img_siren: The model to train
        dataloader: DataLoader with training data
        total_steps: Number of training steps
        img_dim: Dimension of the image
        steps_til_summary: Frequency of progress logging
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
    """
    optim = torch.optim.Adam(lr=lr, params=img_siren.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.to(device), ground_truth.to(device)
    ground_truth = torch.squeeze(ground_truth)

    for step in range(total_steps):
        model_output, coords = img_siren(model_input)
        loss = F.mse_loss(torch.squeeze(model_output), ground_truth)

        if not step % steps_til_summary:
            near_cell = show_cell(img_siren, img_dim, xcell_idx=0, ycell_idx=1, magnif=1, just_return=True, device=device)
            near_loss = F.mse_loss(torch.squeeze(near_cell), ground_truth)
            far_cell = show_cell(img_siren, img_dim, xcell_idx=-10, ycell_idx=10, magnif=1, just_return=True, device=device)
            far_loss = F.mse_loss(torch.squeeze(far_cell), ground_truth)

            print(f"Step {step}, Total loss {loss:.6f}, Near loss {near_loss:.6f}, Far loss {far_loss:.6f}")

        optim.zero_grad()
        loss.backward()
        optim.step()


def show_cell(img_siren, img_dim=256, xcell_idx=0, ycell_idx=0, magnif=1, shift_factor=1, just_return=False, device='cuda'):
    """
    Generate output for a specific spatial cell (potentially out of bounds).
    
    Args:
        img_siren: Trained model
        img_dim: Image dimension
        xcell_idx: X-axis cell index
        ycell_idx: Y-axis cell index
        magnif: Magnification factor
        shift_factor: Shift factor for cell coordinates
        just_return: If True, return tensor without visualizing
        device: Device to run on
    
    Returns:
        Model output for the specified cell
    """
    with torch.no_grad():
        out_of_range_coords = get_mgrid(img_dim, 2) * magnif
        out_of_range_coords[:, 0] = out_of_range_coords[:, 0] + int(xcell_idx * shift_factor * img_dim)
        out_of_range_coords[:, 1] = out_of_range_coords[:, 1] + int(ycell_idx * shift_factor * img_dim)
        if torch.cuda.is_available() and device == 'cuda':
            out_of_range_coords = out_of_range_coords.cuda()
        out_of_range_coords = torch.unsqueeze(out_of_range_coords, 0)
        model_out, _ = img_siren(out_of_range_coords)
        if just_return:
            return model_out
        
        # If not just_return, visualization would happen here
        return model_out


def eval_square(img_siren, img_dim, ground_truth, near_dist=1, sf=1, device='cuda'):
    """
    Evaluate model on a square of cells at given distance.
    
    Args:
        img_siren: Trained model
        img_dim: Image dimension
        ground_truth: Ground truth tensor
        near_dist: Distance of cells from center
        sf: Shift factor
        device: Device to run on
    
    Returns:
        Tuple of (mean_loss, std_loss)
    """
    losses = []
    for cell in range(-near_dist, near_dist+1):
        c1 = show_cell(img_siren, img_dim, xcell_idx=-near_dist, ycell_idx=cell, magnif=1, 
                      shift_factor=sf, just_return=True, device=device)
        c2 = show_cell(img_siren, img_dim, xcell_idx=near_dist, ycell_idx=cell, magnif=1, 
                      shift_factor=sf, just_return=True, device=device)
        c3 = show_cell(img_siren, img_dim, xcell_idx=cell, ycell_idx=-near_dist, magnif=1, 
                      shift_factor=sf, just_return=True, device=device)
        c4 = show_cell(img_siren, img_dim, xcell_idx=cell, ycell_idx=near_dist, magnif=1, 
                      shift_factor=sf, just_return=True, device=device)
        losses = losses + [
            F.mse_loss(torch.squeeze(c1), ground_truth), 
            F.mse_loss(torch.squeeze(c2), ground_truth),
            F.mse_loss(torch.squeeze(c3), ground_truth), 
            F.mse_loss(torch.squeeze(c4), ground_truth)
        ]
    losses = torch.tensor(losses)
    return (torch.mean(losses), torch.std(losses))


def eval_model(img_siren, img_dim, ground_truth, modinfo, near_dist=1, far_dist=10, sf=1, verbose=True, device='cuda'):
    """
    Comprehensive model evaluation.
    
    Args:
        img_siren: Trained model
        img_dim: Image dimension
        ground_truth: Ground truth tensor
        modinfo: Model information string
        near_dist: Near distance for evaluation
        far_dist: Far distance for evaluation
        sf: Shift factor
        verbose: Whether to print results
        device: Device to run on
    
    Returns:
        List of evaluation metrics
    """
    regular_loss, regular_sd = eval_square(img_siren, img_dim, ground_truth, near_dist=0, sf=sf, device=device)
    near_loss, near_sd = eval_square(img_siren, img_dim, ground_truth, near_dist=near_dist, sf=sf, device=device)
    far_loss, far_sd = eval_square(img_siren, img_dim, ground_truth, near_dist=far_dist, sf=sf, device=device)
    
    if verbose:
        print(f"########## MODEL EVAL ##########\n"
              f"Model: {modinfo}\n"
              f"Regular loss: {regular_loss}, Dist {near_dist} tile loss: {near_loss},\n"
              f"Dist {far_dist} tile loss: {far_loss}\n"
              f"Regular SD: {regular_sd}, Dist {near_dist} tile SD: {near_sd},\n"
              f"Dist {far_dist} tile SD: {far_sd}\n"
              f"########## MODEL EVAL ##########")
    
    return [modinfo, regular_loss, near_loss, far_loss, regular_sd, near_sd, far_sd]
