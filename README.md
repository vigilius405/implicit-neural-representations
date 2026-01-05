# Implicit Neural Representations

**CMSC 25500 Final Project: Evaluation of Out-of-Bounds Pattern Conservation in Implicit Neural Representation Networks**

This project evaluates how different implicit neural representation (INR) architectures preserve patterns when queried outside their training bounds. The implementation includes multiple activation types (sinusoidal, ReLU, tanh, RBF, Fourier features, and FKAN) inspired by recent research in the field.

## Project Structure

```
.
├── images/             # Pattern image files  
├── models.py           # Model definitions (Flex_Model and activation layers)
├── data_utils.py       # Data loading and preprocessing utilities
├── train.py            # Training and evaluation functions
├── plotting.py         # Visualization utilities
├── run_training.py     # Main training script
├── visualize.py        # Visualization script for trained models
├── example.py          # Quick start example
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- torchaudio
- numpy
- matplotlib
- Pillow
- scikit-image

### Setup

1. Clone the repository:
```bash
git clone https://github.com/vigilius405/implicit-neural-representations.git
cd implicit-neural-representations
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio numpy matplotlib pillow scikit-image
```

Or use a requirements file if created:
```bash
pip install -r requirements.txt
```

3. Create output directory:
```bash
mkdir -p outputs
```

## Usage

### Training Models

Train models on image patterns using the main training script:

```bash
python run_training.py
```

#### Training Options

- `--patterns`: List of pattern image files (default: images/checker1.png, images/checker2.png, images/checker3.png, images/tartan1.png, images/tartan2.png, images/tartan3.png)
- `--model-types`: Model activation types to train (default: relu, tanh, rbf, sin, ffn, fkan)
- `--sidelen`: Image resolution (default: 128)
- `--hidden-dim`: Hidden layer dimension (default: 256)
- `--nhidden`: Number of hidden layers (default: 3)
- `--freq`: Frequency for sinusoidal models (default: 30)
- `--steps`: Training steps for simple patterns (default: 301)
- `--steps-complex`: Training steps for complex patterns (default: 601)
- `--lr`: Learning rate (default: 1e-4)
- `--device`: Device to use, 'cuda' or 'cpu' (default: auto-detect)

#### Example: Train specific models

```bash
python run_training.py --model-types sin ffn --patterns images/checker1.png images/tartan1.png --steps 500
```

#### Example: Train on CPU

```bash
python run_training.py --device cpu
```

### Visualizing Results

Visualize out-of-bounds behavior of trained models:

```bash
python visualize.py --model-type sin --pattern images/checker1.png --grid-size 5
```

#### Visualization Options

- `--model-type`: Model activation type (required: relu, tanh, rbf, sin, ffn, fkan)
- `--pattern`: Pattern image file (required)
- `--model-path`: Path to saved model weights (optional)
- `--sidelen`: Image side length (default: 128)
- `--grid-size`: Grid size for visualization (default: 5)
- `--device`: Device to use (default: auto-detect)

### Using as a Library

You can also use the modules as a library in your own scripts:

```python
import torch
from models import Flex_Model
from data_utils import PatternFitting
from torch.utils.data import DataLoader
from train import train_siren, eval_model

# Create dataset
dataset = PatternFitting('images/pattern.png', sidelen=128, selected_channel=0)
dataloader = DataLoader(dataset, batch_size=1)

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Flex_Model(2, 1, hidden_dim=256, nhidden=3, activation='sin').to(device)

# Train
train_siren(model, dataloader, total_steps=300, img_dim=128, device=device)

# Evaluate
_, ground_truth = next(iter(dataloader))
ground_truth = ground_truth.to(device).squeeze()
results = eval_model(model, 128, ground_truth, 'sin/pattern', device=device)
```

## Model Types

The framework supports multiple activation/architecture types:

- **sin**: Sinusoidal activation (SIREN) - best for smooth signals
- **relu**: ReLU activation - standard baseline
- **tanh**: Tanh activation - smooth nonlinearity
- **rbf**: Radial basis functions - localized representations
- **ffn**: Fourier feature networks - frequency-based encoding
- **fkan**: Fourier Kolmogorov-Arnold Networks - Chebyshev polynomials

## References

This implementation is inspired by:

- **Sitzmann et al.**: [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/pdf/2006.09661) ([GitHub](https://github.com/vsitzmann/siren))
- **Tancik et al.**: [Fourier Features Let Networks Learn High Frequency Functions](https://arxiv.org/pdf/2006.10739) ([GitHub](https://github.com/tancik/fourier-feature-networks))
- **Mehrabian et al.**: [FKAN: Fourier Kolmogorov-Arnold Networks](https://arxiv.org/pdf/2409.09323) ([GitHub](https://github.com/Ali-Meh619/FKAN))

## License

This project is provided as-is for educational and research purposes.
