# GraphSAGE Implementation - Quick Start Guide

This guide will help you get started with the GraphSAGE implementation quickly.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (optional, for cloning)

## Installation

### Option 1: Quick Start Script (Recommended)

```bash
# Run the quick start script
python scripts/quick_start.py --action install
```

### Option 2: Manual Installation

```bash
# Install the package
pip install -e .[dev]

# Or install from requirements
pip install -r requirements.txt
```

## Quick Training

### Train on Cora Dataset (Default)

```bash
python scripts/train.py
```

### Train on Different Dataset

```bash
python scripts/train.py --dataset citeseer --epochs 100
```

### Available Datasets

- `cora` - Citation network (default)
- `citeseer` - Citation network
- `pubmed` - Citation network
- `reddit` - Social network
- `amazon_photo` - Co-purchase network
- `amazon_computers` - Co-purchase network
- `coauthor_cs` - Co-authorship network
- `coauthor_physics` - Co-authorship network
- `synthetic` - Synthetic graph for testing

## Interactive Demo

### Launch Demo

```bash
# Using the demo script
python scripts/run_demo.py

# Or directly with Streamlit
streamlit run demo/app.py
```

The demo will open in your browser at `http://localhost:8501`

### Demo Features

- **Dataset Exploration**: View dataset statistics and class distributions
- **Node Analysis**: Analyze individual nodes and their neighbors
- **Graph Visualization**: Interactive network visualization
- **Model Predictions**: Real-time prediction analysis
- **Embedding Visualization**: t-SNE plots of learned embeddings

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

## Configuration

### Using Different Configurations

```bash
# Use synthetic data configuration
python scripts/train.py --config configs/synthetic.yaml

# Use large graphs configuration
python scripts/train.py --config configs/large_graphs.yaml
```

### Custom Configuration

Create your own configuration file:

```yaml
# configs/my_config.yaml
model:
  hidden_channels: [128, 64]
  num_layers: 3
  aggregator: "max"

training:
  epochs: 150
  lr: 0.005
  optimizer: "adamw"

data:
  dataset: "pubmed"
```

Then use it:

```bash
python scripts/train.py --config configs/my_config.yaml
```

## Advanced Usage

### Command Line Overrides

```bash
python scripts/train.py \
  --dataset reddit \
  --epochs 200 \
  --lr 0.005 \
  --seed 123
```

### Programmatic Usage

```python
from src.models import GraphSAGE
from src.data import load_dataset
from src.train import train_model

# Load data
data, _ = load_dataset("cora")

# Create model
model = GraphSAGE(
    in_channels=data.num_node_features,
    hidden_channels=[64, 32],
    out_channels=data.num_classes,
)

# Train (simplified)
# ... training code ...
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   python scripts/train.py --device cpu
   ```

2. **Import Errors**
   ```bash
   # Reinstall PyTorch Geometric
   pip uninstall torch-geometric
   pip install torch-geometric
   ```

3. **Dataset Download Issues**
   ```bash
   # Use synthetic data for testing
   python scripts/train.py --dataset synthetic
   ```

### Performance Tips

1. **Use GPU**: Ensure CUDA is available
2. **Reduce Model Size**: Use smaller hidden dimensions
3. **Fewer Epochs**: Start with fewer epochs for testing

## Project Structure

```
graphsage-implementation/
├── src/                    # Source code
│   ├── models/            # GraphSAGE implementations
│   ├── data/              # Data utilities
│   ├── train/             # Training utilities
│   ├── eval/              # Evaluation metrics
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training scripts
├── demo/                  # Interactive demo
├── tests/                 # Unit tests
└── README.md              # Full documentation
```

## Next Steps

1. **Explore the Demo**: Launch the interactive demo to understand the model
2. **Try Different Datasets**: Experiment with various graph datasets
3. **Modify Configurations**: Adjust model parameters and training settings
4. **Read the Full README**: Check `README.md` for comprehensive documentation
5. **Run Tests**: Ensure everything works correctly
6. **Contribute**: Add new features or improvements

## Support

- Check the full `README.md` for detailed documentation
- Run tests to verify installation: `pytest tests/ -v`
- Use the demo to explore the model interactively
- Check GitHub issues for common problems

Happy learning with GraphSAGE! 🚀
