# Configuration Files

This directory contains configuration files for different experiments.

## Usage

Run experiments using configuration files:

```bash
# Use a specific config
python main.py --config configs/hypercube_hurricane.yaml

# Override config parameters
python main.py --config configs/hypercube_hurricane.yaml --k 10 --save
```

## Config Structure

Each config file can specify:
- Dataset settings
- Model settings  
- Retrieval settings
- Evaluation settings
- Output settings

See example configs for reference.