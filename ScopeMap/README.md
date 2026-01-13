# ScopeMap

ScopeMap is a molecular sampling toolkit that implements iterative Centroidal Voronoi Tessellation (CVT) sampling for chemical space exploration and molecular design optimization.

## Overview

This module provides an automated pipeline for intelligent molecular sampling using machine learning-guided approaches. It combines molecular descriptor calculation, iterative sampling strategies, and user-guided labeling to efficiently explore chemical space and identify promising molecular candidates.

## Core Components

### 1. Main Sampling Engine (`sampling.py`)
The primary interface for the sampling workflow with command-line operations:
- **Initialization**: Calculate molecular descriptors and prepare data structure
- **Iterative Sampling**: Perform weighted CVT sampling with user feedback integration
- **Result Management**: Track and finalize sampling results

### 2. Descriptor Generation (`desc_gen_util.py`)
Molecular descriptor calculation using RDKit:
- **Morgan Fingerprints**: Circular fingerprints with 4 radius and 1024 bits
- **MACCS Keys**: 166-bit structural keys for molecular representation
- **Combined Descriptors**: Concatenated fingerprint representations for enhanced molecular encoding

## Installation

### Requirements
```bash
pip install numpy pandas scikit-learn rdkit-pypi xgboost optuna matplotlib seaborn
```

### Dependencies
- **RDKit**: Molecular descriptor calculation and SMILES processing
- **scikit-learn**: Distance calculations and machine learning utilities
- **XGBoost**: Optional machine learning classifier for sampling optimization
- **NumPy/Pandas**: Data manipulation and numerical computations

## Usage

### 1. Initialize Sampling
```bash
python sampling.py init <csv_file>
```
- Input CSV must contain a `smiles` column with SMILES strings
- Calculates molecular descriptors automatically
- Creates `./itr/` directory and initial data files

### 2. Iterative Sampling
```bash
python sampling.py start [sample_num]
```
- Performs CVT sampling (default: 10 points per iteration)
- Generates `samples.csv` with sampled molecules for user evaluation
- Creates iteration files in `./itr/` directory

### 3. Manual Labeling
Edit `samples.csv` to update ScreenLabel column:
- `Pending` → `Sampled` (positive samples)
- `Pending` → `Excluded` (negative samples)

### 4. Finalize Results
```bash
python sampling.py stop
```
- Consolidates all sampling results
- Generates final `all_sampled.csv` and `all_excluded.csv` files

## File Structure

```
ScopeMap/
├── sampling.py              # Main sampling interface
├── sampling_util.py         # CVT algorithms and utilities
├── desc_gen_util.py         # Molecular descriptor generation
├── test_smiles.csv          # Example input data
├── itr/                     # Iteration data directory
│   ├── labeled_points_itr0.csv
│   ├── labeled_points_itr1.csv
│   └── sampled_points_itr*.csv
├── samples.csv              # Current iteration samples (for labeling)
├── all_sampled.csv          # Historical positive samples
└── all_excluded.csv         # Historical negative samples
```

## Key Features

### Intelligent Sampling
- **CVT Algorithm**: Optimal spatial distribution in molecular descriptor space
- **Weighted Sampling**: Incorporates feedback from previous iterations
- **Repulsion Forces**: Avoids resampling similar excluded molecules

### Flexible Input/Output
- **Multiple Identifier Support**: Works with `smiles`, `SMILES`, `Index`, or `index` columns
- **Descriptor Agnostic**: Automatically handles feature vs. non-feature columns
- **Iteration Tracking**: Complete history preservation for reproducibility

### User-Friendly Workflow
- **Simplified Interface**: Command-line operations with clear feedback
- **Manual Control**: User-guided labeling for domain expertise integration
- **Error Handling**: Robust validation and informative error messages

## Algorithm Details

### CVT Sampling Process
1. **Initialization**: Random seed selection for reproducible results
2. **Voronoi Tessellation**: Partition molecular space into regions
3. **Centroid Calculation**: Find optimal representatives for each region
4. **Iterative Refinement**: Minimize energy function until convergence

### Weighted Enhancement
- **Repulsion Model**: Inverse-square distance repulsion from excluded points
- **Energy Optimization**: Gradient descent with adaptive learning rate
- **Multi-metric Support**: Flexible distance calculations for different molecular representations

## Example Workflow

```bash
# 1. Initialize with molecular data
python sampling.py init test_smiles.csv

# 2. Start sampling (will create samples.csv)
python sampling.py start 5

# 3. Edit samples.csv - change 'Pending' to 'Sampled' or 'Excluded'

# 4. Continue sampling
python sampling.py start 5

# 5. Repeat steps 3-4 as needed

# 6. Finalize results
python sampling.py stop
```

## Output Files

- **`all_sampled.csv`**: All molecules labeled as promising candidates
- **`all_excluded.csv`**: All molecules labeled as unsuitable
- **`samples_final.csv`**: Final iteration samples archive
- **`fp_spoc_morgan41024_Maccs_*.csv`**: Generated molecular descriptors

## Performance Characteristics

- **Convergence**: Typically converges within 100-500 iterations
- **Scalability**: Handles datasets from hundreds to tens of thousands of molecules
- **Memory Efficient**: Streaming processing for large molecular databases
- **Reproducible**: Fixed random seeds ensure consistent results

## Advanced Configuration

The sampling algorithm supports several parameters for fine-tuning:
- **Sample Size**: Number of molecules per iteration
- **Distance Metrics**: Euclidean, Manhattan, or Cosine distance
- **Repulsion Strength**: Control avoidance of excluded regions
- **Convergence Tolerance**: Balance between accuracy and speed

## License

This module is designed for academic and research purposes in computational chemistry and molecular design.