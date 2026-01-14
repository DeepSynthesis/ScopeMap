# ScopeMap: An AI-Assisted, Human-in-the-Loop Workflow for Mapping Reaction Scope and Boundaries

ScopeMap, an iterative, human-in-the-loop workflow designed to efficiently map functional limits rather than merely maximizing performance. Leveraging a modified Centroidal Voronoi Tessellation (CVT) algorithm with a dynamic Geometric Repulsion Potential, ScopeMap transforms negative experimental feedback into geometric constraints, actively steering sampling toward unexplored frontiers.Furthermore, we establish the U-Score and R-Score—metrics derived from Spatial Entropy and mean squared distance (MSD)—to provide a standardized framework for quantifying sampling uniformity and representativeness.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Core Features](#core-features)
- [Installation Requirements](#installation-requirements)
- [Quick Start](#quick-start)
- [Example Cases](#example-cases)
- [Output Files](#output-files)
- [Web Service](#web-service)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## Project Overview

This toolkit consists of two main modules:
- **ScopeMap**: Substrate sampling toolkit implementing iterative Centroidal Voronoi Tessellation (CVT) sampling
- **SubstrateScore**: Sampling quality evaluation toolkit with multiple sampling algorithms and evaluation metrics

## Project Structure

```
upload_version/
├── ScopeMap/                    # Substrate sampling module
│   ├── sampling.py              # Main sampling interface
│   ├── sampling_util.py         # CVT algorithms and utilities
│   ├── desc_gen_util.py         # Molecular descriptor generation
│   └── README.md               # Detailed ScopeMap documentation
├── SubstrateScore/             # Sampling evaluation module
│   ├── evaluate.py             # Main evaluation script
│   ├── sampling.py             # Various sampling algorithm implementations
│   ├── score_utils.py          # Evaluation metric calculations
│   ├── desc_gen_util.py        # Molecular descriptor generation utilities
│   └── README.md               # Detailed SubstrateScore documentation
├── Examples/                   # Examples and test cases
│   ├── Aldol/                  # Aldol reaction examples
│   ├── Article_tests/          # Literature test cases
│   │   ├── Alcohols/           # Alcohol substrate tests
│   │   ├── OxoestersOxoamides&Styrenes/ # Oxoesters/oxoamides and styrenes tests
│   │   └── Thiols/             # Thiol compound tests
│   ├── Co_catalytic/          # Co-catalytic reaction examples
│   └── Sampling_eval/         # Sampling method evaluations
│       ├── Model_eval/        # Model evaluation
│       ├── Pareto_frontier/   # Pareto frontier analysis
│       └── Yield_distribution/ # Yield distribution analysis
└── Data/                      # Data files
    ├── Aldol/                 # Aldol reaction data
    └── Co_catalytic/          # Co-catalytic reaction data
```

## Core Features

### ScopeMap - Intelligent Sampling
- **Iterative CVT Sampling**: Optimal spatial distribution in molecular descriptor space
- **User-Guided Labeling**: Manual annotation integrating domain expertise
- **Repulsion Forces**: Avoids resampling similar excluded molecules
- **Command-Line Interface**: Simplified command-line operations with clear feedback

### SubstrateScore - Quality Assessment
- **U-Score**: Entropy-based sampling uniformity evaluation
- **R-Score**: Distance-based sampling representativeness evaluation

## Installation Requirements

```bash
pip install numpy pandas scikit-learn scipy rdkit-pypi
```

### Dependencies
- **RDKit**: Molecular descriptor calculation and SMILES processing
- **scikit-learn**: Distance calculations and machine learning utilities
- **NumPy/Pandas**: Data manipulation and numerical computations
- **SciPy**: Scientific computing

## Quick Start

### 1. Using ScopeMap for Sampling

```bash
# Initialize sampling (input CSV must contain 'smiles' column)
cd ScopeMap
python sampling.py init <csv_file>

# Start iterative sampling
python sampling.py start 10

# Edit samples.csv - change 'Pending' to 'Sampled' or 'Excluded'

# Continue sampling iterations...

# Finalize sampling
python sampling.py stop
```

### 2. Using SubstrateScore for Evaluation

```bash
cd SubstrateScore
python evaluate.py \
    --substrate-file test_space.csv \
    --experimental-file test_exp.csv \
    --distance-metric euclidean \
    --k-neighbors 5 \
    --random-seed 42
```

## Example Cases

This work includes multiple real chemical reaction examples:

1. **Aldol Reaction**: Aldol condensation reaction substrate sampling
2. **Alcohol Selectivity**: Sampling evaluation for different alcohol substrates
3. **Oxo Compounds**: Sampling analysis for oxoesters and oxoamides
4. **Thiol Compounds**: Sampling strategies for sulfur-containing substrates
5. **Co-catalytic Reactions**: Substrate optimization for multi-component cobalt catalytic reactions

Each example includes complete data files, sampling scripts, evaluation results and original figures.

## Output Files

- **Sampling Results**:
  - `all_sampled.csv`: All molecules labeled as promising candidates
  - `all_excluded.csv`: All molecules labeled as unsuitable
  - `samples_final.csv`: Final iteration samples archive

- **Evaluation Reports**:
  - Entropy and MSD values for each sampling method
  - Experimental sampling quality assessment
  - U-Score and R-Score comprehensive ratings

## Web Service

The web service is currently under deployment. Please stay tuned for the upcoming online experience.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Issues and Pull Requests are welcome to improve the project.

## Contact

For questions or suggestions, please contact through Email lijiawei24@mails.tsinghua.edu.cn or GitHub Issues.