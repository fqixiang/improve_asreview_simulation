# ASReview Simulation

This repository contains scripts and resources for running simulations and analyses using the ASReview library. It supports two benchmarks: [**IMPROVE**](https://doi.org/10.1016/j.csbj.2024.10.026) and **SYNERGY**, with flexible filtering options for open access status and abstract length.


## Setup

1. **Install dependencies** (using uv):
   ```bash
   uv sync
   ```

2. **Prepare data**: Place datasets in appropriate folders:
   - IMPROVE: `data/improve/<dataset>/<dataset>.xlsx`
   - SYNERGY: `data/synergy/<dataset>/labels.csv`

3. **Run simulations**:
   ```bash
   python src/simulate.py [options]
   python src/tabulate_and_plot.py [options]
   ```

## Usage Examples

### Basic Simulation
```bash
uv run src/simulate.py --benchmark synergy --dataset Abgaz_2023 --model elas_u4
```

### With Filtering
```bash
uv run src/simulate.py \
  --benchmark synergy \
  --dataset Abgaz_2023 \
  --model elas_h3 \
  --oa_status True \
  --abstract_min_length 100 \
  --n_random_cycles 10
```

### Test Multiple Configurations
```bash
./test_configurations.sh
```
This tests all combinations of models, OA statuses, and abstract lengths on the Abgaz_2023 dataset.



## Project Structure

```
improve_asreview_simulation/
├── data/                           # Dataset storage
│   ├── improve/                    # IMPROVE benchmark datasets
│   │   └── <dataset>/             # Each dataset in its own folder
│   │       └── <dataset>.xlsx     # Dataset file
│   └── synergy/                   # SYNERGY benchmark datasets
│       └── <dataset>/             # Each dataset in its own folder
│           └── labels.csv         # Dataset file with labels
├── embeddings/                    # Pre-computed embeddings cache
│   ├── improve/
│   │   └── <dataset>/
│   │       ├── elas_h3_embeddings.parquet
│   │       └── elas_l2_embeddings.parquet
│   └── synergy/
│       └── <dataset>/
│           ├── elas_h3_embeddings.parquet
│           └── elas_l2_embeddings.parquet
├── results/                       # Simulation results
│   ├── <benchmark>/
│   │   └── <dataset>/
│   │       └── <n_pos>_pos_prior(s)/
│   │           └── <n_neg>_neg_prior(s)/
│   │               └── <oa_status>/    # all/open/closed
│   │                   ├── <model>_abs<length>_results.csv
│   │                   └── <model>_abs<length>_recall_vs_proportion.png
│   └── simulation_summary.csv     # Aggregated summary statistics
├── src/                           # Source code
│   ├── simulate.py               # Main simulation script
│   ├── tabulate_and_plot.py      # Results analysis and visualization
│   └── utils.py                  # Helper functions
├── logs/                         # Log files from batch runs
├── job.sh                        # Batch job script for HPC
├── test_configurations.sh        # Test script for multiple configurations
├── pyproject.toml               # Project dependencies
└── README.md                    # This file
```

## Key Components

### Scripts

#### `simulate.py`
Main simulation script with the following parameters:
- `--benchmark`: Benchmark to use (`improve` or `synergy`)
- `--dataset`: Dataset name
- `--model`: ASReview model (`elas_u4`, `elas_u3`, `elas_l2`, `elas_h3`)
- `--n_pos_priors`: Number of positive priors (default: 1)
- `--n_neg_priors`: Number of negative priors (default: 1)
- `--n_random_cycles`: Number of random cycles to run (default: 10)
- `--oa_status`: Filter by open access (`True`, `False`, or `None` for all) - SYNERGY only
- `--abstract_min_length`: Minimum abstract length (default: `None` for no filter)
- `--transformer_batch_size`: Batch size for embedding computation (default: 32)
- `--initial_random_seed`: Initial random seed (default: 42)

#### `tabulate_and_plot.py`
Generates summary statistics and recall plots. Uses the same parameters as `simulate.py` to locate the correct results.

#### `utils.py`
Contains helper functions:
- `model_configurations`: Configuration for all ASReview models
- `n_query_extreme()`: Adaptive query size based on dataset size
- `get_abstract_length()`: Language-aware abstract length calculation
- `pad_labels()`: Label padding for metrics calculation

### Models

Four models are available:
- **elas_u4**: SVM with TF-IDF (best overall)
- **elas_u3**: Naive Bayes with TF-IDF
- **elas_l2**: SVM with multilingual-e5-large embeddings
- **elas_h3**: SVM with mxbai-embed-large embeddings

### Benchmarks

#### IMPROVE
- Dataset files: `data/improve/<dataset>/<dataset>.xlsx`
- Contains curated medical datasets
- Uses `openalex_id` as identifier

#### SYNERGY
- Dataset files: `data/synergy/<dataset>/labels.csv`
- Includes metadata like `is_open_access`, `language`, etc.
- Uses `openalex_id` as identifier
- Supports OA status filtering

## Filtering Features

### Open Access Status (SYNERGY only)
Filter datasets by open access availability:
- `--oa_status True`: Only open access papers
- `--oa_status False`: Only closed access papers
- `--oa_status None` or omitted: All papers

### Abstract Length
Filter by minimum abstract length:
- For space-separated languages (English, etc.): counts words
- For non-space languages (Chinese, Japanese, Korean, Thai, etc.): counts characters
- `--abstract_min_length 100`: Minimum 100 words/characters
- `--abstract_min_length None` or omitted: No filtering

## Embeddings Cache

For models `elas_l2` and `elas_h3`:
- Embeddings are computed once for the entire dataset
- Cached in `embeddings/<benchmark>/<dataset>/`
- Reused across different filter combinations
- Significantly speeds up subsequent runs

## Output Files

### Results CSV
Located at: `results/<benchmark>/<dataset>/<priors>/<oa_status>/<model>_abs<length>_results.csv`

Contains columns:
- `record_id`: Record identifier
- `label`: True label (0 or 1)
- `seed`: Random seed used
- `total_n_records`: Total records in filtered dataset
- `n_training_records`: Number of records screened
- `total_n_relevant_records`: Total relevant records

### Recall Plot
Located at: `results/<benchmark>/<dataset>/<priors>/<oa_status>/<model>_abs<length>_recall_vs_proportion.png`

Shows recall curves for all random cycles.

### Summary Table
Located at: `results/simulation_summary.csv`

Aggregated statistics with columns:
- `dataset`, `model`, `oa_status`, `min_abstract_words`
- `total_n_records`, `total_n_inclusions`
- `loss_mean`, `loss_sd`, `ndcg_mean`, `ndcg_sd`
- `n_seeds`


## Requirements
- Python 3.11+
- pandas
- numpy
- asreview
- sentence-transformers (for elas_l2 and elas_h3)
- matplotlib
- seaborn
- pyarrow (for parquet support)

## Grant
The IMPROVE project is an Innovative Health Initiative project that has been granted by the European Commission under Grant agreement ID:101132847.

## Contact
For questions or issues, please contact the project maintainer.
