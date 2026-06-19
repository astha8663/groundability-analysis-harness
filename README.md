# Groundability Analysis Harness

Groundability analysis utilities and NTU-60 part-level semantic embedding artifacts.The pipeline consists of a baseline zero-shot classifier, part-level groundability analysis, correlation analysis, and artifact-level sanity check

## Project Structure

```text
groundability-analysis-harness/ 
│ 
├── data/ 
│ ├── features/ 
│ ├── embeddings/ 
│ ├── splits/ 
│ └── descriptions.json 
│ 
├── src/ 
│ ├── load.py 
│ ├── baseline.py 
│ ├── groundability.py 
│ ├── analysis.py 
│ └── check_sanity.py 
│ 
├── tests/ 
│ ├── test_load.py 
│ └── test_groundability.py 
│ 
├── results/ 
├── figures/ 
├── notebooks/ 
│ └── groundability_analysis.ipynb 
│ 
├── requirements.txt 
└── README.md
```

## NTU-60 Part Embeddings

The `ntu60_parts/` directory contains per-class, per-body-part semantic embeddings generated using STAR anatomical descriptions and `sentence-transformers/stsb-bert-large`.
Each embedding matrix:
* corresponds to NTU-60 class ordering
* has shape `[60, 1024]`
* is L2-normalized


## Phase 1: Baseline Zero-Shot Classification

The baseline learns a linear projection from skeleton features to semantic embedding space using only seen classes.
Outputs:
overall accuracy
per-class accuracy
per-class error rates

Example:
python -m src.baseline \
    --seen rs55.npy \
    --unseen ru5.npy \
    --feature_dir shift_ntu60_5_r


## Phase 2: Groundability Analysis

Linear ridge probes are trained on seen classes to predict body-part semantic embeddings from skeleton features.
Outputs:
per-part validation scores
per-class groundability scores
mean aggregation
max aggregation

Example:
python -m src.groundability \
    --seen rs55.npy \
    --unseen ru5.npy \
    --feature_dir shift_ntu60_5_r


## Phase 3: Correlation Analysis

Groundability scores are compared against baseline error rates.
Outputs:
Spearman correlation
scatter plots
results tables

Example:
python -m src.analysis

Generated figures are saved to:
figures/

Generated tables are saved to:
results/


## Sanity Checks

Artifact-level validation is performed using:

python -m src.check_sanity

Checks include:
finite groundability scores
valid score ranges
baseline accuracy above chance
complete results tables
missing-value detection


## Tests

Run all tests:

pytest

Run specific test suites:

pytest tests/test_load.py
pytest tests/test_groundability.py

## Notebook Report

The final analysis notebook is located at:

notebooks/groundability_analysis.ipynb

The notebook reproduces:
results table visualization
48/12 scatter plot
Spearman correlation analysis
written summary of findings
