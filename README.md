# Groundability Analysis Harness

Groundability analysis utilities and NTU-60 part-level semantic embedding artifacts.

## Project Structure

```text
groundability-analysis-harness/
│
├── data/
├── ntu60_parts/
├── verify.py
├── requirements.txt
└── README.md
```

## NTU-60 Part Embeddings

The `ntu60_parts/` directory contains per-class, per-body-part semantic embeddings generated using STAR anatomical descriptions and `sentence-transformers/stsb-bert-large`.

Each embedding matrix:

* corresponds to NTU-60 class ordering
* has shape `[60, 1024]`
* is L2-normalized


