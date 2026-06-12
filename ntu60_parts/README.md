# NTU-60 Part Embeddings

Generated per-class, per-body-part text embeddings for NTU-60 using STAR anatomical part descriptions.

## Encoder

* Model: sentence-transformers/stsb-bert-large
* Library: sentence-transformers

## Embedding Details

* One embedding matrix per anatomical part:

  * head_embeddings.npy
  * hand_embeddings.npy
  * arm_embeddings.npy
  * hip_embeddings.npy
  * leg_embeddings.npy
  * foot_embeddings.npy

* Shape of each embedding matrix:

  * [60, 1024]

* All embeddings were L2-normalized.

## Indexing Verification

Verified that STAR class ordering matches NTU-60 ordering:

0 → drink water
1 → eat meal
2 → brush teeth
3 → brush hair
4 → drop

The first 60 STAR entries were used for NTU-60 alignment.
