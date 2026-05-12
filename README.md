# Graph Neural Network and Latent Behavioral State Modeling

This repository contains code for modeling winery visitor mobility patterns using a two-stage machine learning pipeline. First, an SMA-GNN model learns structural embeddings from winery-to-POI visit relationships. Then, a GRU-based latent behavior model uses those embeddings with temporal features to identify interpretable visitor behavior states and transition patterns.

## Project Structure

```text
data/
    Raw input CSV files.

sma_gnn/
    Scripts for preparing winery-to-POI graph data, training the SMA-GNN model, and extracting embeddings.

prepared_winery_sma/
    Prepared SMA-GNN edge files and node mappings.

sma_embeddings/
    Extracted winery and POI embeddings used by the latent behavior model.

latent_behavior_model/
    Scripts for preparing latent sequence data, training the latent behavior model, extracting latent state outputs, and analyzing behavior states.
```

## Pipeline

Run the scripts in this order from the repository root:

python sma_gnn/data_prep.py
python sma_gnn/train_winery_smagnn.py
python sma_gnn/extract_sma_embeddings.py

python latent_behavior_model/prepare_latent_data.py
python latent_behavior_model/train_latent_behavior_model.py
python latent_behavior_model/extract_latent_outputs.py
python latent_behavior_model/analyze_latent_behavior_states.py