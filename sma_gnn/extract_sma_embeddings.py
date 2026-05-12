import os
import torch
import numpy as np


# Configuration
CHECKPOINT_PATH = "best_winery_smagnn.pt"
OUTPUT_DIR = "sma_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load checkpoint
state = torch.load(CHECKPOINT_PATH, map_location="cpu")


# Extract raw SMA-GNN embeddings
raw_winery_emb = state["features_a"].detach().cpu()
raw_poi_emb = state["features_b"].detach().cpu()


# Apply the trained projection layers
fcx_w = state["fcx.weight"].detach().cpu()
fcx_b = state["fcx.bias"].detach().cpu()

fcy_w = state["fcy.weight"].detach().cpu()
fcy_b = state["fcy.bias"].detach().cpu()

winery_emb = raw_winery_emb @ fcx_w.T + fcx_b
poi_emb = raw_poi_emb @ fcy_w.T + fcy_b

winery_emb = winery_emb.numpy()
poi_emb = poi_emb.numpy()

print("Winery embeddings:", winery_emb.shape)
print("POI embeddings:", poi_emb.shape)


# Save transformed embeddings
np.save(os.path.join(OUTPUT_DIR, "winery_embeddings.npy"), winery_emb)
np.save(os.path.join(OUTPUT_DIR, "poi_embeddings.npy"), poi_emb)

print("\nDone. Saved SMA-GNN embeddings to:", OUTPUT_DIR)