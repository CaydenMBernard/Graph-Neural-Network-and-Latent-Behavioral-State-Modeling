import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

from latent_model import LatentBehaviorModel


# Configuration
DATA_DIR = "prepared_latent_behavior"
MODEL_DIR = "latent_behavior_outputs"
OUTPUT_DIR = "latent_behavior_outputs"

BATCH_SIZE = 64
DEVICE = "cpu"

DEFAULT_HIDDEN_DIM = 128
DEFAULT_N_LATENT_STATES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load prepared latent data
event_vectors_norm = np.load(os.path.join(DATA_DIR, "event_vectors_norm.npy"))
context_vectors = np.load(os.path.join(DATA_DIR, "context_vectors.npy"))

with open(os.path.join(DATA_DIR, "all_sequences.json"), "r", encoding="utf-8") as f:
    sequence_data = json.load(f)

EVENT_DIM = event_vectors_norm.shape[1]
CONTEXT_DIM = context_vectors.shape[1]
INPUT_DIM = EVENT_DIM + CONTEXT_DIM


# Load model settings from training summary
summary_path = os.path.join(MODEL_DIR, "run_summary.json")

if os.path.exists(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        train_summary = json.load(f)

    HIDDEN_DIM = int(train_summary.get("hidden_dim", DEFAULT_HIDDEN_DIM))
    N_LATENT_STATES = int(train_summary.get("num_latent_states", DEFAULT_N_LATENT_STATES))
else:
    HIDDEN_DIM = DEFAULT_HIDDEN_DIM
    N_LATENT_STATES = DEFAULT_N_LATENT_STATES

print("Loaded sequences:", len(sequence_data))
print("Input dim:", INPUT_DIM)
print("Hidden dim:", HIDDEN_DIM)
print("Latent states:", N_LATENT_STATES)


# Dataset for variable-length visitor sequences
class SequenceDataset(Dataset):
    def __init__(self, seqs, event_vectors_norm, context_vectors):
        self.seqs = seqs
        self.event_vectors_norm = event_vectors_norm
        self.context_vectors = context_vectors

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        ids = seq["vec_indices"]

        event = self.event_vectors_norm[ids]
        context = self.context_vectors[ids]
        x = np.concatenate([event, context], axis=1)

        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
            "context": torch.tensor(context, dtype=torch.float32),
            "length": len(ids),
            "meta": seq
        }


# Pads variable-length sequences into batch tensors.
def collate_fn(batch):
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = int(lengths.max())

    x_pad = torch.zeros(len(batch), max_len, INPUT_DIM, dtype=torch.float32)
    event_pad = torch.zeros(len(batch), max_len, EVENT_DIM, dtype=torch.float32)
    context_pad = torch.zeros(len(batch), max_len, CONTEXT_DIM, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    metas = []

    for i, item in enumerate(batch):
        L = item["length"]
        x_pad[i, :L] = item["x"]
        event_pad[i, :L] = item["event"]
        context_pad[i, :L] = item["context"]
        mask[i, :L] = True
        metas.append(item["meta"])

    return {
        "x": x_pad,
        "event": event_pad,
        "context": context_pad,
        "lengths": lengths,
        "mask": mask,
        "metas": metas
    }


# Load trained model
model = LatentBehaviorModel(
    input_dim=INPUT_DIM,
    event_dim=EVENT_DIM,
    context_dim=CONTEXT_DIM,
    hidden_dim=HIDDEN_DIM,
    n_states=N_LATENT_STATES
).to(DEVICE)

model.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, "best_latent_behavior_model.pt"),
    map_location=DEVICE
))

model.eval()


# Extract latent state assignments
event_state_records = []
transition_counts = np.zeros((N_LATENT_STATES, N_LATENT_STATES), dtype=int)
all_state_probs = []

full_loader = DataLoader(
    SequenceDataset(sequence_data, event_vectors_norm, context_vectors),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

with torch.no_grad():
    for batch in full_loader:
        x = batch["x"].to(DEVICE)
        context = batch["context"].to(DEVICE)
        metas = batch["metas"]

        outputs = model(x, context)

        q_probs = outputs["q_probs"].cpu().numpy()
        states = q_probs.argmax(axis=-1)

        for i, meta in enumerate(metas):
            L = len(meta["vec_indices"])
            seq_states = states[i, :L].tolist()

            for t in range(L):
                probs_t = q_probs[i, t]
                all_state_probs.append(probs_t)

                row = {
                    "caid": meta["caid"],
                    "t": t,
                    "date": meta["dates"][t],
                    "winery_safegraph_place_id": meta["wineries"][t],
                    "other_safegraph_place_id": meta["pois"][t],
                    "short_category": meta["short_categories"][t],
                    "latent_state": int(seq_states[t]),
                    "max_state_prob": float(probs_t.max())
                }

                for s in range(N_LATENT_STATES):
                    row[f"state_prob_{s}"] = float(probs_t[s])

                event_state_records.append(row)

            for t in range(L - 1):
                transition_counts[seq_states[t], seq_states[t + 1]] += 1


event_states_df = pd.DataFrame(event_state_records)
event_states_df.to_csv(os.path.join(OUTPUT_DIR, "events_with_latent_behavior_states.csv"), index=False)

all_state_probs = np.asarray(all_state_probs)
np.save(os.path.join(OUTPUT_DIR, "event_state_probabilities.npy"), all_state_probs)


# Save transition matrices
transition_probs = transition_counts.astype(float)
row_sums = transition_probs.sum(axis=1, keepdims=True)

transition_probs = np.divide(
    transition_probs,
    row_sums,
    out=np.zeros_like(transition_probs),
    where=row_sums != 0
)

state_labels = [f"State {i}" for i in range(N_LATENT_STATES)]

pd.DataFrame(
    transition_counts,
    index=state_labels,
    columns=state_labels
).to_csv(os.path.join(OUTPUT_DIR, "latent_transition_matrix_counts.csv"))

pd.DataFrame(
    transition_probs,
    index=state_labels,
    columns=state_labels
).to_csv(os.path.join(OUTPUT_DIR, "latent_transition_matrix_probs.csv"))


# Save latent state summary
state_summary = []

for s in range(N_LATENT_STATES):
    sdf = event_states_df[event_states_df["latent_state"] == s]

    if len(sdf) == 0:
        continue

    top_categories = (
        sdf["short_category"]
        .value_counts(normalize=True)
        .head(6)
        .round(4)
        .to_dict()
    )

    state_summary.append({
        "latent_state": s,
        "num_events": int(len(sdf)),
        "share_of_events": float(len(sdf) / len(event_states_df)),
        "avg_max_state_prob": float(sdf["max_state_prob"].mean()),
        "top_categories": json.dumps(top_categories)
    })

state_summary_df = pd.DataFrame(state_summary)
state_summary_df.to_csv(os.path.join(OUTPUT_DIR, "latent_state_summary.csv"), index=False)


# Generate output figures
state_counts = event_states_df["latent_state"].value_counts().sort_index()

plt.figure(figsize=(7, 4))
plt.bar([f"State {i}" for i in state_counts.index], state_counts.values)
plt.xlabel("Latent State")
plt.ylabel("Number of Events")
plt.title("Learned Latent Behavioral State Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_latent_state_distribution.png"), dpi=300)

plt.figure(figsize=(7, 6))
plt.imshow(transition_probs, aspect="auto")
plt.xlabel("Next State")
plt.ylabel("Current State")
plt.title("Learned Behavioral State Transition Matrix")
plt.xticks(range(N_LATENT_STATES), state_labels, rotation=45, ha="right")
plt.yticks(range(N_LATENT_STATES), state_labels)

cbar = plt.colorbar()
cbar.set_label("Transition Probability")

for i in range(N_LATENT_STATES):
    for j in range(N_LATENT_STATES):
        value = transition_probs[i, j]
        if value > 0:
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_transition_matrix.png"), dpi=300)

state_category = (
    event_states_df.groupby(["latent_state", "short_category"])
    .size()
    .reset_index(name="count")
)

state_category_pivot = state_category.pivot_table(
    index="latent_state",
    columns="short_category",
    values="count",
    fill_value=0
)

state_category_norm = state_category_pivot.div(
    state_category_pivot.sum(axis=1),
    axis=0
).fillna(0)

plt.figure(figsize=(10, 5))
plt.imshow(state_category_norm.values, aspect="auto")
plt.xlabel("POI Category")
plt.ylabel("Latent State")
plt.title("Latent Behavioral State Category Profiles")

plt.xticks(
    range(len(state_category_norm.columns)),
    state_category_norm.columns,
    rotation=35,
    ha="right"
)

plt.yticks(
    range(len(state_category_norm.index)),
    [f"State {i}" for i in state_category_norm.index]
)

cbar = plt.colorbar()
cbar.set_label("Category Share within State")

for i in range(state_category_norm.shape[0]):
    for j in range(state_category_norm.shape[1]):
        value = state_category_norm.values[i, j]
        if value > 0.05:
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_state_category_profiles.png"), dpi=300)


# Generate PCA visualization of latent state probability profiles
if all_state_probs.shape[1] >= 2:
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(all_state_probs)

    pca_df = pd.DataFrame({
        "pc1": pca_coords[:, 0],
        "pc2": pca_coords[:, 1],
        "latent_state": event_states_df["latent_state"].values
    })

    pca_df.to_csv(os.path.join(OUTPUT_DIR, "latent_state_probability_pca.csv"), index=False)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        pca_df["pc1"],
        pca_df["pc2"],
        c=pca_df["latent_state"],
        s=8,
        alpha=0.6
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Latent State Probability Profiles")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Latent State")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_latent_state_probability_pca.png"), dpi=300)


summary = {
    "num_events": int(len(event_states_df)),
    "num_sequences": int(len(sequence_data)),
    "num_latent_states": int(N_LATENT_STATES),
    "outputs_saved_to": OUTPUT_DIR,
}

with open(os.path.join(OUTPUT_DIR, "latent_output_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nDone. Saved latent behavior outputs to:", OUTPUT_DIR)
print(json.dumps(summary, indent=2))