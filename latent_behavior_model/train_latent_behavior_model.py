import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from latent_model import LatentBehaviorModel


# Configuration
DATA_DIR = "prepared_latent_behavior"
OUTPUT_DIR = "latent_behavior_outputs"

N_LATENT_STATES = 5
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.0005
RANDOM_SEED = 42

RECON_WEIGHT = 1.0
KL_WEIGHT_MAX = 0.25
STATE_BALANCE_WEIGHT = 0.10
ENTROPY_WEIGHT = 0.01

DEVICE = "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# Load prepared latent data
event_vectors_norm = np.load(os.path.join(DATA_DIR, "event_vectors_norm.npy"))
context_vectors = np.load(os.path.join(DATA_DIR, "context_vectors.npy"))

with open(os.path.join(DATA_DIR, "train_sequences.json"), "r", encoding="utf-8") as f:
    train_seqs = json.load(f)

with open(os.path.join(DATA_DIR, "val_sequences.json"), "r", encoding="utf-8") as f:
    val_seqs = json.load(f)

with open(os.path.join(DATA_DIR, "test_sequences.json"), "r", encoding="utf-8") as f:
    test_seqs = json.load(f)

EVENT_DIM = event_vectors_norm.shape[1]
CONTEXT_DIM = context_vectors.shape[1]
INPUT_DIM = EVENT_DIM + CONTEXT_DIM

print("Event embedding matrix:", event_vectors_norm.shape)
print("Context matrix:", context_vectors.shape)
print("Input dim:", INPUT_DIM)
print("Train sequences:", len(train_seqs))
print("Val sequences:", len(val_seqs))
print("Test sequences:", len(test_seqs))


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


train_loader = DataLoader(
    SequenceDataset(train_seqs, event_vectors_norm, context_vectors),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    SequenceDataset(val_seqs, event_vectors_norm, context_vectors),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    SequenceDataset(test_seqs, event_vectors_norm, context_vectors),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


model = LatentBehaviorModel(
    input_dim=INPUT_DIM,
    event_dim=EVENT_DIM,
    context_dim=CONTEXT_DIM,
    hidden_dim=HIDDEN_DIM,
    n_states=N_LATENT_STATES
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# Computes MSE only over valid sequence positions.
def masked_mse(pred, target, mask):
    valid_pred = pred[mask]
    valid_target = target[mask]

    if valid_pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    return F.mse_loss(valid_pred, valid_target)


# Computes KL consistency between inferred states and transition prior.
def transition_kl_loss(q_probs, context, mask):
    if q_probs.shape[1] < 2:
        return torch.tensor(0.0, device=q_probs.device)

    q_prev = q_probs[:, :-1, :]
    q_curr = q_probs[:, 1:, :]
    context_curr = context[:, 1:, :]
    mask_curr = mask[:, 1:]

    prior_logits, prior_probs = model.transition_prior(q_prev, context_curr)

    q_valid = q_curr[mask_curr]
    p_valid = prior_probs[mask_curr]

    if q_valid.numel() == 0:
        return torch.tensor(0.0, device=q_probs.device)

    kl = torch.sum(
        q_valid * (torch.log(q_valid + 1e-8) - torch.log(p_valid + 1e-8)),
        dim=-1
    )

    return kl.mean()


# Encourages all latent states to be used across the batch.
def state_balance_loss(q_probs, mask):
    valid_q = q_probs[mask]

    if valid_q.numel() == 0:
        return torch.tensor(0.0, device=q_probs.device)

    avg_usage = valid_q.mean(dim=0)
    target = torch.full_like(avg_usage, 1.0 / avg_usage.numel())

    return torch.mean((avg_usage - target) ** 2)


# Encourages sharper state assignments.
def entropy_loss(q_probs, mask):
    valid_q = q_probs[mask]

    if valid_q.numel() == 0:
        return torch.tensor(0.0, device=q_probs.device)

    entropy = -torch.sum(valid_q * torch.log(valid_q + 1e-8), dim=-1)

    return entropy.mean()


# Increases the KL loss weight gradually during training.
def get_kl_weight(epoch):
    warmup_epochs = max(1, EPOCHS // 3)
    return KL_WEIGHT_MAX * min(1.0, epoch / warmup_epochs)


# Computes the full training objective for one batch.
def compute_loss(batch, epoch):
    x = batch["x"].to(DEVICE)
    event = batch["event"].to(DEVICE)
    context = batch["context"].to(DEVICE)
    mask = batch["mask"].to(DEVICE)

    outputs = model(x, context)

    recon = masked_mse(outputs["recon_event"], event, mask)
    kl = transition_kl_loss(outputs["q_probs"], context, mask)
    balance = state_balance_loss(outputs["q_probs"], mask)
    entropy = entropy_loss(outputs["q_probs"], mask)

    kl_weight = get_kl_weight(epoch)

    loss = (
        RECON_WEIGHT * recon
        + kl_weight * kl
        + STATE_BALANCE_WEIGHT * balance
        + ENTROPY_WEIGHT * entropy
    )

    return loss, {
        "recon": recon.item(),
        "kl": kl.item(),
        "balance": balance.item(),
        "entropy": entropy.item(),
        "kl_weight": kl_weight
    }


# Evaluates average loss components for a data loader.
@torch.no_grad()
def evaluate(loader, epoch):
    model.eval()

    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "kl": 0.0,
        "balance": 0.0,
        "entropy": 0.0
    }

    steps = 0

    for batch in loader:
        loss, parts = compute_loss(batch, epoch)

        totals["loss"] += loss.item()
        for k in ["recon", "kl", "balance", "entropy"]:
            totals[k] += parts[k]

        steps += 1

    for k in totals:
        totals[k] /= max(steps, 1)

    return totals


# Train model
best_val_loss = float("inf")
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()

    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "kl": 0.0,
        "balance": 0.0,
        "entropy": 0.0
    }

    steps = 0

    for batch in train_loader:
        optimizer.zero_grad()

        loss, parts = compute_loss(batch, epoch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        totals["loss"] += loss.item()
        for k in ["recon", "kl", "balance", "entropy"]:
            totals[k] += parts[k]

        steps += 1

    train_metrics = {k: totals[k] / max(steps, 1) for k in totals}
    val_metrics = evaluate(val_loader, epoch)

    record = {
        "epoch": epoch,
        "kl_weight": get_kl_weight(epoch),
        "train_loss": train_metrics["loss"],
        "train_recon": train_metrics["recon"],
        "train_kl": train_metrics["kl"],
        "train_balance": train_metrics["balance"],
        "train_entropy": train_metrics["entropy"],
        "val_loss": val_metrics["loss"],
        "val_recon": val_metrics["recon"],
        "val_kl": val_metrics["kl"],
        "val_balance": val_metrics["balance"],
        "val_entropy": val_metrics["entropy"],
    }

    history.append(record)

    print(
        f"Epoch {epoch:03d} SUMMARY | "
        f"train_loss={train_metrics['loss']:.4f} | "
        f"train_recon={train_metrics['recon']:.4f} | "
        f"train_kl={train_metrics['kl']:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"val_recon={val_metrics['recon']:.4f} | "
        f"val_kl={val_metrics['kl']:.4f}"
    )

    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_latent_behavior_model.pt"))


# Evaluate best checkpoint
model.load_state_dict(torch.load(
    os.path.join(OUTPUT_DIR, "best_latent_behavior_model.pt"),
    map_location=DEVICE
))

test_metrics = evaluate(test_loader, EPOCHS)

print("\nFinal Test Results")
print(f"Test loss:   {test_metrics['loss']:.4f}")
print(f"Test recon:  {test_metrics['recon']:.4f}")
print(f"Test KL:     {test_metrics['kl']:.4f}")
print(f"Test entropy:{test_metrics['entropy']:.4f}")


# Save training history and run summary
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

summary = {
    "prepared_data_dir": DATA_DIR,
    "num_train_sequences": len(train_seqs),
    "num_val_sequences": len(val_seqs),
    "num_test_sequences": len(test_seqs),
    "event_dim": EVENT_DIM,
    "context_dim": CONTEXT_DIM,
    "input_dim": INPUT_DIM,
    "hidden_dim": HIDDEN_DIM,
    "num_latent_states": N_LATENT_STATES,
    "best_val_loss": float(best_val_loss),
    "test_loss": float(test_metrics["loss"]),
    "test_recon": float(test_metrics["recon"]),
    "test_kl": float(test_metrics["kl"]),
    "outputs_saved_to": OUTPUT_DIR
}

with open(os.path.join(OUTPUT_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved training outputs to:", OUTPUT_DIR)
print(json.dumps(summary, indent=2))


# Generate training figures
plt.figure(figsize=(7, 4))
plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Latent Behavioral Model Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_training_loss.png"), dpi=300)

plt.figure(figsize=(7, 4))
plt.plot(history_df["epoch"], history_df["train_recon"], label="Train Reconstruction")
plt.plot(history_df["epoch"], history_df["val_recon"], label="Validation Reconstruction")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Structural Embedding Reconstruction Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_reconstruction_loss.png"), dpi=300)