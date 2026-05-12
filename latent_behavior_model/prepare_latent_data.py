import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Configuration
INPUT_CSV = "winery_visits.csv"
DATA_DIR = "prepared_winery_sma"
EMBEDDING_DIR = "sma_embeddings"
OUTPUT_DIR = "prepared_latent_behavior"

MAX_SEQ_LEN = 50
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Load data and embeddings
df = pd.read_csv(INPUT_CSV)

with open(os.path.join(DATA_DIR, "winery_to_idx.json"), "r", encoding="utf-8") as f:
    winery_to_idx = json.load(f)

with open(os.path.join(DATA_DIR, "other_to_idx.json"), "r", encoding="utf-8") as f:
    other_to_idx = json.load(f)

winery_embeddings = np.load(os.path.join(EMBEDDING_DIR, "winery_embeddings.npy"))
poi_embeddings = np.load(os.path.join(EMBEDDING_DIR, "poi_embeddings.npy"))

print("Loaded winery embeddings:", winery_embeddings.shape)
print("Loaded POI embeddings:", poi_embeddings.shape)


# Clean visit data
required_cols = [
    "caid",
    "utc_date",
    "winery_safegraph_place_id",
    "other_safegraph_place_id",
    "other_top_category"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required_cols).copy()
df["utc_date"] = pd.to_datetime(df["utc_date"], errors="coerce")
df = df.dropna(subset=["utc_date"])

df["caid"] = df["caid"].astype(str)
df["winery_safegraph_place_id"] = df["winery_safegraph_place_id"].astype(str)
df["other_safegraph_place_id"] = df["other_safegraph_place_id"].astype(str)

df = df[
    df["winery_safegraph_place_id"].isin(winery_to_idx)
    & df["other_safegraph_place_id"].isin(other_to_idx)
].copy()

print("Rows after map filtering:", len(df))


# Standardize POI category labels
category_map = {
    "Restaurants and Other Eating Places": "Restaurants",
    "Museums, Historical Sites, and Similar Institutions": "Museums",
    "Gasoline Stations": "Gas",
    "Other Amusement and Recreation Industries": "Recreation",
    "Grocery Stores": "Grocery",
    "Traveler Accommodation": "Hotels",
    "Other Miscellaneous Store Retailers": "Retail",
    "Religious Organizations": "Religious",
    "General Merchandise Stores, including Warehouse Clubs and Supercenters": "General Stores",
    "Other Miscellaneous Manufacturing": "Manufacturing",
}

df["short_category"] = df["other_top_category"].map(category_map).fillna("Other")

df = df.drop_duplicates(subset=[
    "caid",
    "utc_date",
    "winery_safegraph_place_id",
    "other_safegraph_place_id",
    "short_category"
]).copy()


# Build temporal context features
df["month"] = df["utc_date"].dt.month
df["weekday"] = df["utc_date"].dt.weekday

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(float)

TEMP_COLS = [
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend"
]


# Builds one structural event vector from winery and POI embeddings.
def build_event_vector(row):
    winery_idx = winery_to_idx[row["winery_safegraph_place_id"]]
    poi_idx = other_to_idx[row["other_safegraph_place_id"]]

    w_emb = winery_embeddings[winery_idx]
    p_emb = poi_embeddings[poi_idx]

    return np.concatenate([w_emb, p_emb]).astype(np.float32)


# Builds one temporal context vector.
def build_context_vector(row):
    return row[TEMP_COLS].values.astype(np.float32)


# Build event and context matrices
df = df.sort_values(["caid", "utc_date", "winery_safegraph_place_id", "other_safegraph_place_id"])
df["event_index"] = df.groupby("caid").cumcount()

event_vectors = []
context_vectors = []

for _, row in df.iterrows():
    event_vectors.append(build_event_vector(row))
    context_vectors.append(build_context_vector(row))

df["event_vector_index"] = np.arange(len(event_vectors))

event_vectors = np.stack(event_vectors)
context_vectors = np.stack(context_vectors)

event_mean = event_vectors.mean(axis=0, keepdims=True)
event_std = event_vectors.std(axis=0, keepdims=True) + 1e-6
event_vectors_norm = (event_vectors - event_mean) / event_std

np.save(os.path.join(OUTPUT_DIR, "event_vectors_norm.npy"), event_vectors_norm)
np.save(os.path.join(OUTPUT_DIR, "context_vectors.npy"), context_vectors)
np.save(os.path.join(OUTPUT_DIR, "event_embedding_mean.npy"), event_mean)
np.save(os.path.join(OUTPUT_DIR, "event_embedding_std.npy"), event_std)

event_dim = event_vectors_norm.shape[1]
context_dim = context_vectors.shape[1]
input_dim = event_dim + context_dim

print("Event embedding matrix:", event_vectors_norm.shape)
print("Context matrix:", context_vectors.shape)
print("Input dim:", input_dim)


# Build visitor sequences
sequence_data = []

for caid, group in df.groupby("caid"):
    group = group.sort_values(["utc_date", "event_index"])

    if len(group) < 2:
        continue

    if len(group) > MAX_SEQ_LEN:
        group = group.iloc[:MAX_SEQ_LEN]

    sequence_data.append({
        "caid": caid,
        "vec_indices": group["event_vector_index"].astype(int).tolist(),
        "dates": group["utc_date"].astype(str).tolist(),
        "wineries": group["winery_safegraph_place_id"].tolist(),
        "pois": group["other_safegraph_place_id"].tolist(),
        "short_categories": group["short_category"].tolist(),
    })

print("Sequences:", len(sequence_data))

train_seqs, temp_seqs = train_test_split(
    sequence_data,
    test_size=0.30,
    random_state=RANDOM_SEED
)

val_seqs, test_seqs = train_test_split(
    temp_seqs,
    test_size=0.50,
    random_state=RANDOM_SEED
)

print("Train sequences:", len(train_seqs))
print("Val sequences:", len(val_seqs))
print("Test sequences:", len(test_seqs))


# Save prepared latent data
with open(os.path.join(OUTPUT_DIR, "all_sequences.json"), "w", encoding="utf-8") as f:
    json.dump(sequence_data, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "train_sequences.json"), "w", encoding="utf-8") as f:
    json.dump(train_seqs, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "val_sequences.json"), "w", encoding="utf-8") as f:
    json.dump(val_seqs, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "test_sequences.json"), "w", encoding="utf-8") as f:
    json.dump(test_seqs, f, indent=2)

summary = {
    "input_csv": INPUT_CSV,
    "num_rows_after_filtering": int(len(df)),
    "num_sequences": int(len(sequence_data)),
    "num_train_sequences": int(len(train_seqs)),
    "num_val_sequences": int(len(val_seqs)),
    "num_test_sequences": int(len(test_seqs)),
    "event_dim": int(event_dim),
    "context_dim": int(context_dim),
    "input_dim": int(input_dim),
    "max_seq_len": int(MAX_SEQ_LEN),
}

with open(os.path.join(OUTPUT_DIR, "latent_data_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nDone. Prepared latent behavior data saved to:", OUTPUT_DIR)
print(json.dumps(summary, indent=2))