import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Configuration
INPUT_CSV = "winery_visits.csv"
OUTPUT_DIR = "prepared_winery_sma"
RANDOM_SEED = 42
NEGATIVE_RATIO = 1  # Number of negative edges sampled per positive edge


# Setup
np.random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load and validate input data
df = pd.read_csv(INPUT_CSV)

required_cols = [
    "utc_date",
    "caid",
    "winery_safegraph_place_id",
    "other_safegraph_place_id",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["utc_date"] = pd.to_datetime(df["utc_date"], errors="coerce")

df = df.dropna(subset=["winery_safegraph_place_id", "other_safegraph_place_id"])

df["winery_safegraph_place_id"] = df["winery_safegraph_place_id"].astype(str).str.strip()
df["other_safegraph_place_id"] = df["other_safegraph_place_id"].astype(str).str.strip()
df["caid"] = df["caid"].astype(str).str.strip()

df = df[
    (df["winery_safegraph_place_id"] != "") &
    (df["other_safegraph_place_id"] != "")
].copy()

df = df.drop_duplicates(subset=[
    "utc_date",
    "caid",
    "winery_safegraph_place_id",
    "other_safegraph_place_id"
])

print("Rows after cleaning:", len(df))


# Build bipartite node mappings
unique_wineries = sorted(df["winery_safegraph_place_id"].unique())
unique_others = sorted(df["other_safegraph_place_id"].unique())

winery_to_idx = {place_id: i for i, place_id in enumerate(unique_wineries)}
other_to_idx = {place_id: i for i, place_id in enumerate(unique_others)}

print("Number of winery nodes:", len(winery_to_idx))
print("Number of other nodes:", len(other_to_idx))

with open(os.path.join(OUTPUT_DIR, "winery_to_idx.json"), "w") as f:
    json.dump(winery_to_idx, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "other_to_idx.json"), "w") as f:
    json.dump(other_to_idx, f, indent=2)


# Save metadata for interpreting model outputs
winery_meta = (
    df[[
        "winery_safegraph_place_id",
        "winery_locations_name",
        "winery_city",
        "winery_state",
        "winery_top_category",
        "winery_sub_category"
    ]]
    .drop_duplicates("winery_safegraph_place_id")
    .fillna("")
)

other_meta = (
    df[[
        "other_safegraph_place_id",
        "other_location_name",
        "other_city",
        "other_state",
        "other_top_category",
        "other_sub_category"
    ]]
    .drop_duplicates("other_safegraph_place_id")
    .fillna("")
)

winery_meta.to_csv(os.path.join(OUTPUT_DIR, "winery_metadata.csv"), index=False)
other_meta.to_csv(os.path.join(OUTPUT_DIR, "other_metadata.csv"), index=False)


# Build positive edges from observed winery-to-POI visits
positive_edges_df = (
    df[["winery_safegraph_place_id", "other_safegraph_place_id"]]
    .drop_duplicates()
    .copy()
)

positive_edges_df["a"] = positive_edges_df["winery_safegraph_place_id"].map(winery_to_idx)
positive_edges_df["b"] = positive_edges_df["other_safegraph_place_id"].map(other_to_idx)
positive_edges_df["label"] = 1

positive_edges = list(
    positive_edges_df[["a", "b", "label"]].itertuples(index=False, name=None)
)

print("Unique positive edges:", len(positive_edges))


# Sample negative edges from unobserved winery-to-POI pairs
positive_set = set((a, b) for a, b, _ in positive_edges)

all_winery_indices = np.array(list(range(len(winery_to_idx))))
all_other_indices = np.array(list(range(len(other_to_idx))))

num_negative_needed = len(positive_edges) * NEGATIVE_RATIO
negative_set = set()

attempts = 0
max_attempts = num_negative_needed * 20

while len(negative_set) < num_negative_needed and attempts < max_attempts:
    a = np.random.choice(all_winery_indices)
    b = np.random.choice(all_other_indices)
    if (a, b) not in positive_set and (a, b) not in negative_set:
        negative_set.add((a, b))
    attempts += 1

negative_edges = [(a, b, -1) for a, b in negative_set]

print("Negative edges created:", len(negative_edges))

if len(negative_edges) < num_negative_needed:
    print("Warning: Could not generate as many negatives as requested.")


# Combine edges and create train/validation/test splits
all_edges = positive_edges + negative_edges
edges_df = pd.DataFrame(all_edges, columns=["a", "b", "label"])

train_df, temp_df = train_test_split(
    edges_df,
    test_size=0.30,
    stratify=edges_df["label"],
    random_state=RANDOM_SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=RANDOM_SEED
)

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))


# Save edge files as CSV
edges_df.to_csv(os.path.join(OUTPUT_DIR, "all_edges.csv"), index=False)
train_df.to_csv(os.path.join(OUTPUT_DIR, "train_edges.csv"), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, "val_edges.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_edges.csv"), index=False)


# Save edge files in the text format expected by the SMA-GNN training script
def save_txt_edge_file(df_part, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("a\tb\tlabel\n")
        for row in df_part.itertuples(index=False):
            f.write(f"{int(row.a)}\t{int(row.b)}\t{int(row.label)}\n")


save_txt_edge_file(train_df, os.path.join(OUTPUT_DIR, "winery_training.txt"))
save_txt_edge_file(val_df, os.path.join(OUTPUT_DIR, "winery_validation.txt"))
save_txt_edge_file(test_df, os.path.join(OUTPUT_DIR, "winery_testing.txt"))


# Save dataset summary statistics
stats = {
    "num_rows_after_cleaning": int(len(df)),
    "num_unique_visitors": int(df["caid"].nunique()),
    "num_winery_nodes": int(len(winery_to_idx)),
    "num_other_nodes": int(len(other_to_idx)),
    "num_positive_edges": int(len(positive_edges)),
    "num_negative_edges": int(len(negative_edges)),
    "num_train_edges": int(len(train_df)),
    "num_val_edges": int(len(val_df)),
    "num_test_edges": int(len(test_df)),
    "date_min": str(df["utc_date"].min()),
    "date_max": str(df["utc_date"].max()),
}

with open(os.path.join(OUTPUT_DIR, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print("\nDone. Files saved to:", OUTPUT_DIR)
print(json.dumps(stats, indent=2))