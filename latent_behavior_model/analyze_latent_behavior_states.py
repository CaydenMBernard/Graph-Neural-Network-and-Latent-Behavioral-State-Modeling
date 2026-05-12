import os
import json
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar

# Configuration
INPUT_CSV = "winery_visits.csv"
MODEL_OUTPUT_DIR = "latent_behavior_outputs"
ANALYSIS_OUTPUT_DIR = "latent_behavior_analysis_clean"

EVENT_STATES_PATH = os.path.join(
    MODEL_OUTPUT_DIR,
    "events_with_latent_behavior_states.csv"
)

TRANSITION_PROBS_PATH = os.path.join(
    MODEL_OUTPUT_DIR,
    "latent_transition_matrix_probs.csv"
)

TRANSITION_COUNTS_PATH = os.path.join(
    MODEL_OUTPUT_DIR,
    "latent_transition_matrix_counts.csv"
)

os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# Human-readable behavior labels
STATE_NAME_MAP = {
    0: "Core Explorers",
    1: "Routine Visitors",
    2: "Local Dense Activity",
    3: "Weekend Light Visitors",
    4: "Transitional Users",
}

STATE_SHORT_NAME_MAP = {
    0: "Core\nExplorers",
    1: "Routine\nVisitors",
    2: "Local Dense\nActivity",
    3: "Weekend Light\nVisitors",
    4: "Transitional\nUsers",
}

STATE_DESCRIPTION_MAP = {
    0: (
        "High-connectivity, high-confidence behavior associated with broad activity "
        "through central wineries and many connected POIs."
    ),
    1: (
        "Structured repeat behavior involving high-connectivity wineries but a smaller "
        "set of surrounding POIs."
    ),
    2: (
        "Dense localized activity with high POI connectivity and frequent winery-to-POI "
        "interactions."
    ),
    3: (
        "Lower-intensity activity with a stronger weekend pattern, suggesting lighter "
        "tourism-oriented visits."
    ),
    4: (
        "Mixed behavior that bridges other patterns and appears across a range of "
        "structural contexts."
    ),
}

# Saves the current matplotlib figure.
def savefig(name):
    path = os.path.join(ANALYSIS_OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)

# Wraps long labels for plot axes.
def wrap_label(label, width=16):
    return "\n".join(textwrap.wrap(str(label), width=width))

# Normalizes each numeric column to the range 0-1.
def normalize_columns(df):
    out = df.copy()

    for col in out.columns:
        min_val = out[col].min()
        max_val = out[col].max()

        if max_val > min_val:
            out[col] = (out[col] - min_val) / (max_val - min_val)
        else:
            out[col] = 0.0

    return out

# Load latent states and raw visit data
events = pd.read_csv(EVENT_STATES_PATH)
raw = pd.read_csv(INPUT_CSV)

raw = raw.dropna(subset=[
    "winery_safegraph_place_id",
    "other_safegraph_place_id"
]).copy()

raw["winery_safegraph_place_id"] = raw["winery_safegraph_place_id"].astype(str)
raw["other_safegraph_place_id"] = raw["other_safegraph_place_id"].astype(str)

events["winery_safegraph_place_id"] = events["winery_safegraph_place_id"].astype(str)
events["other_safegraph_place_id"] = events["other_safegraph_place_id"].astype(str)

events["date"] = pd.to_datetime(events["date"], errors="coerce")
events["month"] = events["date"].dt.month
events["weekday"] = events["date"].dt.weekday
events["is_weekend"] = events["weekday"].isin([5, 6]).astype(int)

events["behavior_type"] = events["latent_state"].map(STATE_NAME_MAP)
events["behavior_type_short"] = events["latent_state"].map(STATE_SHORT_NAME_MAP)

print("Loaded latent behavior events:", len(events))

# Add structural graph features
edges = raw[[
    "winery_safegraph_place_id",
    "other_safegraph_place_id"
]].drop_duplicates()

winery_degree = edges["winery_safegraph_place_id"].value_counts()
poi_degree = edges["other_safegraph_place_id"].value_counts()

winery_poi_freq = (
    raw.groupby(["winery_safegraph_place_id", "other_safegraph_place_id"])
    .size()
    .reset_index(name="visit_pair_frequency")
)

events["winery_connectivity"] = (
    events["winery_safegraph_place_id"].map(winery_degree).fillna(0)
)

events["poi_connectivity"] = (
    events["other_safegraph_place_id"].map(poi_degree).fillna(0)
)

events = events.merge(
    winery_poi_freq,
    on=["winery_safegraph_place_id", "other_safegraph_place_id"],
    how="left"
)

events["visit_pair_frequency"] = events["visit_pair_frequency"].fillna(0)

# Build behavior type summary table
summary_rows = []

for state in sorted(events["latent_state"].unique()):
    sdf = events[events["latent_state"] == state]

    top_categories = (
        sdf["short_category"]
        .value_counts(normalize=True)
        .head(6)
        .round(4)
        .to_dict()
    )

    top_wineries = (
        sdf["winery_safegraph_place_id"]
        .value_counts()
        .head(8)
        .to_dict()
    )

    top_pois = (
        sdf["other_safegraph_place_id"]
        .value_counts()
        .head(8)
        .to_dict()
    )

    summary_rows.append({
        "latent_state": state,
        "behavior_type": STATE_NAME_MAP[state],
        "description": STATE_DESCRIPTION_MAP[state],
        "num_events": len(sdf),
        "share_of_events": len(sdf) / len(events),
        "avg_max_state_prob": sdf["max_state_prob"].mean(),
        "avg_winery_connectivity": sdf["winery_connectivity"].mean(),
        "avg_poi_connectivity": sdf["poi_connectivity"].mean(),
        "avg_visit_pair_frequency": sdf["visit_pair_frequency"].mean(),
        "weekend_share": sdf["is_weekend"].mean(),
        "top_categories": json.dumps(top_categories),
        "top_wineries": json.dumps(top_wineries),
        "top_pois": json.dumps(top_pois),
    })

state_summary = pd.DataFrame(summary_rows)

state_summary.to_csv(
    os.path.join(ANALYSIS_OUTPUT_DIR, "interpreted_behavior_type_summary.csv"),
    index=False
)

print("\nBehavior type summary:")
print(state_summary[[
    "latent_state",
    "behavior_type",
    "num_events",
    "share_of_events",
    "avg_winery_connectivity",
    "avg_poi_connectivity",
    "avg_visit_pair_frequency",
    "weekend_share",
    "avg_max_state_prob"
]])

# Load and reshape transition matrices
transition_probs = pd.read_csv(TRANSITION_PROBS_PATH, index_col=0)
transition_counts = pd.read_csv(TRANSITION_COUNTS_PATH, index_col=0)

# Converts labels like "State 0" to integer state IDs.
def state_label_to_int(label):
    return int(str(label).replace("State", "").strip())

transition_long = []

for from_state_label in transition_probs.index:
    for to_state_label in transition_probs.columns:
        from_state = state_label_to_int(from_state_label)
        to_state = state_label_to_int(to_state_label)

        prob = float(transition_probs.loc[from_state_label, to_state_label])
        count = int(transition_counts.loc[from_state_label, to_state_label])

        transition_long.append({
            "from_state": from_state,
            "to_state": to_state,
            "from_behavior": STATE_NAME_MAP[from_state],
            "to_behavior": STATE_NAME_MAP[to_state],
            "transition_label": f"{STATE_NAME_MAP[from_state]} → {STATE_NAME_MAP[to_state]}",
            "count": count,
            "probability": prob,
            "is_self_transition": from_state == to_state,
        })

transition_long = pd.DataFrame(transition_long)

transition_long.to_csv(
    os.path.join(ANALYSIS_OUTPUT_DIR, "all_behavior_type_transitions.csv"),
    index=False
)

top_nonself = (
    transition_long[transition_long["is_self_transition"] == False]
    .sort_values(["probability", "count"], ascending=False)
    .head(15)
)

top_nonself.to_csv(
    os.path.join(ANALYSIS_OUTPUT_DIR, "top_behavior_shifts.csv"),
    index=False
)

print("\nTop behavior shifts:")
print(top_nonself[[
    "transition_label",
    "count",
    "probability"
]])

# Save behavior interpretation table
label_rows = []

for _, row in state_summary.iterrows():
    state = int(row["latent_state"])
    label_rows.append({
        "latent_state": state,
        "behavior_type": STATE_NAME_MAP[state],
        "short_interpretation": STATE_DESCRIPTION_MAP[state],
        "event_share": row["share_of_events"],
        "weekend_share": row["weekend_share"],
        "avg_winery_connectivity": row["avg_winery_connectivity"],
        "avg_poi_connectivity": row["avg_poi_connectivity"],
        "avg_visit_pair_frequency": row["avg_visit_pair_frequency"],
    })

labels_df = pd.DataFrame(label_rows)

labels_df.to_csv(
    os.path.join(ANALYSIS_OUTPUT_DIR, "behavior_type_interpretations.csv"),
    index=False
)

# Plot behavior type distribution
ordered_states = sorted(state_summary["latent_state"].tolist())
ordered_summary = state_summary.set_index("latent_state").loc[ordered_states].reset_index()

plt.figure(figsize=(9, 5))
bars = plt.bar(
    ordered_summary["behavior_type"],
    ordered_summary["share_of_events"]
)

plt.xlabel("Behavior Type")
plt.ylabel("Share of Visit Events")
plt.title("Distribution of Visitor Behavior Types")
plt.xticks(rotation=20, ha="right")

for bar, value in zip(bars, ordered_summary["share_of_events"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1%}",
        ha="center",
        va="bottom",
        fontsize=9
    )

savefig("figure_behavior_type_distribution.png")

# Plot structural and temporal behavior profiles
profile_cols = [
    "avg_winery_connectivity",
    "avg_poi_connectivity",
    "avg_visit_pair_frequency",
    "weekend_share",
    "avg_max_state_prob"
]

friendly_feature_names = [
    "Winery\nConnectivity",
    "POI\nConnectivity",
    "Visit\nIntensity",
    "Weekend\nBias",
    "State\nConfidence"
]

profile = ordered_summary[profile_cols].copy()
profile_norm = normalize_columns(profile)

plt.figure(figsize=(9, 5))
plt.imshow(profile_norm.values, aspect="auto")

plt.xticks(
    range(len(friendly_feature_names)),
    friendly_feature_names
)

plt.yticks(
    range(len(ordered_summary)),
    [STATE_SHORT_NAME_MAP[s] for s in ordered_summary["latent_state"]]
)

plt.xlabel("Behavior Characteristic")
plt.ylabel("Behavior Type")
plt.title("Behavioral Characteristics of Each Visitor Type")

cbar = plt.colorbar()
cbar.set_label("Relative Strength")

for i in range(profile_norm.shape[0]):
    for j in range(profile_norm.shape[1]):
        value = profile_norm.iloc[i, j]

        if value >= 0.67:
            label = "High"
        elif value >= 0.34:
            label = "Med"
        else:
            label = "Low"

        plt.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="black"
        )

savefig("figure_behavior_characteristics_heatmap.png")

# Plot top cross-state behavior shifts
if len(top_nonself) > 0:
    plot_transitions = top_nonself.head(10).copy()
    transition_labels = [
        wrap_label(label, width=34)
        for label in plot_transitions["transition_label"]
    ]

    plt.figure(figsize=(10, 6))
    plt.barh(transition_labels[::-1], plot_transitions["probability"][::-1])

    plt.xlabel("Transition Probability")
    plt.ylabel("Behavior Shift")
    plt.title("Most Common Shifts Between Visitor Behavior Types")

    for i, value in enumerate(plot_transitions["probability"][::-1]):
        plt.text(
            value,
            i,
            f" {value:.1%}",
            va="center",
            fontsize=8
        )

    savefig("figure_behavior_shift_transitions.png")

# Plot seasonal behavior patterns
month_state = (
    events.groupby(["month", "latent_state"])
    .size()
    .reset_index(name="count")
)

month_pivot = month_state.pivot_table(
    index="month",
    columns="latent_state",
    values="count",
    fill_value=0
)

for state in ordered_states:
    if state not in month_pivot.columns:
        month_pivot[state] = 0

month_pivot = month_pivot[ordered_states]
month_norm = month_pivot.div(month_pivot.sum(axis=1), axis=0).fillna(0)

plt.figure(figsize=(10, 6))
plt.imshow(month_norm.values, aspect="auto")

plt.xticks(
    range(len(ordered_states)),
    [STATE_SHORT_NAME_MAP[s] for s in ordered_states],
    rotation=20,
    ha="right"
)

plt.yticks(
    range(len(month_norm.index)),
    [calendar.month_name[m] for m in month_norm.index]
)

plt.xlabel("Behavior Type")
plt.ylabel("Month")
plt.title("Seasonal Patterns of Visitor Behavior")

cbar = plt.colorbar()
cbar.set_label("Share of Monthly Activity")

savefig("figure_seasonal_behavior_patterns.png")

# Plot self-transition stability
self_transition_rows = transition_long[transition_long["is_self_transition"]].copy()
self_transition_rows = self_transition_rows.sort_values("from_state")

plt.figure(figsize=(9, 5))
bars = plt.bar(
    self_transition_rows["from_behavior"],
    self_transition_rows["probability"]
)

plt.xlabel("Behavior Type")
plt.ylabel("Self-Transition Probability")
plt.title("Behavioral Stability Across Trajectory Steps")
plt.xticks(rotation=20, ha="right")

for bar, value in zip(bars, self_transition_rows["probability"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1%}",
        ha="center",
        va="bottom",
        fontsize=9
    )

savefig("figure_behavioral_stability.png")

# Plot named transition matrix
transition_matrix_named = np.zeros((len(ordered_states), len(ordered_states)))

for i, from_state in enumerate(ordered_states):
    for j, to_state in enumerate(ordered_states):
        row = transition_long[
            (transition_long["from_state"] == from_state)
            & (transition_long["to_state"] == to_state)
        ]

        if len(row) > 0:
            transition_matrix_named[i, j] = row.iloc[0]["probability"]

plt.figure(figsize=(8, 7))
plt.imshow(transition_matrix_named, aspect="auto")

plt.xticks(
    range(len(ordered_states)),
    [STATE_SHORT_NAME_MAP[s] for s in ordered_states],
    rotation=25,
    ha="right"
)

plt.yticks(
    range(len(ordered_states)),
    [STATE_SHORT_NAME_MAP[s] for s in ordered_states]
)

plt.xlabel("Next Behavior Type")
plt.ylabel("Current Behavior Type")
plt.title("Behavior Transition Matrix")

cbar = plt.colorbar()
cbar.set_label("Transition Probability")

for i in range(transition_matrix_named.shape[0]):
    for j in range(transition_matrix_named.shape[1]):
        value = transition_matrix_named[i, j]

        if value > 0:
            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

savefig("figure_named_transition_matrix.png")

print("\nSaved clean analysis outputs to:", ANALYSIS_OUTPUT_DIR)

plt.show()