import os
import math
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="prepared_winery_sma")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--emb_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.002)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--neighbors", type=int, default=100)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_path", type=str, default="best_winery_smagnn.pt")
args = parser.parse_args()

# Reproducibility
# Sets random seeds for reproducible training.
def setup_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(args.seed)

# Helper functions and graph data structures
# Custom GELU activation used by the SMA-GNN layers.
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# Stores signed bipartite edges and neighbor lookup tables.
class Edge:
    def __init__(self, edge_list, n=None, m=None):
        edge_list = np.asarray(edge_list, dtype=np.int64)

        if n is None:
            n = int(edge_list[:, 0].max()) + 1
        if m is None:
            m = int(edge_list[:, 1].max()) + 1

        self.n = n
        self.m = m
        self.edge_list = edge_list
        self.edge_set = set()

        self.pos = []
        self.neg = []

        self.edge_abp = [[] for _ in range(self.n)]
        self.edge_bap = [[] for _ in range(self.m)]
        self.edge_abn = [[] for _ in range(self.n)]
        self.edge_ban = [[] for _ in range(self.m)]

        self.edge_abps = [set() for _ in range(self.n)]
        self.edge_abns = [set() for _ in range(self.n)]

        self.dega = np.zeros(self.n, dtype=np.int64)
        self.degb = np.zeros(self.m, dtype=np.int64)

        for a, b, s in edge_list:
            a, b, s = int(a), int(b), int(s)
            self.edge_set.add((a, b))
            self.dega[a] += 1
            self.degb[b] += 1

            if s == 1:
                self.pos.append((a, b))
                self.edge_abp[a].append(b)
                self.edge_abps[a].add(b)
                self.edge_bap[b].append(a)
            elif s == -1:
                self.neg.append((a, b))
                self.edge_abn[a].append(b)
                self.edge_abns[a].add(b)
                self.edge_ban[b].append(a)
            else:
                raise ValueError("Edge labels must be +1 or -1")

# Builds a signed adjacency matrix for selected a-side and b-side nodes.
def sub_edge(a_node, b_node, edge_obj):
    sub_edge_list = [[], []]

    map_a = {node_id: i for i, node_id in enumerate(a_node)}
    map_b = {node_id: i for i, node_id in enumerate(b_node)}

    n = len(a_node)
    m = len(b_node)
    sb = set(b_node)

    for an in a_node:
        for p in edge_obj.edge_abps[an] & sb:
            sub_edge_list[0].append((map_a[an], map_b[p]))
        for p in edge_obj.edge_abns[an] & sb:
            sub_edge_list[1].append((map_a[an], map_b[p]))

    adj = torch.zeros((n, m), dtype=torch.long)

    for i in range(2):
        if len(sub_edge_list[i]) > 0:
            pairs = torch.tensor(sub_edge_list[i], dtype=torch.long)
            adj[pairs[:, 0], pairs[:, 1]] = -1 if i == 1 else 1

    return adj

# Pads variable-sized tensors so they can be stacked into a batch.
def pad_tensor(batch):
    max_size = [max(tensor.size(dim) for tensor in batch) for dim in range(batch[0].dim())]
    batch_size = len(batch)
    out = torch.zeros([batch_size] + max_size, dtype=batch[0].dtype)

    for i, tensor in enumerate(batch):
        slices = tuple(slice(0, s) for s in tensor.size())
        out[i][slices] = tensor

    return out

# Collates variable-sized graph samples into padded batch tensors.
def collate_fn(batch):
    batched = {key: [] for key in batch[0]}

    for item in batch:
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                batched[key].append(value)
            elif isinstance(value, list):
                batched[key].append(torch.tensor(value, dtype=torch.long))
            else:
                batched[key].append(value)

    for key in batched:
        if isinstance(batched[key][0], torch.Tensor):
            if all(t.shape == batched[key][0].shape for t in batched[key]):
                batched[key] = torch.stack(batched[key])
            else:
                batched[key] = pad_tensor(batched[key])
        else:
            batched[key] = torch.tensor(batched[key])

    return batched

# Data loading
# Reads a tab-separated edge file into a NumPy array.
def read_edge_file(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            a, b, s = map(int, line.strip().split("\t"))
            rows.append((a, b, s))

    return np.asarray(rows, dtype=np.int64)

# Loads train, validation, and test edge files.
def load_data(data_dir):
    train_path = os.path.join(data_dir, "winery_training.txt")
    val_path = os.path.join(data_dir, "winery_validation.txt")
    test_path = os.path.join(data_dir, "winery_testing.txt")

    train_edges = read_edge_file(train_path)
    val_edges = read_edge_file(val_path)
    test_edges = read_edge_file(test_path)

    n = int(max(train_edges[:, 0].max(), val_edges[:, 0].max(), test_edges[:, 0].max())) + 1
    m = int(max(train_edges[:, 1].max(), val_edges[:, 1].max(), test_edges[:, 1].max())) + 1

    return train_edges, val_edges, test_edges, n, m

# Creates graph samples for SMA-GNN training and evaluation.
class GraphDataset(Dataset):
    def __init__(self, edge_list, full_train_graph, max_neighbors=100):
        self.val = Edge(edge_list, n=full_train_graph.n, m=full_train_graph.m)
        self.edges = full_train_graph
        self.max_neighbors = max_neighbors

    def __len__(self):
        return len(self.val.edge_list)

    def __getitem__(self, index):
        edge = self.val.edge_list[index]
        left = [int(edge[0])]
        right = [int(edge[1])]
        label = 1 if int(edge[2]) == 1 else 0

        neighbor = np.array(self.edges.edge_abn[left[0]] + self.edges.edge_abp[left[0]], dtype=np.int64)
        if len(neighbor) == 0:
            left_n = []
        else:
            deg = self.edges.degb[neighbor]
            deg_idx = np.argsort(deg)[::-1]
            neighbor = neighbor[deg_idx].tolist()
            samples = neighbor[:self.max_neighbors]
            left_n = list(set(samples) - set(right))

        neighbor = np.array(self.edges.edge_ban[right[0]] + self.edges.edge_bap[right[0]], dtype=np.int64)
        if len(neighbor) == 0:
            right_n = []
        else:
            deg = self.edges.dega[neighbor]
            deg_idx = np.argsort(deg)[::-1]
            neighbor = neighbor[deg_idx].tolist()
            samples = neighbor[:self.max_neighbors]
            right_n = list(set(samples) - set(left))

        sub_0 = sub_edge(left, left_n, self.edges)
        sub_1 = sub_edge(right_n, left_n, self.edges)
        sub_2 = sub_edge(right_n, right, self.edges)

        return {
            "left": left,
            "right": right,
            "left_n": left_n,
            "right_n": right_n,
            "sub_0": sub_0,
            "sub_1": sub_1,
            "sub_2": sub_2,
            "edge_s": label
        }

# Model
# Computes signed attention over neighboring graph features.
class Attention(nn.Module):
    def __init__(self, input_dim, head=4):
        super().__init__()

        self.bt_pre = nn.Linear(input_dim, 6)
        self.bt_cur = nn.Linear(input_dim, 6)

        self.fcc = nn.Sequential(
            GELU(),
            nn.Linear(8, head),
        )

        self.fcg = nn.Sequential(
            GELU(),
            nn.Linear(8, input_dim),
        )

        self.head = head
        self.dim = input_dim

        self.fuse = nn.Sequential(
            GELU(),
            nn.Linear(12, 4),
        )

        self.ffn = nn.Sequential(
            GELU(),
            nn.Linear(12, 8),
        )

    def forward(self, prev, curr, edges):
        bt_pre = self.bt_pre(prev)
        bt_cur = self.bt_cur(curr)

        shape = (bt_pre.shape[0], bt_pre.shape[1], bt_cur.shape[1], bt_pre.shape[2])
        bt_pre = bt_pre.unsqueeze(2).expand(shape)
        bt_cur = bt_cur.unsqueeze(1).expand(shape)

        c = torch.cat((bt_pre, bt_cur), dim=-1)
        c = self.fuse(c)
        c = c.clone()
        c[edges == 0] = 0

        edges1 = torch.sum(edges, dim=1) + 1
        edges1[edges1 == 0] = 1

        d = torch.sum(c, dim=1).unsqueeze(1).expand(c.shape) / edges1.unsqueeze(1).unsqueeze(-1)
        e = torch.max(c, dim=1)[0].unsqueeze(1).expand(c.shape)

        fused = self.ffn(torch.cat((c, d, e), dim=-1))
        c = self.fcc(fused).transpose(-2, -3)
        mask = edges.transpose(-1, -2)

        res_list = []

        for i in range(self.head):
            c_true = c[:, :, :, i]

            if c_true.shape[2] != 1:
                c_true = c_true.masked_fill(mask == 0, -9e15)
                c_true = F.softmax(c_true, dim=2)
                c_true = c_true.masked_fill(mask == 0, 0.0)

            start = self.dim // self.head * i
            end = self.dim // self.head * (i + 1)
            res = torch.bmm(c_true, prev[:, :, start:end])
            res_list.append(res)

        res = torch.cat(res_list, dim=-1)

        fused = fused.clone()
        fused[edges == 0] = 0
        res = res + self.fcg(torch.sum(fused, dim=1) / edges1.unsqueeze(-1))

        return res

# Applies multi-head signed attention and residual feature updates.
class MultiHeadAttLayer(nn.Module):
    def __init__(self, input_dim, output_dim=None):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.weight_curr = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.normal_(self.weight_curr, mean=0, std=0.1)

        self.fcp = nn.Linear(input_dim, output_dim)
        self.fcn = nn.Linear(input_dim, output_dim)

        self.ffn = nn.Sequential(
            GELU(),
            nn.Linear(input_dim, input_dim // 2),
            GELU(),
            nn.Linear(input_dim // 2, output_dim),
        )

        self.attn = Attention(input_dim)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, prev_layer_features, current_layer_features, edges):
        positive_features = self.attn(prev_layer_features, current_layer_features, edges == 1)
        negative_features = self.attn(prev_layer_features, current_layer_features, edges == -1)

        transformed_agg_features = self.fcp(positive_features) - self.fcn(negative_features)
        current_layer_features = torch.matmul(current_layer_features, self.weight_curr) + transformed_agg_features
        current_layer_features = self.ln(current_layer_features)

        return self.ffn(current_layer_features) + current_layer_features

# Applies the stacked subgraph attention layers used by SMA-GNN.
class SubGraphLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.attn1 = MultiHeadAttLayer(input_dim)
        self.attn2 = MultiHeadAttLayer(input_dim)
        self.attn3 = MultiHeadAttLayer(input_dim)
        self.attnp = MultiHeadAttLayer(input_dim)
        self.attnq = MultiHeadAttLayer(input_dim)

    def forward(self, emb_a, emb_an, emb_bn, emb_b, sub_0, sub_1, sub_2):
        sub_1_t = sub_1.transpose(1, 2)

        sub_11 = sub_2.transpose(1, 2).clone()
        sub_11[sub_11 != 0] = 1

        sub_22 = sub_0.transpose(1, 2).clone()
        sub_22[sub_22 != 0] = 1

        x0 = emb_a
        x1 = self.attn1(x0, emb_bn, sub_0)
        x2 = self.attn2(x1, emb_an, sub_1_t) + self.attnp(emb_a, emb_an, sub_11)
        x3 = self.attn3(x2, emb_b, sub_2) + self.attnq(emb_bn, emb_b, sub_22)

        return x3

# SMA-GNN model for signed winery-to-POI link prediction.
class SMAGNN(nn.Module):
    def __init__(self, n, m, emb_size=32):
        super().__init__()

        self.features_a = nn.Parameter(torch.randn((n, emb_size)), requires_grad=True)
        self.features_b = nn.Parameter(torch.randn((m, emb_size)), requires_grad=True)

        self.suba = SubGraphLayer(emb_size)
        self.subb = SubGraphLayer(emb_size)

        self.fcx = nn.Linear(emb_size, emb_size)
        self.fcy = nn.Linear(emb_size, emb_size)
        self.B = nn.Linear(emb_size, emb_size)

        self.C = nn.Sequential(
            GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_size * 5, emb_size),
            GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_size, emb_size // 4),
            GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_size // 4, 1),
        )

        self.emb_size = emb_size

    # Returns projected winery and POI embeddings.
    def get_embeddings(self, a, b, detach_main=False):
        emb_a = self.features_a[a.long()]
        emb_b = self.features_b[b.long()]

        if detach_main:
            emb_a = emb_a.detach()
            emb_b = emb_b.detach()

        emb_a = self.fcx(emb_a)
        emb_b = self.fcy(emb_b)

        return emb_a, emb_b

    # Runs the SMA-GNN forward pass for one batch.
    def forward(self, left, left_n, right_n, right, sub_0, sub_1, sub_2, **kwargs):
        embed_a, embed_b = self.get_embeddings(left, right, detach_main=True)
        embed_an, embed_bn = self.get_embeddings(right_n, left_n, detach_main=False)

        x = self.suba(embed_a, embed_an, embed_bn, embed_b, sub_0, sub_1, sub_2)

        y = self.subb(
            embed_b,
            embed_bn,
            embed_an,
            embed_a,
            sub_2.transpose(1, 2),
            sub_1.transpose(1, 2),
            sub_0.transpose(1, 2),
        )

        fuse = torch.cat((self.B(x) * y, x, y, embed_a, embed_b), dim=-1)
        fuse = self.C(fuse).squeeze(-1).squeeze(-1) * 4

        return torch.sigmoid(fuse)

    # Computes weighted binary cross-entropy loss and accuracy.
    def loss(self, pred_y, y):
        y = y.float()
        assert pred_y.size() == y.size(), "pred and y must have same shape"

        pos_ratio = y.mean().clamp(min=1e-6, max=1 - 1e-6)
        weight = torch.where(y > 0.5, 1.0 / pos_ratio, 1.0 / (1.0 - pos_ratio))
        loss = F.binary_cross_entropy(pred_y, y, weight=weight)

        acc = ((pred_y > 0.5) == (y > 0.5)).float().mean()

        return loss, acc

# Training and evaluation
# Evaluates the model using AUC, F1, and accuracy.
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    preds_all = []
    labels_all = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(**batch)

        preds_all.append(pred.detach().cpu())
        labels_all.append(batch["edge_s"].float().detach().cpu())

    preds = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()

    hard_preds = (preds >= 0.5).astype(int)

    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, hard_preds)
    acc = (hard_preds == labels).mean()

    return {
        "auc": float(auc),
        "f1": float(f1),
        "acc": float(acc),
    }

# Runs model training, validation, checkpointing, and final testing.
def train():
    device = torch.device(args.device)

    train_edges, val_edges, test_edges, n, m = load_data(args.data_dir)

    print(f"Train edges: {len(train_edges)}")
    print(f"Val edges:   {len(val_edges)}")
    print(f"Test edges:  {len(test_edges)}")
    print(f"Left nodes (wineries): {n}")
    print(f"Right nodes (other POIs): {m}")
    print(f"Using device: {device}")

    train_graph = Edge(train_edges, n=n, m=m)

    train_dataset = GraphDataset(train_edges, train_graph, max_neighbors=args.neighbors)
    val_dataset = GraphDataset(val_edges, train_graph, max_neighbors=args.neighbors)
    test_dataset = GraphDataset(test_edges, train_graph, max_neighbors=args.neighbors)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model = SMAGNN(n=n, m=m, emb_size=args.emb_size).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_auc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        steps = 0
        start_time = time.time()

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            pred = model(**batch)
            y = batch["edge_s"].float()

            loss, acc = model.loss(pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            steps += 1

            print(f"\r{steps}/1243", end='', flush=True)

        train_loss = epoch_loss / max(steps, 1)
        train_acc = epoch_acc / max(steps, 1)

        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            torch.save(model.state_dict(), args.save_path)

    print(f"\nBest model saved to: {args.save_path}")
    print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    model.load_state_dict(torch.load(args.save_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    print("\nFinal Test Results")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test F1:  {test_metrics['f1']:.4f}")
    print(f"Test ACC: {test_metrics['acc']:.4f}")

if __name__ == "__main__":
    train()