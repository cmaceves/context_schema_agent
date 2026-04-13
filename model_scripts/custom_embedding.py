import os
import numpy as np
import pandas as pd
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


# ---------------------------
# Configurable RGCN
# ---------------------------
class ConfigurableRGCN(nn.Module):
    def __init__(
        self,
        n_entities,
        n_relations,
        emb_dim_init,
        hidden_dim,
        out_dim,
        n_layers,
        device="cuda",
    ):
        super().__init__()
        self.num_bases = 34
        self.device = device
        self.num_ent = n_entities
        self.n_relations = n_relations
        self.n_layers = n_layers

        self.node_emb = nn.Parameter(torch.randn(n_entities, emb_dim_init))

        self.conv_layers = nn.ModuleList()

        if n_layers == 1:
            self.conv_layers.append(
                RGCNConv(
                    emb_dim_init,
                    out_dim,
                    n_relations,
                    num_bases=self.num_bases,
                )
            )
        else:
            self.conv_layers.append(
                RGCNConv(
                    emb_dim_init,
                    hidden_dim,
                    n_relations,
                    num_bases=self.num_bases,
                )
            )

            for _ in range(n_layers - 2):
                self.conv_layers.append(
                    RGCNConv(
                        hidden_dim,
                        hidden_dim,
                        n_relations,
                        num_bases=self.num_bases,
                    )
                )

            self.conv_layers.append(
                RGCNConv(
                    hidden_dim,
                    out_dim,
                    n_relations,
                    num_bases=self.num_bases,
                )
            )

        self.to(device)

    def forward(
        self,
        node_index,
        edge_index,
        edge_type,
        node_frequency,
        last_epoch=False,
        node_weights=None,
        epoch=None,
        max_epochs=None,
        use_node_weights_embedding=False,
    ):
        x = self.node_emb[node_index]

        if epoch is None:
            node_weights = None

        for i, conv in enumerate(self.conv_layers):
            src, dst = edge_index

            if not last_epoch:
                if use_node_weights_embedding and node_weights is not None:
                    w = node_weights[node_index].to(self.device).unsqueeze(1)
                    x = x * w

                x = conv(x, edge_index, edge_type)
            else:
                x = conv(x, edge_index, edge_type)

        return x


# ---------------------------
# Build vocab
# ---------------------------
def build_vocab(df):
    nodes = pd.Index(pd.concat([df["head"], df["tail"]]).unique())
    node2id = {n: i for i, n in enumerate(nodes)}
    relations = pd.Index(df["relation"].unique())
    rel2id = {r: i for i, r in enumerate(relations)}
    return node2id, rel2id


# ---------------------------
# Train/test split on target relation
# ---------------------------
def train_test_split(
    df,
    node2id,
    rel2id,
    rel_name="treats",
    test_frac=0.1,
    seed=42,
    forced_test_edges=None,
):
    random.seed(seed)

    target_df = df[df["relation"] == rel_name].copy()
    target_df = target_df.reset_index(drop=True)

    forced_test_idx = set()
    if forced_test_edges is not None:
        forced_set = set(forced_test_edges)
        for i, row in target_df.iterrows():
            if (row["head"], row["tail"]) in forced_set:
                forced_test_idx.add(i)

    all_indices = set(range(len(target_df)))
    remaining_indices = list(all_indices - forced_test_idx)
    random.shuffle(remaining_indices)

    test_size = int(len(target_df) * test_frac)
    remaining_test_needed = max(0, test_size - len(forced_test_idx))

    test_idx = list(forced_test_idx) + remaining_indices[:remaining_test_needed]
    train_idx = remaining_indices[remaining_test_needed:]

    train_df = target_df.iloc[train_idx]
    test_df = target_df.iloc[test_idx]

    def df_to_edge(df_subset):
        ei = torch.tensor(
            [
                [node2id[h] for h in df_subset["head"]],
                [node2id[t] for t in df_subset["tail"]],
            ],
            dtype=torch.long,
        )
        et = torch.tensor(
            [rel2id[r] for r in df_subset["relation"]],
            dtype=torch.long,
        )
        return ei, et

    train_edge_index, train_edge_type = df_to_edge(train_df)
    test_edge_index, test_edge_type = df_to_edge(test_df)

    return (train_edge_index, train_edge_type), (test_edge_index, test_edge_type)


# ---------------------------
# Remove actual held-out test edges from graph
# ---------------------------
def remove_actual_test_edges_from_graph(df, test_edge_index, node2id, rel_name="treats"):
    id2node = {v: k for k, v in node2id.items()}

    test_heads = [id2node[h] for h in test_edge_index[0].tolist()]
    test_tails = [id2node[t] for t in test_edge_index[1].tolist()]

    test_df = pd.DataFrame({
        "head": test_heads,
        "tail": test_tails
    })

    # forward test edges: (head, rel_name, tail)
    forward_test = test_df.copy()
    forward_test["relation"] = rel_name

    # inverse test edges: (tail, rel_name_inv, head)
    inverse_test = test_df.rename(columns={"head": "tail", "tail": "head"}).copy()
    inverse_test["relation"] = rel_name + "_inv"

    remove_df = pd.concat([forward_test, inverse_test], ignore_index=True)

    # left merge to identify rows to remove
    merged = df.merge(
        remove_df.drop_duplicates(),
        on=["head", "relation", "tail"],
        how="left",
        indicator=True
    )

    return merged.loc[merged["_merge"] == "left_only", ["head", "relation", "tail"]].reset_index(drop=True)

# ---------------------------
# Evaluation
# ---------------------------
@torch.no_grad()
def evaluate_link_prediction(
    model,
    data,
    edge_index,
    node2id,
    device="cuda",
    batch_size=32,
    k_list=[1, 3, 10, 100],
    print_bool=True,
    embeddings=None,
    candidate_mask=None,
    filter_true_tails=None,
):
    """
    filter_true_tails: dict mapping global head ID -> tensor of ALL known true
    tail global IDs (train + test) for the target relation. When provided,
    other true tails are masked out before computing the rank of each target
    tail (standard filtered evaluation protocol).
    """
    model.eval()
    id2node = {v: k for k, v in node2id.items()}

    edge_index = edge_index.to(device)

    num_edges = edge_index.size(1)
    mrr_total = 0.0
    hits_total = {k: 0 for k in k_list}

    # Pre-compute candidate embeddings (filtered to valid tail types if mask provided)
    all_emb = F.normalize(embeddings, p=2, dim=1)
    if candidate_mask is not None:
        candidate_mask = candidate_mask.to(device)
        candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
        candidate_emb = all_emb[candidate_indices]  # [C, D]
        # Build reverse map: global ID -> position in candidate list
        global_to_candidate = torch.full(
            (embeddings.size(0),), -1, dtype=torch.long, device=device
        )
        global_to_candidate[candidate_indices] = torch.arange(
            len(candidate_indices), device=device
        )
    else:
        candidate_indices = None
        candidate_emb = all_emb  # [N, D]
        global_to_candidate = None

    for start in range(0, num_edges, batch_size):
        end = min(start + batch_size, num_edges)
        batch_edge_index = edge_index[:, start:end]
        h_batch = batch_edge_index[0]
        t_batch = batch_edge_index[1]

        h_emb = F.normalize(embeddings[h_batch], p=2, dim=1)
        scores = torch.matmul(h_emb, candidate_emb.T)  # [B, C]

        # Map true tail global IDs to candidate-space indices
        if candidate_indices is not None:
            t_in_candidates = torch.bucketize(t_batch, candidate_indices)
        else:
            t_in_candidates = t_batch

        for i in range(end - start):
            row_scores = scores[i].clone()
            target_cand_idx = t_in_candidates[i]

            # Filtered protocol: mask out OTHER known true tails for this head
            if filter_true_tails is not None:
                head_global = h_batch[i].item()
                if head_global in filter_true_tails:
                    true_tails_global = filter_true_tails[head_global]
                    if global_to_candidate is not None:
                        true_tails_cand = global_to_candidate[true_tails_global]
                        true_tails_cand = true_tails_cand[true_tails_cand >= 0]
                    else:
                        true_tails_cand = true_tails_global
                    # Mask all true tails, then restore the one we're evaluating
                    row_scores[true_tails_cand] = -1e9
                    row_scores[target_cand_idx] = scores[i][target_cand_idx]

            _, ranking = row_scores.sort(descending=True)
            rank = (ranking == target_cand_idx).nonzero(as_tuple=True)[0].item() + 1
            mrr_total += 1.0 / rank
            for k in k_list:
                if rank <= k:
                    hits_total[k] += 1

            if print_bool:
                head_name = id2node[int(h_batch[i].item())]
                tail_name = id2node[int(t_batch[i].item())]
                print(f"Test edge {start+i}: head={head_name} tail={tail_name} rank={rank}")

    mrr = mrr_total / num_edges
    hits = {f"hits@{k}": hits_total[k] / num_edges for k in k_list}
    return mrr, hits


# ---------------------------
# Loss
# ---------------------------
def link_loss_degree_aware_hard_negatives(
    embeddings,
    edge_index,
    prev_embeddings,
    node_degree,
    num_nodes,
    true_tail_tensors,
    base_num_hard_neg=50,
    base_num_random_neg=50,
    temperature=0.10,
    node_weights=None,
    id2node=None,
    global_n_id=None,
    total_num_nodes=None,
    use_node_weights_loss=False,
    neg_type_mask=None,
    prev_norm_candidates=None,
    allowed_indices=None,
    global_to_allowed=None,
):

    src, dst = edge_index
    batch_size = src.size(0)
    device = embeddings.device

    global_srcs = global_n_id[src]   # local -> global
    global_dsts = global_n_id[dst]   # local -> global

    # -----------------------------------------------------
    # Precompute head/tail embeddings (current forward pass — has grad)
    # -----------------------------------------------------
    embeddings_scored = F.normalize(embeddings, p=2, dim=1)
    head_emb = embeddings_scored[src]     # [B, D]
    tail_emb = embeddings_scored[dst]     # [B, D]
    pos_scores = torch.sum(head_emb * tail_emb, dim=1)  # [B]

    if node_weights is not None and use_node_weights_loss:
        head_w = node_weights[global_srcs]
        tail_w = node_weights[global_dsts]
        edge_weights = (head_w + tail_w) / 2
        edge_weights = torch.clamp(edge_weights, 0.05, 10.0)

    # Precompute normalized prev_embeddings for negative scoring.
    # prev_embeddings are computed on train_data (test edges removed),
    # so using them directly for negatives introduces no leakage.
    # Gradients flow only through head_emb (current forward pass).
    prev_emb_norm = F.normalize(prev_embeddings, p=2, dim=1)

    # -----------------------------------------------------
    # Compute distance matrix for hard negatives (single cdist call)
    # prev_norm_candidates and allowed_indices are precomputed per-epoch
    # -----------------------------------------------------
    if prev_norm_candidates is not None and base_num_hard_neg > 0:
        unique_heads, inverse_idx = torch.unique(global_srcs, return_inverse=True)

        with torch.no_grad():
            heads_norm = F.normalize(prev_embeddings[unique_heads], p=2, dim=1)
            dist_matrix = torch.cdist(heads_norm, prev_norm_candidates)
    else:
        dist_matrix = None
        inverse_idx = None

    # -----------------------------------------------------
    # Random negatives (vectorized)
    # Scored via prev_embeddings (global) — no subgraph mapping needed
    # -----------------------------------------------------

    if base_num_random_neg > 0:
        degree_weights = node_degree.float().clone()
        degree_weights[global_srcs] = 0

        # Exclude true tails
        for s in range(batch_size):
            true_tails = true_tail_tensors[global_srcs[s].item()]
            degree_weights[true_tails] = 0

        # Restrict to allowed tail types (e.g. Disease/PhenotypicFeature for "treats")
        if neg_type_mask is not None:
            degree_weights = degree_weights * neg_type_mask.float()

        degree_weights = degree_weights / degree_weights.sum()

        # Sample negatives for all heads
        rand_global_all = torch.multinomial(
            degree_weights,
            batch_size * base_num_random_neg,
            replacement=True
        ).view(batch_size, base_num_random_neg)  # [B, R]

        # Gather from prev_embeddings directly (all global IDs are valid)
        rand_emb = prev_emb_norm[rand_global_all]  # [B, R, D]

    else:
        rand_emb = None

    # -----------------------------------------------------
    # Hard negatives (per-edge sampling)
    # Scored via prev_embeddings (global) — no subgraph mapping needed
    # -----------------------------------------------------
    hard_global = torch.zeros(
        (batch_size, base_num_hard_neg),
        dtype=torch.long,
        device=device
    )
    hard_valid = torch.zeros(
        (batch_size, base_num_hard_neg),
        dtype=torch.bool,
        device=device
    )
    if dist_matrix is not None and base_num_hard_neg > 0:

        for i in range(batch_size):

            global_src = global_srcs[i]
            global_dst = global_dsts[i]

            true_tails = true_tail_tensors[global_src.item()]
            dist_vec = dist_matrix[inverse_idx[i]].clone()

            # Exclude true tails (mapped to allowed_indices space)
            if allowed_indices is not None:
                dst_in_allowed = global_to_allowed[global_dst]
                if dst_in_allowed >= 0:
                    dist_vec[dst_in_allowed] = float("inf")
                tt_mapped = global_to_allowed[true_tails]
                valid = tt_mapped >= 0
                if valid.any():
                    dist_vec[tt_mapped[valid]] = float("inf")
            else:
                dist_vec[global_dst] = float("inf")
                dist_vec[true_tails] = float("inf")

            candidate_k = base_num_hard_neg * 5

            hard_candidates_local = torch.topk(
                -dist_vec,
                k=min(candidate_k, len(dist_vec)-1)
            ).indices

            # Map back to global node IDs
            hard_candidates = allowed_indices[hard_candidates_local]

            candidate_degrees = node_degree[hard_candidates].float()

            degree_weights_h = candidate_degrees + 1e-6
            degree_weights_h = degree_weights_h / degree_weights_h.sum()

            sample_k = min(base_num_hard_neg, len(hard_candidates))

            sampled_idx = torch.multinomial(
                degree_weights_h,
                sample_k,
                replacement=False
            )

            hard_neg_global = hard_candidates[sampled_idx]
            n = min(base_num_hard_neg, len(hard_neg_global))

            if n > 0:
                hard_global[i, :n] = hard_neg_global[:n]
                hard_valid[i, :n] = True

        # Gather from prev_embeddings directly using global IDs
        hard_emb = prev_emb_norm[hard_global]  # [B, H, D]
    else:
        hard_emb = None

    # Free hard-negative scratch memory before score computation / backward
    del dist_matrix

    # -----------------------------------------------------
    # Vectorized negative score computation
    # -----------------------------------------------------
    neg_scores = []

    if hard_emb is not None:
        hard_scores = torch.sum(
            head_emb.unsqueeze(1) * hard_emb,
            dim=2
        )  # [B, H]
        # Mask out unfilled slots so they don't affect the loss
        hard_scores[~hard_valid] = -1e9
        neg_scores.append(hard_scores)

    if rand_emb is not None:
        rand_scores = torch.sum(head_emb.unsqueeze(1) * rand_emb, dim=2)  # [B, R]
        neg_scores.append(rand_scores)

    if len(neg_scores) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    neg_scores = torch.cat(neg_scores, dim=1)  # [B, Nneg]

    # -----------------------------------------------------
    # Vectorized logits and loss
    # -----------------------------------------------------
    logits = torch.cat(
        [pos_scores.unsqueeze(1), neg_scores],
        dim=1
    ) / temperature

    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    if node_weights is not None and use_node_weights_loss:
        loss_per_edge = F.cross_entropy(logits, labels, reduction="none")
        loss = loss_per_edge * edge_weights
    else:
        loss = F.cross_entropy(logits, labels, reduction="none")

    loss = loss.mean()

    return loss
# ---------------------------
# Adaptive loaders
# ---------------------------
def create_adaptive_neighbor_loader_target_heads(
    data,
    seed_nodes,
    train_edge_index,
    batch_size=1024,
    neighbor_sizes=[50, 2],
    alpha=0.75,
    min_weight=0.10,
):
    seed_nodes = torch.as_tensor(seed_nodes, dtype=torch.long).cpu()
    train_edge_index = torch.as_tensor(train_edge_index, dtype=torch.long).cpu()

    num_seed_nodes = seed_nodes.numel()
    if num_seed_nodes == 0:
        raise ValueError("seed_nodes is empty")

    num_nodes = data.num_nodes
    head_counts = torch.zeros(num_nodes, dtype=torch.float)
    train_heads = train_edge_index[0]
    head_counts.scatter_add_(0, train_heads, torch.ones_like(train_heads, dtype=torch.float))

    seed_head_counts = head_counts[seed_nodes]
    weights = torch.clamp(seed_head_counts, min=0.0) + min_weight
    weights = weights.pow(alpha)
    weights = weights / weights.sum()

    loader = NeighborLoader(
        data,
        num_neighbors=neighbor_sizes,
        input_nodes=seed_nodes,
        batch_size=batch_size,
        shuffle=False,
    )

    def adaptive_iter():
        num_batches = (num_seed_nodes + batch_size - 1) // batch_size
        for _ in range(num_batches):
            sampled_idx = torch.multinomial(weights, batch_size, replacement=True)
            sampled_seeds = seed_nodes[sampled_idx]
            yield loader.sample(sampled_seeds)

    loader.__iter__ = adaptive_iter
    return loader, weights



# ---------------------------
# Utilities
# ---------------------------
def compute_node_frequency(edge_index, num_nodes):
    degree = torch.zeros(num_nodes, dtype=torch.long)
    src = edge_index[0]
    dst = edge_index[1]
    degree.scatter_add_(0, src, torch.ones_like(src))
    degree.scatter_add_(0, dst, torch.ones_like(dst))
    return degree


def build_true_tail_dict(edge_index):
    true_dict = defaultdict(set)
    src, dst = edge_index
    for i in range(edge_index.size(1)):
        h = src[i].item()
        t = dst[i].item()
        true_dict[h].add(t)
    return true_dict


def compute_full_embeddings_batched(
    model,
    data,
    node_frequency=None,
    batch_size=4096,
    neighbor_sizes=[-1],
    device="cuda",
    last_epoch=False,
    node_weights=None,
    epoch=None,
    max_epochs=None,
    use_node_weights_embedding=False,
):
    model.eval()

    loader = NeighborLoader(
        data,
        num_neighbors=neighbor_sizes,
        input_nodes=None,
        batch_size=batch_size,
        shuffle=False,
    )

    out_dim = model.conv_layers[-1].out_channels
    out = torch.zeros(data.num_nodes, out_dim, device=device)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            emb = model(
                node_index=batch.n_id,
                edge_index=batch.edge_index,
                edge_type=batch.edge_type,
                node_frequency=node_frequency,
                last_epoch=last_epoch,
                node_weights=node_weights,
                epoch=epoch,
                max_epochs=max_epochs,
                use_node_weights_embedding=use_node_weights_embedding,
            )

            out[batch.n_id[:batch.batch_size]] = emb[:batch.batch_size]

    return out


def save_embeddings(embeddings, path):
    np.save(path, embeddings.detach().cpu().numpy())
    print(f"Saved embeddings to {path}")


def load_embeddings(path, device):
    emb = np.load(path)
    emb = torch.tensor(emb, dtype=torch.float, device=device)
    print(f"Loaded embeddings from {path}")
    return emb



# ---------------------------
# Run configs
# ---------------------------
def _base_training_config(base_epochs):
    return {
        "total_epochs": base_epochs,
        "emb_file": "./model_outputs/base_embeddings.npy",
        "output_node_file": "./model_outputs/node_order_all.txt",
        "output_optimizer": "./model_outputs/optimizer.pt",
        "output_model": "./model_outputs/model.pt",
        "model_training_file": "./model_outputs/training_tracking.tsv",
        "base_random_state_file": "./model_outputs/rng_state.pt",
        "test_edge_file": "./model_outputs/test_edges.tsv",
        "load_model": None,
        "load_optimizer": None,
    }


def _finetune_config(finetune_epochs, fine_tune_dir, fine_tune_ext):
    os.makedirs(os.path.join("./model_outputs", fine_tune_dir), exist_ok=True)
    return {
        "total_epochs": finetune_epochs,
        "emb_file": f"./model_outputs/{fine_tune_dir}/finetuned_embeddings_{fine_tune_ext}.npy",
        "output_node_file": f"./model_outputs/{fine_tune_dir}/node_order_{fine_tune_ext}.txt",
        "output_optimizer": f"./model_outputs/{fine_tune_dir}/optimizer_{fine_tune_ext}.pt",
        "output_model": f"./model_outputs/{fine_tune_dir}/model_{fine_tune_ext}.pt",
        "model_training_file": f"./model_outputs/{fine_tune_dir}/training_tracking_{fine_tune_ext}.tsv",
        "base_random_state_file": "./model_outputs/rng_state.pt",
        "test_edge_file": f"./model_outputs/{fine_tune_dir}/test_edges_{fine_tune_ext}.tsv",
        "load_model": "./model_outputs/model.pt",
        "load_optimizer": "./model_outputs/optimizer.pt",
    }


def _negative_control_config(finetune_epochs):
    return {
        "total_epochs": finetune_epochs,
        "emb_file": "./model_outputs/finetuned_embeddings_negative_control.npy",
        "output_node_file": "./model_outputs/node_order_negative_control.txt",
        "output_optimizer": "./model_outputs/optimizer_negative_control.pt",
        "output_model": "./model_outputs/model_negative_control.pt",
        "model_training_file": "./model_outputs/training_tracking_negative_control.tsv",
        "base_random_state_file": "./model_outputs/rng_state.pt",
        "test_edge_file": "./model_outputs/test_edges_negative_control.tsv",
        "load_model": "./model_outputs/model.pt",
        "load_optimizer": "./model_outputs/optimizer.pt",
    }


def _restore_rng_state(path):
    rng_state = torch.load(path)
    torch.set_rng_state(rng_state["torch_cpu"])
    torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["python"])


# ---------------------------
# Main
# ---------------------------
def main(
    neighbor_sizes=[100],
    batch_size=512,
    eval_batch_size=512,
    emb_batch_size=512,
    n_layers=1,
    emb_dim=1024,
    hidden_dim=1024,
    rel_name="treats",
    base_training=False,
    base_epochs=20,
    finetune_epochs=10,
    negative_control=True,
    use_node_weights=False,
    use_node_weights_embedding=False,
    use_node_weights_loss=False,
    fine_tune_dir="",
    fine_tune_ext="",
    schema_dir="../output/archive",
    target_node=None,
    weight_method="cosine",
    weighting_type="schema",
):
    forced_test_edges_df = pd.read_table("./model_outputs/forced_test_edges.tsv")
    forced_test_edges = list(zip(forced_test_edges_df["head"], forced_test_edges_df["tail"]))

    device = torch.device("cuda:1")
    print("use device", device)
    torch.cuda.empty_cache()

    if base_training:
        cfg = _base_training_config(base_epochs)
        torch.manual_seed(12345)
        np.random.seed(12345)
        random.seed(12345)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(12345)
    elif not negative_control:
        cfg = _finetune_config(finetune_epochs, fine_tune_dir, fine_tune_ext)
        _restore_rng_state(cfg["base_random_state_file"])
    else:
        cfg = _negative_control_config(finetune_epochs)
        _restore_rng_state(cfg["base_random_state_file"])

    total_epochs = cfg["total_epochs"]
    test_edge_file = cfg["test_edge_file"]
    output_node_file = cfg["output_node_file"]
    output_model = cfg["output_model"]
    output_optimizer = cfg["output_optimizer"]
    model_training_file = cfg["model_training_file"]
    emb_file = cfg["emb_file"]

    df = pd.read_csv("../db/edges.tsv", sep="\t")

    #add in inverse edges
    tmp_df = df.copy()
    tmp_df["head"] = df["tail"].tolist()
    tmp_df["tail"] = df["head"].tolist()
    tmp_df["relation"] = df["relation"] + "_inv"
    df = pd.concat([df, tmp_df], ignore_index=True)
    del tmp_df

    entities = pd.concat([df["head"], df["tail"]]).unique()
    print("Number unique entities:", len(entities))
    n_entities = len(entities)
    n_relations = df["relation"].nunique()
    print("Number unique relations:", n_relations)

    node2id, rel2id = build_vocab(df)
    id2node = {v: k for k, v in node2id.items()}

    num_nodes = len(node2id)
    num_relations = len(rel2id)

    # Build type-constraint mask: restrict "treats" tail negatives to Disease + PhenotypicFeature
    nodes_df = pd.read_csv("../db/nodes.csv", low_memory=False)
    tail_types = {"Disease", "PhenotypicFeature"}
    allowed_tail_ids = set(nodes_df.loc[nodes_df["label"].isin(tail_types), "id"])
    neg_type_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for nid in allowed_tail_ids:
        if nid in node2id:
            neg_type_mask[node2id[nid]] = True
    del nodes_df
    print(f"neg_type_mask: {neg_type_mask.sum().item()} allowed tail nodes out of {num_nodes}")

    # ---------------------------------
    # Target-relation split
    # ---------------------------------
    (train_edge_index, train_edge_type), (test_edge_index, test_edge_type) = train_test_split(
        df,
        node2id,
        rel2id,
        rel_name=rel_name,
        test_frac=0.10,
        forced_test_edges=forced_test_edges,
    )

    # ---------------------------------
    # Write test edges
    # ---------------------------------
    relid2name = {v: k for k, v in rel2id.items()}
    test_heads = [id2node[idx] for idx in test_edge_index[0].tolist()]
    test_tails = [id2node[idx] for idx in test_edge_index[1].tolist()]
    test_rels = [relid2name[r] for r in test_edge_type.tolist()]

    test_df = pd.DataFrame(
        {
            "head": test_heads,
            "relation": test_rels,
            "tail": test_tails,
        }
    )
    test_df.to_csv(test_edge_file, sep="\t", index=False)

    print("test", test_edge_index.shape)
    print("train", train_edge_index.shape)

    # ---------------------------------
    # Filtered evaluation: all known true tails per head (train + test)
    # ---------------------------------
    all_treats_edge_index = torch.cat(
        [train_edge_index, test_edge_index], dim=1
    )
    _filter_dict = defaultdict(set)
    for i in range(all_treats_edge_index.size(1)):
        h = all_treats_edge_index[0, i].item()
        t = all_treats_edge_index[1, i].item()
        _filter_dict[h].add(t)
    filter_true_tails = {
        h: torch.tensor(list(tails), dtype=torch.long, device=device)
        for h, tails in _filter_dict.items()
    }
    del _filter_dict
    print(f"Filtered eval: {len(filter_true_tails)} heads with known true tails")

    # ---------------------------------
    # Leakage-free train graph
    # ---------------------------------
    train_graph_df = remove_actual_test_edges_from_graph(
        df=df,
        test_edge_index=test_edge_index,
        node2id=node2id,
        rel_name=rel_name,
    )

    train_graph_edge_index = torch.tensor(
        [
            [node2id[h] for h in train_graph_df["head"]],
            [node2id[t] for t in train_graph_df["tail"]],
        ],
        dtype=torch.long,
    )
    train_graph_edge_type = torch.tensor(
        [rel2id[r] for r in train_graph_df["relation"]],
        dtype=torch.long,
    )
    train_data = Data(
        edge_index=train_graph_edge_index,
        edge_type=train_graph_edge_type,
        num_nodes=num_nodes,
    )

    del train_graph_df
    del df

    # ---------------------------------
    # Train-graph-only tails / degrees
    # ---------------------------------
    train_true_tail_dict = build_true_tail_dict(train_graph_edge_index)
    true_tail_tensors = {
        k: torch.tensor(list(v), device=device) for k, v in train_true_tail_dict.items()
    }

    node_frequency = compute_node_frequency(train_graph_edge_index, num_nodes).to(device)
    neg_type_mask = neg_type_mask.to(device)

    # ---------------------------------
    # Node weights
    # ---------------------------------
    node_weights = None
    if not base_training and not negative_control and use_node_weights:
        from context_utils import (
            load_latest_schema,
            load_classified_nodes_from_file,
            get_controlled_fields,
            build_context_vectors,
            compute_schema_weights,
        )

        if target_node is None:
            raise ValueError("--target_node is required when --weighting_type=schema")
        if target_node not in node2id:
            raise ValueError(f"target_node '{target_node}' not found in graph")

        schema = load_latest_schema(schema_dir)
        classified_nodes = load_classified_nodes_from_file(
            "../output/drug_disease/nodes_9.json"
        )
        controlled_fields = get_controlled_fields(schema)
        context_vectors, has_context, _ = build_context_vectors(
            node2id, classified_nodes, controlled_fields
        )
        node_weights = compute_schema_weights(
            context_vectors,
            has_context,
            target_idx=node2id[target_node],
            method=weight_method,
        )
        node_weights = node_weights.to(device)


    # ---------------------------------
    # Model
    # ---------------------------------
    model = ConfigurableRGCN(
        n_entities,
        n_relations,
        emb_dim,
        hidden_dim,
        emb_dim,
        n_layers,
        device=device,
    )

    print(model)

    # Precompute allowed tail indices and mapping for hard negatives
    allowed_indices = neg_type_mask.nonzero(as_tuple=True)[0]
    global_to_allowed = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    global_to_allowed[allowed_indices] = torch.arange(len(allowed_indices), device=device)

    def precompute_hard_neg_cache(prev_emb):
        """Normalize allowed tail embeddings once per epoch."""
        with torch.no_grad():
            return F.normalize(prev_emb[allowed_indices], p=2, dim=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    update_embeddings = 1

    if cfg["load_model"] is not None:
        model.load_state_dict(torch.load(cfg["load_model"], map_location=device))
        optimizer.load_state_dict(torch.load(cfg["load_optimizer"], map_location=device))
        print("Initialized model with pretrained embeddings")
        prev_embeddings = load_embeddings("./model_outputs/base_embeddings.npy", device)
    else:
        with torch.no_grad():
            prev_embeddings = model.node_emb.detach().clone()

    prev_norm_candidates = precompute_hard_neg_cache(prev_embeddings)

    id2node_write = [None] * model.num_ent
    for name, idx in node2id.items():
        id2node_write[idx] = name

    with open(output_node_file, "w") as f:
        f.write("\n".join(id2node_write))
    print(f"Node order saved to {output_node_file}")

    # ---------------------------------
    # Seed nodes from target-relation train edges only
    # ---------------------------------
    train_nodes = torch.unique(train_edge_index.flatten())

    train_loader_adaptive, seed_sampling_weights = create_adaptive_neighbor_loader_target_heads(
        data=train_data,
        seed_nodes=train_nodes,
        train_edge_index=train_edge_index,
        batch_size=batch_size,
        neighbor_sizes=neighbor_sizes,
        alpha=0.5,
        min_weight=0.10,
    )

    if base_training or negative_control or not use_node_weights:
        node_weights = None

    training_tracking_dict = {
        "loss": [],
        "training_mrr": [],
        "testing_mrr": [],
        "use_node_weights": use_node_weights,
    }

    print("Node Weight in Use:", use_node_weights)

    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        print("Start epoch", epoch + 1)

        model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in train_loader_adaptive:
            batch = batch.to(device)
            optimizer.zero_grad()

            embeddings = model(
                node_index=batch.n_id,
                edge_index=batch.edge_index,
                edge_type=batch.edge_type,
                node_frequency=node_frequency,
                node_weights=node_weights,
                epoch=epoch,
                max_epochs=total_epochs,
                use_node_weights_embedding=use_node_weights_embedding,
            )

            rel_id = rel2id[rel_name]
            main_mask = batch.edge_type == rel_id
            pos_edge_index = batch.edge_index[:, main_mask]

            if pos_edge_index.size(1) == 0:
                continue

            if prev_embeddings is None:
                base_num_hard_neg = 0
                base_num_random_neg = 50
            else:
                base_num_hard_neg = 25
                base_num_random_neg = 50

            loss = link_loss_degree_aware_hard_negatives(
                embeddings=embeddings,
                edge_index=pos_edge_index,
                prev_embeddings=prev_embeddings,
                node_degree=node_frequency,
                num_nodes=len(batch.n_id),
                true_tail_tensors=true_tail_tensors,
                base_num_hard_neg=base_num_hard_neg,
                base_num_random_neg=base_num_random_neg,
                temperature=0.20,
                node_weights=node_weights,
                id2node=id2node,
                global_n_id=batch.n_id,
                total_num_nodes=model.num_ent,
                use_node_weights_loss=use_node_weights_loss,
                neg_type_mask=neg_type_mask,
                prev_norm_candidates=prev_norm_candidates,
                allowed_indices=allowed_indices,
                global_to_allowed=global_to_allowed,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0

        train_mrr = "None"
        test_mrr = "None"

        if (((epoch + 1) % update_embeddings == 0) or epoch < 2):
            with torch.no_grad():
                prev_embeddings = compute_full_embeddings_batched(
                    model,
                    train_data,
                    batch_size=emb_batch_size,
                    neighbor_sizes=neighbor_sizes,
                    device=device,
                    node_frequency=node_frequency,
                    node_weights=node_weights,
                    epoch=epoch,
                    max_epochs=total_epochs,
                    last_epoch=True,
                    use_node_weights_embedding=use_node_weights_embedding,
                )
                prev_norm_candidates = precompute_hard_neg_cache(prev_embeddings)

        if (epoch + 1) % update_embeddings == 0:
            print(f"Epoch {epoch+1}/{total_epochs}, Loss: {avg_loss:.4f}")

            with torch.no_grad():
                std_val = torch.std(prev_embeddings, dim=0).mean().item()
                print("Embedding std:", std_val)

                train_mrr, train_hits = evaluate_link_prediction(
                    model=model,
                    data=train_data,
                    edge_index=train_edge_index,
                    node2id=node2id,
                    device=device,
                    batch_size=eval_batch_size,
                    k_list=[1, 3, 10, 100],
                    print_bool=False,
                    embeddings=prev_embeddings,
                    candidate_mask=neg_type_mask,
                    filter_true_tails=filter_true_tails,
                )
                print(f"Train MRR: {train_mrr:.4f}")

                test_mrr, test_hits = evaluate_link_prediction(
                    model=model,
                    data=train_data,
                    edge_index=test_edge_index,
                    node2id=node2id,
                    device=device,
                    batch_size=eval_batch_size,
                    k_list=[1, 3, 10, 100],
                    print_bool=False,
                    embeddings=prev_embeddings,
                    candidate_mask=neg_type_mask,
                    filter_true_tails=filter_true_tails,
                )
                print(f"Test MRR: {test_mrr:.4f}")

        training_tracking_dict["loss"].append(avg_loss)
        training_tracking_dict["training_mrr"].append(train_mrr)
        training_tracking_dict["testing_mrr"].append(test_mrr)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1}/{total_epochs} finished in {epoch_duration:.2f} seconds")

    # ---------------------------------
    # Final embeddings: leakage-free
    # ---------------------------------
    embeddings = compute_full_embeddings_batched(
        model,
        train_data,
        batch_size=emb_batch_size,
        neighbor_sizes=neighbor_sizes,
        device=device,
        node_frequency=node_frequency,
        last_epoch=True,
        node_weights=node_weights,
        epoch=epoch,
        max_epochs=total_epochs,
        use_node_weights_embedding=use_node_weights_embedding,
    )

    save_embeddings(embeddings, emb_file)
    torch.save(model.state_dict(), output_model)
    torch.save(optimizer.state_dict(), output_optimizer)
    training_df = pd.DataFrame(training_tracking_dict)
    training_df.to_csv(model_training_file, sep="\t", index=None)

    if base_training:
        rng_state = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        torch.save(rng_state, cfg["base_random_state_file"])

    # ---------------------------------
    # Final evaluation
    # ---------------------------------
    mrr, hits = evaluate_link_prediction(
        model,
        train_data,
        test_edge_index,
        node2id,
        device=device,
        batch_size=eval_batch_size,
        embeddings=embeddings,
        print_bool=False,
        candidate_mask=neg_type_mask,
        filter_true_tails=filter_true_tails,
    )

    print(f"\nTest MRR: {mrr:.4f}")
    for k, v in hits.items():
        try:
            print(f"{k}: {v:.4f}")
        except:
            print("not working")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["base", "finetune", "negative_control"],
        default="base",
    )
    parser.add_argument("--use_node_weights", action="store_true")
    parser.add_argument("--use_node_weights_embedding", action="store_true")
    parser.add_argument("--use_node_weights_loss", action="store_true")
    parser.add_argument("--weighting_type", type=str, default=None)
    parser.add_argument("--fine_tune_dir", type=str, default="")
    parser.add_argument("--fine_tune_ext", type=str, default="")
    parser.add_argument("--schema_dir", type=str, default="../output/archive")
    parser.add_argument("--target_node", type=str, default=None)
    parser.add_argument("--weight_method", type=str, default="cosine",
                        choices=["cosine", "overlap"])
    args = parser.parse_args()

    main(
        base_training=(args.mode == "base"),
        negative_control=(args.mode == "negative_control"),
        use_node_weights=args.use_node_weights,
        use_node_weights_embedding=args.use_node_weights_embedding,
        use_node_weights_loss=args.use_node_weights_loss,
        fine_tune_dir=args.fine_tune_dir,
        fine_tune_ext=args.fine_tune_ext,
        schema_dir=args.schema_dir,
        target_node=args.target_node,
        weight_method=args.weight_method,
    )


