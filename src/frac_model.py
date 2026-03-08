"""Fractional-Order Physics-Informed Graph Diffusion for DTI Prediction.

Two model variants:
  1. FracDTI      — Direction 1: global fractional orders (α_drug ≠ α_prot)
  2. FracAdaptDTI — Direction 1+3: per-node adaptive α_i via hypernetwork

Core theory:
  Fractional diffusion filter in spectral domain:
      h(λ) = (1 + t·λ)^{-α}

  where λ are Laplacian eigenvalues, α ∈ (0, 3] is the fractional order,
  t > 0 is diffusion time/scale.

  Polynomial approximation via Taylor expansion in adjacency eigenvalues:
      X* = Σ_{k=0}^{K} c_k(α, t) · Ã^k · X₀

  where c_k(α, t) = (1+t)^{-α} · [α(α+1)···(α+k-1) / k!] · (t/(1+t))^k

  Special cases:
    α = 1, t → ∞  ⟹  PPR (Personalized PageRank)
    α → 0          ⟹  identity (no diffusion)
    α = 1          ⟹  standard heat-kernel diffusion

  Learnable α allows the model to discover the optimal diffusion order
  *per node type* (drug vs protein), revealing asymmetric diffusion dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Sparse helpers (shared with e2e_model.py)
# ---------------------------------------------------------------------------

def scipy_sparse_to_torch(adj_sp, device="cpu"):
    adj_coo = adj_sp.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.vstack([adj_coo.row, adj_coo.col]))
    values = torch.FloatTensor(adj_coo.data)
    shape = torch.Size(adj_coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to(device)


def enhance_adj_knn(adj, features, drug_num, prot_num, k=10, weighted=False):
    """Enhance adjacency by adding Top-K cosine-similarity KNN edges
    within drug-drug and protein-protein subsets.

    Args:
        weighted: if True, use actual cosine similarity as edge weight
                  instead of binary 1.0.

    Returns a new sparse adjacency (union of original + KNN edges).
    """
    feat_np = features.numpy() if torch.is_tensor(features) else np.array(features)
    drug_feat = feat_np[:drug_num]
    prot_feat = feat_np[drug_num:drug_num + prot_num]
    n_total = adj.shape[0]
    enhanced = adj.tolil(copy=True)

    for offset, feat_block in [(0, drug_feat), (drug_num, prot_feat)]:
        n = feat_block.shape[0]
        if n <= 1 or k <= 0:
            continue
        actual_k = min(k + 1, n)  # +1 because self is the closest
        batch_sz = 512
        for start in range(0, n, batch_sz):
            end = min(start + batch_sz, n)
            sim = cosine_similarity(feat_block[start:end], feat_block)  # (batch, n)
            for i_local in range(end - start):
                row = sim[i_local]
                row[start + i_local] = -1.0  # exclude self
                topk_idx = np.argpartition(row, -actual_k + 1)[-(actual_k - 1):]
                gi = offset + start + i_local
                for j_local in topk_idx:
                    gj = offset + j_local
                    w = max(float(row[j_local]), 0.01) if weighted else 1.0
                    enhanced[gi, gj] = w
                    enhanced[gj, gi] = w

    print("  Graph enhanced (KNN k={}, weighted={}): nnz {} -> {}".format(
        k, weighted, adj.nnz, enhanced.nnz))
    return enhanced.tocsr()


def enhance_adj_full(adj, features, drug_num, prot_num, threshold=0.5):
    """Enhance adjacency by adding ALL cosine-similarity edges above
    a threshold within drug-drug and protein-protein subsets.

    Returns a new sparse adjacency (union of original + similarity edges).
    """
    feat_np = features.numpy() if torch.is_tensor(features) else np.array(features)
    drug_feat = feat_np[:drug_num]
    prot_feat = feat_np[drug_num:drug_num + prot_num]
    enhanced = adj.tolil(copy=True)

    for offset, feat_block, label in [(0, drug_feat, "drug"), (drug_num, prot_feat, "prot")]:
        n = feat_block.shape[0]
        if n <= 1:
            continue
        added = 0
        batch_sz = 512
        for start in range(0, n, batch_sz):
            end = min(start + batch_sz, n)
            sim = cosine_similarity(feat_block[start:end], feat_block)  # (batch, n)
            rows, cols = np.where(sim >= threshold)
            for idx in range(len(rows)):
                gi = offset + start + rows[idx]
                gj = offset + cols[idx]
                if gi != gj and enhanced[gi, gj] == 0:
                    enhanced[gi, gj] = 1.0
                    enhanced[gj, gi] = 1.0
                    added += 1
        print("    {} edges added for {} (threshold={})".format(added, label, threshold))

    print("  Graph enhanced (full, thr={}): nnz {} -> {}".format(
        threshold, adj.nnz, enhanced.nnz))
    return enhanced.tocsr()


def build_bipartite_adj(pairs, n_total):
    """Build adjacency from ONLY drug-protein pairs (pure bipartite).
    No drug-drug or protein-protein similarity edges."""
    adj = sp.lil_matrix((n_total, n_total), dtype=np.float32)
    for d, p in pairs:
        adj[d, p] = 1.0
        adj[p, d] = 1.0
    return adj.tocsr()


def compute_norm_adj(adj, self_loop=True, weighted=False):
    if sp.issparse(adj):
        adj = adj.tocsr()
    else:
        adj = sp.csr_matrix(adj)
    if weighted:
        adj = adj.maximum(adj.T)
    else:
        adj = adj + adj.T
        adj.data = np.clip(adj.data, 0, 1)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
    return (d_inv_sqrt @ adj @ d_inv_sqrt).tocsr()


# ---------------------------------------------------------------------------
# Fractional diffusion coefficient computation
# ---------------------------------------------------------------------------

def _frac_coefficients(alpha, t, K):
    """Compute c_k(α, t) for k = 0 .. K.  Fully differentiable in α and t.

    Returns list of K+1 tensors (same shape as alpha/t).
    """
    alpha = alpha.clamp(min=0.05, max=3.0)
    t = t.clamp(min=0.01, max=10.0)

    s = t / (1.0 + t)
    base = (1.0 + t).pow(-alpha)

    coeffs = [base]
    rising = torch.ones_like(alpha)
    s_pow = torch.ones_like(alpha)

    for k in range(1, K + 1):
        rising = rising * (alpha + k - 1) / k
        s_pow = s_pow * s
        coeffs.append(base * rising * s_pow)

    return coeffs


# ---------------------------------------------------------------------------
# Direction 1: FracGraphFilter — type-level fractional parameters
# ---------------------------------------------------------------------------

class FracGraphFilter(nn.Module):
    """Fractional graph filter with separate (α, t) for drug / protein.

    Learns 4 scalar parameters: α_drug, α_prot, t_drug, t_prot.
    After training, comparing α_drug vs α_prot reveals whether drugs and
    proteins require fundamentally different diffusion dynamics.
    """

    def __init__(self, K=15, drug_num=0):
        super().__init__()
        self.K = K
        self.drug_num = drug_num

        self.log_alpha_drug = nn.Parameter(torch.tensor(0.0))   # softplus → ~0.69
        self.log_alpha_prot = nn.Parameter(torch.tensor(0.0))
        self.log_t_drug = nn.Parameter(torch.tensor(0.0))
        self.log_t_prot = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha_drug(self):
        return F.softplus(self.log_alpha_drug)

    @property
    def alpha_prot(self):
        return F.softplus(self.log_alpha_prot)

    @property
    def t_drug(self):
        return F.softplus(self.log_t_drug)

    @property
    def t_prot(self):
        return F.softplus(self.log_t_prot)

    def forward(self, x, adj_norm_sparse):
        drug_coeffs = _frac_coefficients(self.alpha_drug, self.t_drug, self.K)
        prot_coeffs = _frac_coefficients(self.alpha_prot, self.t_prot, self.K)

        out = torch.zeros_like(x)
        power = x

        for k in range(self.K + 1):
            c_k = torch.empty(x.shape[0], 1, device=x.device)
            c_k[:self.drug_num] = drug_coeffs[k]
            c_k[self.drug_num:] = prot_coeffs[k]
            out = out + c_k * power
            if k < self.K:
                power = torch.sparse.mm(adj_norm_sparse, power)

        return out

    def get_params(self):
        return dict(
            alpha_drug=self.alpha_drug.item(),
            alpha_prot=self.alpha_prot.item(),
            t_drug=self.t_drug.item(),
            t_prot=self.t_prot.item(),
        )


# ---------------------------------------------------------------------------
# Direction 1+3: FracAdaptFilter — node-adaptive fractional parameters
# ---------------------------------------------------------------------------

class FracAdaptFilter(nn.Module):
    """Node-adaptive fractional graph filter.

    A lightweight hypernetwork maps each node's context (projected features,
    log-degree, type indicator) to per-node (α_i, t_i).

    This captures fine-grained differences: e.g. hub drugs need broader
    diffusion than orphan drugs, essential proteins need deeper propagation.
    """

    def __init__(self, nfeat, K=15, drug_num=0, hyper_hidden=32, feat_proj_dim=16):
        super().__init__()
        self.K = K
        self.drug_num = drug_num

        self.feat_proj = nn.Linear(nfeat, feat_proj_dim)
        hyper_in = feat_proj_dim + 2  # +1 log_degree, +1 type_indicator

        self.hyper = nn.Sequential(
            nn.Linear(hyper_in, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, 2),
        )

        self.alpha_bias_drug = nn.Parameter(torch.tensor(0.0))
        self.alpha_bias_prot = nn.Parameter(torch.tensor(0.0))
        self.t_bias = nn.Parameter(torch.tensor(0.0))

    def _build_context(self, x, adj_norm_sparse):
        N = x.shape[0]
        device = x.device

        feat_low = self.feat_proj(x)

        deg = torch.sparse.sum(adj_norm_sparse, dim=1).to_dense().unsqueeze(1)
        log_deg = torch.log1p(deg)

        ntype = torch.zeros(N, 1, device=device)
        ntype[self.drug_num:] = 1.0

        return torch.cat([feat_low, log_deg, ntype], dim=1)

    def forward(self, x, adj_norm_sparse):
        N = x.shape[0]
        device = x.device

        ctx = self._build_context(x, adj_norm_sparse)
        raw = self.hyper(ctx)  # (N, 2)

        alpha_bias = torch.zeros(N, device=device)
        alpha_bias[:self.drug_num] = self.alpha_bias_drug
        alpha_bias[self.drug_num:] = self.alpha_bias_prot

        alpha = F.softplus(raw[:, 0] + alpha_bias) + 0.05
        t = F.softplus(raw[:, 1] + self.t_bias) + 0.01

        coeffs = _frac_coefficients(alpha, t, self.K)

        out = torch.zeros_like(x)
        power = x

        for k in range(self.K + 1):
            c_k = coeffs[k].unsqueeze(1)  # (N, 1)
            out = out + c_k * power
            if k < self.K:
                power = torch.sparse.mm(adj_norm_sparse, power)

        return out, alpha, t


# ---------------------------------------------------------------------------
# Shared encoder & predictor
# ---------------------------------------------------------------------------

class NodeEncoder(nn.Module):
    def __init__(self, nfeat, hidden=256, embed_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nfeat, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class LinkPredictor(nn.Module):
    def __init__(self, embed_dim=128, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_drug, h_prot):
        pair = torch.cat([h_drug, h_prot, h_drug * h_prot], dim=1)
        return self.net(pair).squeeze(-1)


# ---------------------------------------------------------------------------
# Full models
# ---------------------------------------------------------------------------

class FracDTI(nn.Module):
    """Direction 1: Global fractional-order DTI model.

    4 key learnable physics parameters: α_drug, α_prot, t_drug, t_prot
    """

    def __init__(self, nfeat, drug_num, hidden=256, embed_dim=128,
                 pred_hidden=64, K=15, dropout=0.3):
        super().__init__()
        self.graph_filter = FracGraphFilter(K=K, drug_num=drug_num)
        self.encoder = NodeEncoder(nfeat, hidden, embed_dim, dropout)
        self.predictor = LinkPredictor(embed_dim, pred_hidden, dropout)

    def get_embeddings(self, x, adj):
        filtered = self.graph_filter(x, adj)
        return self.encoder(filtered)

    def forward(self, x, adj, drug_idx, prot_idx):
        emb = self.get_embeddings(x, adj)
        return self.predictor(emb[drug_idx], emb[prot_idx])

    def predict_proba(self, x, adj, drug_idx, prot_idx):
        return torch.sigmoid(self.forward(x, adj, drug_idx, prot_idx))

    def get_diffusion_params(self):
        return self.graph_filter.get_params()


class FracAdaptDTI(nn.Module):
    """Direction 1+3: Node-adaptive fractional-order DTI model.

    Per-node (α_i, t_i) generated by hypernetwork conditioned on features,
    degree, and node type.
    """

    def __init__(self, nfeat, drug_num, hidden=256, embed_dim=128,
                 pred_hidden=64, K=15, dropout=0.3, hyper_hidden=32):
        super().__init__()
        self.graph_filter = FracAdaptFilter(
            nfeat, K=K, drug_num=drug_num, hyper_hidden=hyper_hidden)
        self.encoder = NodeEncoder(nfeat, hidden, embed_dim, dropout)
        self.predictor = LinkPredictor(embed_dim, pred_hidden, dropout)
        self.drug_num = drug_num

    def get_embeddings(self, x, adj):
        filtered, alpha, t = self.graph_filter(x, adj)
        return self.encoder(filtered), alpha, t

    def forward(self, x, adj, drug_idx, prot_idx):
        emb, alpha, t = self.get_embeddings(x, adj)
        logits = self.predictor(emb[drug_idx], emb[prot_idx])
        return logits, alpha, t

    def predict_proba(self, x, adj, drug_idx, prot_idx):
        logits, _, _ = self.forward(x, adj, drug_idx, prot_idx)
        return torch.sigmoid(logits)

    def get_diffusion_stats(self, x, adj):
        _, alpha, t = self.graph_filter(x, adj)
        da = alpha[:self.drug_num]
        pa = alpha[self.drug_num:]
        dt = t[:self.drug_num]
        pt = t[self.drug_num:]
        return dict(
            drug_alpha_mean=da.mean().item(), drug_alpha_std=da.std().item(),
            prot_alpha_mean=pa.mean().item(), prot_alpha_std=pa.std().item(),
            drug_t_mean=dt.mean().item(), drug_t_std=dt.std().item(),
            prot_t_mean=pt.mean().item(), prot_t_std=pt.std().item(),
        )
