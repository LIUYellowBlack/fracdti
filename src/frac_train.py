"""Training & evaluation for Fractional-Order DTI models.

Supports:
  --model frac       → FracDTI      (Direction 1: global α_drug, α_prot)
  --model fracadapt  → FracAdaptDTI (Direction 1+3: per-node adaptive α)

  --mode warm  → 10-fold CV (transductive)
  --mode cold  → hold-out drug / target entities (inductive)

After training, prints and saves the learned diffusion parameters (α, t)
for scientific analysis.

Usage:
  python frac_train.py --model frac --dataset biosnap --mode warm --device cuda:0
  python frac_train.py --model fracadapt --dataset biosnap --mode warm --device cuda:0
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.model_selection import StratifiedKFold

from utils import load_data
from frac_model import (
    FracDTI, FracAdaptDTI,
    compute_norm_adj, scipy_sparse_to_torch,
    enhance_adj_knn, enhance_adj_full, build_bipartite_adj,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_true, y_score):
    """Find threshold that maximizes F1 score."""
    from sklearn.metrics import precision_recall_curve
    precs, recs, thresholds = precision_recall_curve(y_true, y_score)
    f1s = np.where((precs + recs) > 0, 2 * precs * recs / (precs + recs), 0)
    best_idx = np.argmax(f1s)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


def evaluate_predictions(y_true, y_score, threshold=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    aupr = average_precision_score(y_true, y_score)

    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    y_true = np.array(y_true).astype(int)
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())

    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0

    return dict(auc=roc_auc, aupr=aupr, f1=f1, sens=sens, spec=spec, prec=prec, acc=acc,
                threshold=threshold)


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_negatives(pos_pairs, n_drug, n_prot, drug_offset, prot_offset,
                     positive_set, rng, ratio=1):
    needed = len(pos_pairs) * ratio
    neg = []
    while len(neg) < needed:
        d = rng.randint(drug_offset, drug_offset + n_drug)
        p = rng.randint(prot_offset, prot_offset + n_prot)
        if (d, p) not in positive_set:
            neg.append([d, p])
    return np.array(neg[:needed])


def focal_bce_loss(logits, labels, gamma=2.0, label_smoothing=0.0):
    """Focal loss with optional label smoothing."""
    if label_smoothing > 0:
        labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


def hard_negative_mining(model, features, adj_sp, pos_pairs, drug_num, prot_num,
                         positive_set, device, n_needed, hard_ratio=0.5):
    """Mine hard negatives: pairs the model incorrectly predicts as positive."""
    n_hard = int(n_needed * hard_ratio)
    n_rand = n_needed - n_hard
    pool_size = min(n_hard * 5, drug_num * prot_num // 10)

    cands = []
    rng = np.random.RandomState()
    while len(cands) < pool_size:
        d = rng.randint(0, drug_num)
        p = rng.randint(drug_num, drug_num + prot_num)
        if (d, p) not in positive_set:
            cands.append([d, p])
    cands = np.array(cands)

    model.eval()
    with torch.no_grad():
        scores = []
        for s in range(0, len(cands), 4096):
            e = min(s + 4096, len(cands))
            bp = cands[s:e]
            di = torch.LongTensor(bp[:, 0]).to(device)
            pi = torch.LongTensor(bp[:, 1]).to(device)
            sc = model.predict_proba(features, adj_sp, di, pi)
            scores.append(sc.cpu().numpy())
        scores = np.concatenate(scores)

    hard_idx = np.argsort(scores)[-n_hard:]
    hard_neg = cands[hard_idx]

    rand_neg = sample_negatives(pos_pairs, drug_num, prot_num, 0, drug_num,
                                positive_set, rng)[:n_rand]
    return np.concatenate([hard_neg, rand_neg])


# ---------------------------------------------------------------------------
# Remove test edges from adjacency
# ---------------------------------------------------------------------------

def remove_edges(adj, pairs):
    if len(pairs) == 0:
        return adj
    lil = adj.tolil(copy=True)
    for d, p in pairs:
        lil[d, p] = 0.0
        lil[p, d] = 0.0
    return lil.tocsr()


def build_bipartite_adj(pairs, n_total):
    """Build adjacency from ONLY the given drug-protein pairs (pure bipartite).
    No drug-drug or protein-protein similarity edges."""
    adj = sp.lil_matrix((n_total, n_total), dtype=np.float32)
    for d, p in pairs:
        adj[d, p] = 1.0
        adj[p, d] = 1.0
    return adj.tocsr()


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(args, nfeat, drug_num, device):
    common = dict(nfeat=nfeat, drug_num=drug_num, hidden=args.hidden,
                  embed_dim=args.embed_dim, pred_hidden=args.pred_hidden,
                  K=args.K, dropout=args.dropout,
                  appnp_alpha=args.appnp_alpha)
    if args.model == "frac":
        model = FracDTI(**common)
        if args.alpha_init != 0.0:
            with torch.no_grad():
                model.graph_filter.log_alpha_drug.fill_(args.alpha_init)
                model.graph_filter.log_alpha_prot.fill_(args.alpha_init)
                model.graph_filter.log_t_drug.fill_(args.alpha_init)
                model.graph_filter.log_t_prot.fill_(args.alpha_init)
    elif args.model == "fracadapt":
        model = FracAdaptDTI(hyper_hidden=args.hyper_hidden, **common)
    else:
        raise ValueError("Unknown model: {}".format(args.model))
    return model.to(device)


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_epoch(model, optimizer, features, adj_sp, pos_pairs, neg_pairs,
                device, batch_size, model_type,
                use_focal=False, focal_gamma=2.0, label_smoothing=0.0):
    model.train()
    pairs = np.concatenate([pos_pairs, neg_pairs])
    labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))])
    perm = np.random.permutation(len(pairs))
    pairs, labels = pairs[perm], labels[perm]

    total_loss = 0.0
    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        bp = pairs[start:end]
        bl = torch.FloatTensor(labels[start:end]).to(device)
        di = torch.LongTensor(bp[:, 0]).to(device)
        pi = torch.LongTensor(bp[:, 1]).to(device)

        if model_type == "fracadapt":
            logits, _, _ = model(features, adj_sp, di, pi)
        else:
            logits = model(features, adj_sp, di, pi)

        if use_focal:
            loss = focal_bce_loss(logits, bl, gamma=focal_gamma,
                                  label_smoothing=label_smoothing)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, bl)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(bp)

    return total_loss / len(pairs)


# ---------------------------------------------------------------------------
# Eval pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_model(model, features, adj_sp, pairs, device, model_type, batch_size=4096):
    model.eval()
    if len(pairs) == 0:
        return np.array([])
    scores = []
    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        bp = pairs[start:end]
        di = torch.LongTensor(bp[:, 0]).to(device)
        pi = torch.LongTensor(bp[:, 1]).to(device)
        proba = model.predict_proba(features, adj_sp, di, pi)
        scores.append(proba.cpu().numpy())
    return np.concatenate(scores)


@torch.no_grad()
def eval_cold_with_tta(model, features, adj_scipy, pairs, device,
                       model_type, weighted=False, conf_thr=0.8,
                       blend=0.6, batch_size=4096):
    """Test-time augmentation for cold-start: high-confidence predictions
    become temporary edges, then re-run inference on the augmented graph."""
    model.eval()

    adj_norm = compute_norm_adj(adj_scipy, weighted=weighted)
    adj_sp = scipy_sparse_to_torch(adj_norm, device)

    scores_r1 = []
    for s in range(0, len(pairs), batch_size):
        e = min(s + batch_size, len(pairs))
        bp = pairs[s:e]
        di = torch.LongTensor(bp[:, 0]).to(device)
        pi = torch.LongTensor(bp[:, 1]).to(device)
        scores_r1.append(model.predict_proba(features, adj_sp, di, pi).cpu().numpy())
    scores_r1 = np.concatenate(scores_r1)

    high_conf = scores_r1 > conf_thr
    n_aug = int(high_conf.sum())
    if n_aug < 1:
        return scores_r1

    new_edges = pairs[high_conf]
    adj_aug = adj_scipy.tolil(copy=True)
    for d, p in new_edges:
        adj_aug[d, p] = 1.0
        adj_aug[p, d] = 1.0
    adj_aug = adj_aug.tocsr()
    adj_norm_aug = compute_norm_adj(adj_aug, weighted=weighted)
    adj_sp_aug = scipy_sparse_to_torch(adj_norm_aug, device)

    scores_r2 = []
    for s in range(0, len(pairs), batch_size):
        e = min(s + batch_size, len(pairs))
        bp = pairs[s:e]
        di = torch.LongTensor(bp[:, 0]).to(device)
        pi = torch.LongTensor(bp[:, 1]).to(device)
        scores_r2.append(model.predict_proba(features, adj_sp_aug, di, pi).cpu().numpy())
    scores_r2 = np.concatenate(scores_r2)

    print("    TTA: {} high-conf edges added, blending {:.0f}% R2 + {:.0f}% R1".format(
        n_aug, blend * 100, (1 - blend) * 100))
    return blend * scores_r2 + (1 - blend) * scores_r1


# ---------------------------------------------------------------------------
# Warm-start 10-fold CV
# ---------------------------------------------------------------------------

def run_warm(args, feat_t, adj_full, pos_pairs, neg_pairs_all, drug_num, prot_num):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nfeat = feat_t.shape[1]
    pos_set = set(map(tuple, pos_pairs))

    all_pairs = np.concatenate([pos_pairs, neg_pairs_all])
    all_labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs_all))])

    skf = StratifiedKFold(n_splits=args.k_fold, random_state=args.seed, shuffle=True)
    metrics_list = []
    diffusion_params_list = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(all_pairs, all_labels)):
        print("\n--- Fold {}/{} ---".format(fold + 1, args.k_fold))

        tr_pairs, tr_labels = all_pairs[tr_idx], all_labels[tr_idx]
        te_pairs, te_labels = all_pairs[te_idx], all_labels[te_idx]

        # Split training into actual_train + val (val for early stopping ONLY)
        val_ratio = 0.1
        fold_rng = np.random.RandomState(args.seed + fold)
        n_tr = len(tr_pairs)
        perm = fold_rng.permutation(n_tr)
        n_val = max(1, int(val_ratio * n_tr))
        val_idx = perm[:n_val]
        actual_tr_idx = perm[n_val:]

        val_pairs = tr_pairs[val_idx]
        val_labels = tr_labels[val_idx]
        actual_tr_pairs = tr_pairs[actual_tr_idx]
        actual_tr_labels = tr_labels[actual_tr_idx]

        tr_pos = actual_tr_pairs[actual_tr_labels == 1]
        te_pos = te_pairs[te_labels == 1]

        if args.no_edge_removal:
            adj_fold = adj_full
        else:
            adj_fold = remove_edges(adj_full, te_pos)
        adj_norm = compute_norm_adj(adj_fold, weighted=args.weighted_knn)
        adj_sp = scipy_sparse_to_torch(adj_norm, device)
        feat_dev = feat_t.to(device)

        print("  Train: {} (pos {}), Val: {}, Test: {}".format(
            len(actual_tr_pairs), len(tr_pos), len(val_pairs), len(te_pairs)))

        model = build_model(args, nfeat, drug_num, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        best_val_auc = 0.0
        patience_cnt = 0
        best_state = None

        for epoch in range(args.epochs):
            if args.hard_neg and epoch >= args.hard_neg_start:
                epoch_neg = hard_negative_mining(
                    model, feat_dev, adj_sp, tr_pos, drug_num, prot_num,
                    pos_set, device, len(tr_pos), args.hard_neg_ratio)
            else:
                rng = np.random.RandomState(args.seed + epoch)
                epoch_neg = sample_negatives(
                    tr_pos, drug_num, prot_num, 0, drug_num, pos_set, rng)

            loss = train_epoch(model, optimizer, feat_dev, adj_sp,
                               tr_pos, epoch_neg, device, args.batch_size,
                               args.model, use_focal=args.focal_loss,
                               focal_gamma=args.focal_gamma,
                               label_smoothing=args.label_smoothing)
            scheduler.step()

            if (epoch + 1) % args.eval_every == 0:
                # Early stopping on VALIDATION set (not test!)
                val_score = eval_model(model, feat_dev, adj_sp, val_pairs,
                                       device, args.model)
                vm = evaluate_predictions(val_labels, val_score)
                if args.verbose:
                    print("  Epoch {:3d}: loss={:.4f}  val_AUC={:.4f}".format(
                        epoch + 1, loss, vm["auc"]))

                if vm["auc"] > best_val_auc:
                    best_val_auc = vm["auc"]
                    patience_cnt = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_cnt += 1

                if patience_cnt >= args.patience:
                    if args.verbose:
                        print("  Early stop at epoch {}".format(epoch + 1))
                    break

        model.load_state_dict(best_state)
        y_score = eval_model(model, feat_dev, adj_sp, te_pairs, device, args.model)
        m = evaluate_predictions(te_labels, y_score)
        metrics_list.append(m)

        # Record diffusion parameters (only for models that support it)
        if args.model in ("frac", "fracadapt") and hasattr(model, "get_diffusion_params"):
            dp = model.get_diffusion_params()
            diffusion_params_list.append(dp)
            msg = "  Diffusion: alpha_drug={:.4f}, alpha_prot={:.4f}, " \
                  "t_drug={:.4f}, t_prot={:.4f}".format(
                      dp["alpha_drug"], dp["alpha_prot"],
                      dp["t_drug"], dp["t_prot"])
            if "gamma_drug" in dp:
                msg += ", gamma_drug={:.4f}, gamma_prot={:.4f}".format(
                    dp["gamma_drug"], dp["gamma_prot"])
            print(msg)
        elif args.model == "fracadapt" and hasattr(model, "get_diffusion_stats"):
            stats = model.get_diffusion_stats(feat_dev, adj_sp)
            diffusion_params_list.append(stats)
            print("  Diffusion: drug_alpha={:.3f}+/-{:.3f}, prot_alpha={:.3f}+/-{:.3f}".format(
                stats["drug_alpha_mean"], stats["drug_alpha_std"],
                stats["prot_alpha_mean"], stats["prot_alpha_std"]))

        print("  Fold {} → AUC={:.4f} AUPR={:.4f} F1={:.4f} Acc={:.4f} (thr={:.3f})".format(
            fold + 1, m["auc"], m["aupr"], m["f1"], m["acc"], m["threshold"]))

    return metrics_list, diffusion_params_list


# ---------------------------------------------------------------------------
# Debiased cold-start splitting
# ---------------------------------------------------------------------------

def _load_cluster_mapping(dataset, cold_type):
    """Load cluster info from full.csv and entity-to-index mapping."""
    full_df = pd.read_csv("../data/{}/full.csv".format(dataset))
    if cold_type == "drug":
        idx_map = pkl.load(open("../data/{}/drug2index.pkl".format(dataset), "rb"))
        ent_df = full_df.drop_duplicates("SMILES")[["SMILES", "drug_cluster"]].copy()
        ent_df["idx"] = ent_df["SMILES"].map(idx_map)
        ent_df = ent_df.dropna(subset=["idx"])
        ent_df["idx"] = ent_df["idx"].astype(int)
        return ent_df.groupby("drug_cluster")["idx"].apply(list).to_dict()
    else:
        idx_map = pkl.load(open("../data/{}/prot2index.pkl".format(dataset), "rb"))
        ent_df = full_df.drop_duplicates("Protein")[["Protein", "target_cluster"]].copy()
        ent_df["idx"] = ent_df["Protein"].map(idx_map)
        ent_df = ent_df.dropna(subset=["idx"])
        ent_df["idx"] = ent_df["idx"].astype(int)
        return ent_df.groupby("target_cluster")["idx"].apply(list).to_dict()


def debiased_cluster_split(dataset, cold_type, cold_ratio, rng):
    """Hold out entire clusters so test entities share no cluster with train."""
    clusters = _load_cluster_mapping(dataset, cold_type)
    cluster_ids = list(clusters.keys())
    rng.shuffle(cluster_ids)

    all_ent = sum(len(v) for v in clusters.values())
    target_test = max(1, int(cold_ratio * all_ent))

    test_ent = []
    for cid in cluster_ids:
        test_ent.extend(clusters[cid])
        if len(test_ent) >= target_test:
            break
    test_ent = np.array(test_ent)
    print("  Debiased (cluster): {} test entities from {} clusters "
          "(target ~{})".format(len(test_ent), len(test_ent), target_test))
    return test_ent


def debiased_tanimoto_split(dataset, cold_type, cold_ratio, threshold, rng):
    """Select test entities whose max Tanimoto to all train entities < threshold."""
    if cold_type == "drug":
        sim = pkl.load(open("../data/{}/drug_similarity_matrix.pkl".format(dataset), "rb"))
    else:
        sim = pkl.load(open("../data/{}/prot_similarity_matrix.pkl".format(dataset), "rb"))
    sim = np.array(sim, dtype=np.float32)
    n = sim.shape[0]
    np.fill_diagonal(sim, 0.0)

    target_test = max(1, int(cold_ratio * n))
    max_sim = sim.max(axis=1)
    candidates = np.where(max_sim < threshold)[0]

    if len(candidates) >= target_test:
        chosen = rng.choice(candidates, target_test, replace=False)
    else:
        sorted_by_isolation = np.argsort(max_sim)
        chosen = sorted_by_isolation[:target_test]
        actual_thr = max_sim[chosen[-1]]
        print("  WARNING: Only {} drugs below threshold {:.2f}; "
              "relaxed to {:.4f} to get {} test entities".format(
                  len(candidates), threshold, actual_thr, target_test))

    if cold_type == "drug":
        test_ent = chosen
    else:
        num_info = pkl.load(open("../data/{}/num.pkl".format(dataset), "rb"))
        test_ent = chosen + num_info["drug_num"]

    train_mask = np.ones(n, dtype=bool)
    train_mask[chosen] = False
    if train_mask.any():
        train_indices = np.where(train_mask)[0]
        max_sim_to_train = sim[chosen][:, train_indices].max(axis=1)
        print("  Debiased (tanimoto thr={:.2f}): {} test entities, "
              "max_sim_to_train: mean={:.4f}, max={:.4f}".format(
                  threshold, len(test_ent),
                  max_sim_to_train.mean(), max_sim_to_train.max()))
    return test_ent


# ---------------------------------------------------------------------------
# Cold-start evaluation
# ---------------------------------------------------------------------------

def run_cold(args, feat_t, adj_full, pos_pairs, neg_pairs_all, drug_num, prot_num):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    nfeat = feat_t.shape[1]
    pos_set = set(map(tuple, pos_pairs))

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    metrics_list = []
    diffusion_params_list = []

    for run_seed in seeds:
        print("\n--- Cold-start ({}) seed {} ---".format(args.cold_type, run_seed))
        rng = np.random.RandomState(run_seed)

        drugs = np.arange(drug_num)
        targets = np.arange(drug_num, drug_num + prot_num)

        if args.debiased == "cluster":
            test_ent = debiased_cluster_split(
                args.dataset, args.cold_type, args.cold_ratio, rng)
            if args.cold_type == "target":
                pass
            mask_pos = np.isin(pos_pairs[:, 0 if args.cold_type == "drug" else 1], test_ent)
            mask_neg = np.isin(neg_pairs_all[:, 0 if args.cold_type == "drug" else 1], test_ent)
        elif args.debiased == "tanimoto":
            test_ent = debiased_tanimoto_split(
                args.dataset, args.cold_type, args.cold_ratio,
                args.debiased_threshold, rng)
            mask_pos = np.isin(pos_pairs[:, 0 if args.cold_type == "drug" else 1], test_ent)
            mask_neg = np.isin(neg_pairs_all[:, 0 if args.cold_type == "drug" else 1], test_ent)
        elif args.cold_type == "drug":
            test_ent = rng.choice(drugs, max(1, int(args.cold_ratio * drug_num)), replace=False)
            mask_pos = np.isin(pos_pairs[:, 0], test_ent)
            mask_neg = np.isin(neg_pairs_all[:, 0], test_ent)
        else:
            test_ent = rng.choice(targets, max(1, int(args.cold_ratio * prot_num)), replace=False)
            mask_pos = np.isin(pos_pairs[:, 1], test_ent)
            mask_neg = np.isin(neg_pairs_all[:, 1], test_ent)

        train_pos = pos_pairs[~mask_pos]
        test_pos = pos_pairs[mask_pos]
        train_neg_pool = neg_pairs_all[~mask_neg]
        test_neg_pool = neg_pairs_all[mask_neg]

        print("  DEBUG: test_ent size={}, mask_pos True={}/{}, mask_neg True={}".format(
            len(test_ent), mask_pos.sum(), len(mask_pos), mask_neg.sum()))
        if len(test_pos) > 0:
            print("  DEBUG: test_pos col1 range=[{}, {}], test_ent range=[{}, {}]".format(
                test_pos[:, 1].min(), test_pos[:, 1].max(),
                test_ent.min(), test_ent.max()))

        rng2 = np.random.RandomState(run_seed)
        nr = args.neg_ratio
        if nr == 0:
            train_neg = train_neg_pool
        else:
            n_want_tr = len(train_pos) * nr
            if len(train_neg_pool) > n_want_tr:
                idx = rng2.choice(len(train_neg_pool), n_want_tr, replace=False)
                train_neg = train_neg_pool[idx]
            else:
                train_neg = train_neg_pool

        if nr == 0:
            test_neg = test_neg_pool if len(test_neg_pool) > 0 else sample_negatives(
                test_pos, drug_num, prot_num, 0, drug_num, pos_set, rng2)
        else:
            n_want_te = len(test_pos) * nr
            if len(test_neg_pool) > n_want_te:
                idx = rng2.choice(len(test_neg_pool), n_want_te, replace=False)
                test_neg = test_neg_pool[idx]
            elif len(test_neg_pool) > 0:
                test_neg = test_neg_pool
            else:
                test_neg = sample_negatives(
                    test_pos, drug_num, prot_num, 0, drug_num, pos_set, rng2)

        test_pos = np.array(test_pos).reshape(-1, 2)
        test_neg = np.array(test_neg).reshape(-1, 2)

        print("  Train: {} pos, {} neg | Test: {} pos, {} neg".format(
            len(train_pos), len(train_neg), len(test_pos), len(test_neg)))

        if len(test_pos) == 0:
            print("  WARNING: Empty test set for seed {}, skipping.".format(run_seed))
            continue

        adj_train = remove_edges(adj_full, test_pos)
        adj_norm = compute_norm_adj(adj_train, weighted=args.weighted_knn)
        adj_sp = scipy_sparse_to_torch(adj_norm, device)
        feat_dev = feat_t.to(device)

        model = build_model(args, nfeat, drug_num, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        # Val split
        val_ratio = 0.1
        n_vp = max(1, int(val_ratio * len(train_pos)))
        n_vn = max(1, int(val_ratio * len(train_neg)))
        pp = rng2.permutation(len(train_pos))
        pn = rng2.permutation(len(train_neg))
        val_pos, val_neg = train_pos[pp[:n_vp]], train_neg[pn[:n_vn]]
        act_tr_pos, act_tr_neg = train_pos[pp[n_vp:]], train_neg[pn[n_vn:]]
        val_pairs = np.concatenate([val_pos, val_neg])
        val_labels = np.concatenate([np.ones(len(val_pos)), np.zeros(len(val_neg))])

        best_auc = 0.0
        patience_cnt = 0
        best_state = None

        for epoch in range(args.epochs):
            if args.hard_neg and epoch >= args.hard_neg_start:
                n_neg_epoch = len(act_tr_pos) * max(1, args.neg_ratio) if args.neg_ratio != 0 else len(act_tr_neg)
                epoch_neg = hard_negative_mining(
                    model, feat_dev, adj_sp, act_tr_pos, drug_num, prot_num,
                    pos_set, device, n_neg_epoch, args.hard_neg_ratio)
            elif args.neg_ratio == 0:
                epoch_neg = act_tr_neg
            elif args.neg_ratio > 1:
                ern = np.random.RandomState(run_seed + epoch)
                neg_parts = []
                for _ in range(args.neg_ratio):
                    neg_parts.append(sample_negatives(
                        act_tr_pos, drug_num, prot_num, 0, drug_num, pos_set, ern))
                epoch_neg = np.concatenate(neg_parts)
            else:
                ern = np.random.RandomState(run_seed + epoch)
                epoch_neg = sample_negatives(
                    act_tr_pos, drug_num, prot_num, 0, drug_num, pos_set, ern)
            loss = train_epoch(model, optimizer, feat_dev, adj_sp,
                               act_tr_pos, epoch_neg, device, args.batch_size,
                               args.model, use_focal=args.focal_loss,
                               focal_gamma=args.focal_gamma,
                               label_smoothing=args.label_smoothing)
            scheduler.step()

            if (epoch + 1) % args.eval_every == 0:
                vs = eval_model(model, feat_dev, adj_sp, val_pairs, device, args.model)
                vm = evaluate_predictions(val_labels, vs)
                if args.verbose:
                    print("  Epoch {:3d}: loss={:.4f}  val_AUC={:.4f}".format(
                        epoch + 1, loss, vm["auc"]))
                if vm["auc"] > best_auc:
                    best_auc = vm["auc"]
                    patience_cnt = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_cnt += 1
                if patience_cnt >= args.patience:
                    if args.verbose:
                        print("  Early stop at epoch {}".format(epoch + 1))
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        te_all = np.concatenate([test_pos, test_neg])
        te_lbl = np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))])
        if args.test_time_aug:
            y_score = eval_cold_with_tta(
                model, feat_dev, adj_train, te_all, device, args.model,
                weighted=args.weighted_knn, conf_thr=args.tta_threshold)
        else:
            y_score = eval_model(model, feat_dev, adj_sp, te_all, device, args.model)
        m = evaluate_predictions(te_lbl, y_score)
        metrics_list.append(m)

        if args.model in ("frac", "fracadapt") and hasattr(model, "get_diffusion_params"):
            dp = model.get_diffusion_params()
            diffusion_params_list.append(dp)
            msg = "  Diffusion: alpha_drug={:.4f}, alpha_prot={:.4f}".format(
                dp["alpha_drug"], dp["alpha_prot"])
            if "gamma_drug" in dp:
                msg += ", gamma_drug={:.4f}, gamma_prot={:.4f}".format(
                    dp["gamma_drug"], dp["gamma_prot"])
            print(msg)
        elif args.model == "fracadapt" and hasattr(model, "get_diffusion_stats"):
            stats = model.get_diffusion_stats(feat_dev, adj_sp)
            diffusion_params_list.append(stats)
            print("  Diffusion: drug_alpha={:.3f}+/-{:.3f}, prot_alpha={:.3f}+/-{:.3f}".format(
                stats["drug_alpha_mean"], stats["drug_alpha_std"],
                stats["prot_alpha_mean"], stats["prot_alpha_std"]))

        print("  Seed {} → AUC={:.4f} AUPR={:.4f} F1={:.4f} Acc={:.4f} (thr={:.3f})".format(
            run_seed, m["auc"], m["aupr"], m["f1"], m["acc"], m["threshold"]))

    return metrics_list, diffusion_params_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("FracDTI / FracAdaptDTI Training")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="biosnap")
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--seeds", default="520,521,522,523,524",
                        help="Seeds for cold-start multi-run")

    all_models = ["frac", "fracadapt"]
    parser.add_argument("--model", default="frac", choices=all_models)
    parser.add_argument("--mode", default="warm", choices=["warm", "cold"])
    parser.add_argument("--cold_type", default="drug", choices=["drug", "target"])
    parser.add_argument("--cold_ratio", type=float, default=0.2)
    parser.add_argument("--debiased", default="none",
                        choices=["none", "cluster", "tanimoto"],
                        help="Debiased cold-start split method")
    parser.add_argument("--debiased_threshold", type=float, default=0.5,
                        help="Tanimoto threshold for debiased=tanimoto (test drugs "
                             "have max similarity < this to any train drug)")
    parser.add_argument("--no_edge_removal", action="store_true",
                        help="Warm-start WITHOUT removing test edges (leak test)")

    parser.add_argument("--alpha_init", type=float, default=0.0,
                        help="Initial value for log_alpha_drug/prot (before softplus). "
                             "Default 0.0 → alpha≈0.69. Use -0.94 for ~0.25, "
                             "0.54 for ~1.0, 1.31 for ~1.5")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--pred_hidden", type=int, default=64)
    parser.add_argument("--K", type=int, default=15, help="Polynomial order")
    parser.add_argument("--appnp_alpha", type=float, default=0.1,
                        help="Teleport probability for APPNP baseline")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hyper_hidden", type=int, default=32,
                        help="Hidden dim for FracAdapt hypernetwork")

    parser.add_argument("--graph_mode", default="original",
                        choices=["original", "knn", "full"],
                        help="Graph construction: original, knn (Top-K), full (threshold)")
    parser.add_argument("--knn_k", type=int, default=10,
                        help="K for KNN graph enhancement")
    parser.add_argument("--sim_threshold", type=float, default=0.5,
                        help="Cosine similarity threshold for full graph")
    parser.add_argument("--weighted_knn", action="store_true",
                        help="Use cosine similarity as edge weight (not binary)")

    parser.add_argument("--focal_loss", action="store_true",
                        help="Use focal loss instead of BCE")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing (0=off, 0.05=recommended)")

    parser.add_argument("--hard_neg", action="store_true",
                        help="Enable hard negative mining")
    parser.add_argument("--hard_neg_ratio", type=float, default=0.5,
                        help="Fraction of negatives that are hard-mined")
    parser.add_argument("--hard_neg_start", type=int, default=30,
                        help="Epoch to start hard negative mining")

    parser.add_argument("--test_time_aug", action="store_true",
                        help="Test-time augmentation for cold-start")
    parser.add_argument("--tta_threshold", type=float, default=0.8,
                        help="Confidence threshold for TTA edge insertion")

    parser.add_argument("--neg_ratio", type=int, default=1,
                        help="Negative-to-positive ratio (1=balanced, 10=1:10, 0=all negatives)")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--k_fold", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--verbose", default="True")
    parser.add_argument("--result_tag", default="")

    args = parser.parse_args()
    args.verbose = str(args.verbose).lower() == "true"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("log", exist_ok=True)
    rtag = "_{}".format(args.result_tag) if args.result_tag else ""
    debiased_tag = ""
    if args.debiased != "none":
        debiased_tag = "_debiased_{}".format(args.debiased)
        if args.debiased == "tanimoto":
            debiased_tag += "_thr{}".format(args.debiased_threshold)
    log_dir = os.environ.get("FRAC_LOG_DIR", "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "{}_{}_{}{}{}_{}_{}.txt".format(
        args.model, args.mode, args.dataset,
        "_cold_{}".format(args.cold_type) if args.mode == "cold" else "",
        debiased_tag, rtag, ts))
    sys.stdout = Logger(log_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    if args.dataset == "bindingdb":
        drug_num, prot_num = 14643, 2623
    elif args.dataset in ("DrugBank1.4", "DrugBank"):
        drug_num, prot_num = 6645, 4254
    else:
        num = pkl.load(open("../data/{}/num.pkl".format(args.dataset), "rb"))
        drug_num = num["drug_num"]
        prot_num = num["prot_num"]

    print("=" * 60)
    print("  Model      : {} ({})".format(
        args.model.upper(),
        "Global fractional order" if args.model == "frac" else "Node-adaptive fractional order"))
    edge_tag = " (NO edge removal)" if args.no_edge_removal else ""
    debiased_info = ""
    if args.debiased != "none":
        debiased_info = " [DEBIASED: {}".format(args.debiased)
        if args.debiased == "tanimoto":
            debiased_info += ", thr={}".format(args.debiased_threshold)
        debiased_info += "]"
    print("  Mode       : {} {}{}{}".format(
        args.mode, "(cold={})".format(args.cold_type) if args.mode == "cold" else "",
        edge_tag, debiased_info))
    print("  Dataset    : {} (drugs={}, proteins={})".format(args.dataset, drug_num, prot_num))
    print("  K={}, hidden={}, embed={}, epochs={}, lr={}".format(
        args.K, args.hidden, args.embed_dim, args.epochs, args.lr))
    print("=" * 60)

    feat_df = pd.read_csv("../data/{}/AllNodeAttribute_DrPr.csv".format(args.dataset), header=None)
    feat_df = feat_df.iloc[:, 1:]
    n_total = len(feat_df)

    pos_pairs = pd.read_csv("../data/{}/DrPrNum_DrPr.csv".format(args.dataset), header=None).values
    neg_pairs_all = pd.read_csv("../data/{}/AllNegative_DrPr.csv".format(args.dataset), header=None).values

    feat_t = torch.FloatTensor(np.array(sp.csr_matrix(feat_df, dtype=np.float32).todense()))
    feat_t = F.normalize(feat_t, p=2)

    if args.graph_mode == "original":
        dpe = pd.read_csv("../data/{}/drug_prot_edge.csv".format(args.dataset), header=None)
        labels_dummy = pd.DataFrame(np.zeros(n_total))
        labels_dummy[:drug_num] = 0
        labels_dummy[drug_num:] = 1
        labels_dummy = labels_dummy[0]
        adj_full, _, _, _, _, _ = load_data(dpe, feat_df, labels_dummy)
        print("  Graph: original (drug_prot_edge.csv), nnz={}".format(adj_full.nnz))
    else:
        adj_full = build_bipartite_adj(pos_pairs, n_total)
        print("  Graph base: pure drug-protein interactions only, nnz={}".format(adj_full.nnz))
        if args.graph_mode == "knn":
            adj_full = enhance_adj_knn(adj_full, feat_t, drug_num, prot_num,
                                       k=args.knn_k, weighted=args.weighted_knn)
        elif args.graph_mode == "full":
            adj_full = enhance_adj_full(adj_full, feat_t, drug_num, prot_num, threshold=args.sim_threshold)

    extras = []
    if args.weighted_knn:
        extras.append("weighted_knn")
    if args.focal_loss:
        extras.append("focal(γ={},ls={})".format(args.focal_gamma, args.label_smoothing))
    if args.hard_neg:
        extras.append("hard_neg(r={},s={})".format(args.hard_neg_ratio, args.hard_neg_start))
    if args.test_time_aug:
        extras.append("TTA(thr={})".format(args.tta_threshold))
    if extras:
        print("  Improvements: {}".format(", ".join(extras)))

    print("Positive: {}, Negative: {}".format(len(pos_pairs), len(neg_pairs_all)))
    if args.neg_ratio != 1:
        ratio_tag = "ALL" if args.neg_ratio == 0 else "1:{}".format(args.neg_ratio)
        print("  Neg Ratio  : {} (imbalanced)".format(ratio_tag))

    if args.mode == "warm":
        metrics, diff_params = run_warm(
            args, feat_t, adj_full, pos_pairs, neg_pairs_all, drug_num, prot_num)
    else:
        metrics, diff_params = run_cold(
            args, feat_t, adj_full, pos_pairs, neg_pairs_all, drug_num, prot_num)

    # Summary
    keys = ["auc", "aupr", "f1", "sens", "spec", "prec", "acc"]
    if len(metrics) == 0:
        print("\nERROR: No valid results collected. All seeds may have empty test sets.")
        print("Check data format: pos_pairs col ranges vs drug_num/prot_num.")
        return
    arr = np.array([[m[k] for k in keys] for m in metrics])
    mean, std = arr.mean(0), arr.std(0)

    print("\n" + "=" * 60)
    print("  SUMMARY: {} {} ({})".format(
        args.model.upper(), args.mode,
        args.cold_type if args.mode == "cold" else "10-fold"))
    print("=" * 60)
    for i, k in enumerate(keys):
        print("  {:6s}: {:.4f} +/- {:.4f}".format(k.upper(), mean[i], std[i]))

    # Diffusion parameter analysis
    if diff_params:
        print("\n" + "-" * 60)
        print("  LEARNED DIFFUSION PARAMETERS (key scientific finding)")
        print("-" * 60)
        if args.model in ("frac", "fracadapt") and "alpha_drug" in diff_params[0]:
            ad = np.mean([d["alpha_drug"] for d in diff_params])
            ap = np.mean([d["alpha_prot"] for d in diff_params])
            td = np.mean([d["t_drug"] for d in diff_params])
            tp = np.mean([d["t_prot"] for d in diff_params])
            print("  alpha_drug  = {:.4f}  (avg over folds/seeds)".format(ad))
            print("  alpha_prot  = {:.4f}".format(ap))
            print("  t_drug      = {:.4f}".format(td))
            print("  t_prot      = {:.4f}".format(tp))
            print("  alpha_drug {} alpha_prot  (ratio: {:.2f}x)".format(
                ">" if ad > ap else "<", ad / max(ap, 1e-8)))
            if abs(ad - ap) > 0.1:
                print("  >>> FINDING: Drug and protein sides require DIFFERENT diffusion orders!")
            else:
                print("  >>> Drug and protein diffusion orders are similar.")
            if "gamma_drug" in diff_params[0]:
                gd = np.mean([d["gamma_drug"] for d in diff_params])
                gp = np.mean([d["gamma_prot"] for d in diff_params])
                print("  gamma_drug  = {:.4f}  (graph blending ratio)".format(gd))
                print("  gamma_prot  = {:.4f}".format(gp))
                print("  >>> Drug uses {:.0f}% graph info, Protein uses {:.0f}% graph info".format(
                    gd * 100, gp * 100))
        elif args.model == "fracadapt" and "drug_alpha_mean" in diff_params[0]:
            dam = np.mean([d["drug_alpha_mean"] for d in diff_params])
            das = np.mean([d["drug_alpha_std"] for d in diff_params])
            pam = np.mean([d["prot_alpha_mean"] for d in diff_params])
            pas = np.mean([d["prot_alpha_std"] for d in diff_params])
            print("  Drug  alpha = {:.4f} +/- {:.4f}  (mean over nodes)".format(dam, das))
            print("  Prot  alpha = {:.4f} +/- {:.4f}".format(pam, pas))
            if das > 0.1 or pas > 0.1:
                print("  >>> FINDING: Individual nodes learn DIVERSE diffusion orders!")
            if abs(dam - pam) > 0.1:
                print("  >>> FINDING: Drug/protein sides have DIFFERENT mean diffusion orders!")
    else:
        print("\n  (No diffusion parameters for model '{}')".format(args.model))

    # Save results
    csv_dir = os.path.join(log_dir, "frac_results")
    os.makedirs(csv_dir, exist_ok=True)
    mode_tag = args.mode if args.mode == "warm" else "cold_{}".format(args.cold_type)
    csv_path = os.path.join(csv_dir, "{}_{}_{}{}{}{}.csv".format(
        args.model, args.dataset, mode_tag, debiased_tag, rtag, ts))
    with open(csv_path, "w") as f:
        f.write(",".join(["metric", "mean", "std"]) + "\n")
        for i, k in enumerate(keys):
            f.write("{},{:.6f},{:.6f}\n".format(k, mean[i], std[i]))
    print("\nMetrics CSV: {}".format(csv_path))

    if diff_params:
        json_path = csv_path.replace(".csv", "_diffusion.json")
        with open(json_path, "w") as f:
            json.dump(diff_params, f, indent=2)
        print("Diffusion params JSON: {}".format(json_path))


if __name__ == "__main__":
    main()
