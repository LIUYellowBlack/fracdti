"""Generate meaningful node features using pretrained embeddings.

Replaces the untrained GCN features in AllNodeAttribute_DrPr.csv with:
  - Drugs:    Morgan fingerprint (1024-bit) + RDKit 2D descriptors
  - Proteins: ESM-2 embeddings (--mode esm) OR AAC + DPC features (--mode fast)

Usage:
  # Fast mode (no GPU, no deep learning dependencies):
  python data_proc_pretrained.py --dataset biosnap --mode fast

  # ESM-2 mode (needs fair-esm + GPU):
  python data_proc_pretrained.py --dataset biosnap --mode esm --device cuda:0

  # All datasets:
  for d in biosnap human DrugBank bindingdb; do
      python data_proc_pretrained.py --dataset $d --mode fast
  done

Output:
  Saves to ../data/{dataset}/AllNodeAttribute_DrPr_pretrained.csv
  Also backs up original to AllNodeAttribute_DrPr_gcn_backup.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from collections import Counter

# ---------------------------------------------------------------------------
# Drug featurization: Morgan FP + RDKit descriptors
# ---------------------------------------------------------------------------

def drug_morgan_fp(smiles, radius=2, n_bits=1024):
    """Morgan fingerprint as bit vector."""
    from rdkit import Chem
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.float32)
    for idx in fp.GetOnBits():
        arr[idx] = 1.0
    return arr


def drug_rdkit_descriptors(smiles):
    """Compute a curated set of 2D RDKit molecular descriptors."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(10, dtype=np.float32)

    desc = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        Lipinski.NumAromaticRings(mol),
    ], dtype=np.float32)
    return desc


DRUG_FP_DIM = 1024
DRUG_DESC_DIM = 10
DRUG_DIM = DRUG_FP_DIM + DRUG_DESC_DIM  # 1034


def featurize_drug(smiles):
    fp = drug_morgan_fp(smiles, n_bits=DRUG_FP_DIM)
    desc = drug_rdkit_descriptors(smiles)
    return np.concatenate([fp, desc])


# ---------------------------------------------------------------------------
# Protein featurization: AAC + DPC (fast mode, no dependencies)
# ---------------------------------------------------------------------------

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

DIPEPTIDES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]  # 400
DP_TO_IDX = {dp: i for i, dp in enumerate(DIPEPTIDES)}

# 7 physicochemical groups (Conjoint Triad encoding)
PHYSCHEM_GROUPS = {
    'A': 0, 'G': 0, 'V': 0,
    'I': 1, 'L': 1, 'F': 1, 'P': 1,
    'Y': 2, 'M': 2, 'T': 2, 'S': 2,
    'H': 3, 'N': 3, 'Q': 3, 'W': 3,
    'R': 4, 'K': 4,
    'D': 5, 'E': 5,
    'C': 6,
}


def protein_aac(sequence):
    """Amino Acid Composition: 20-dim frequency vector."""
    counts = Counter(sequence)
    total = max(len(sequence), 1)
    return np.array([counts.get(aa, 0) / total for aa in AMINO_ACIDS], dtype=np.float32)


def protein_dpc(sequence):
    """Dipeptide Composition: 400-dim frequency vector."""
    counts = Counter()
    for i in range(len(sequence) - 1):
        dp = sequence[i:i+2]
        if dp in DP_TO_IDX:
            counts[dp] += 1
    total = max(len(sequence) - 1, 1)
    return np.array([counts.get(dp, 0) / total for dp in DIPEPTIDES], dtype=np.float32)


def protein_ctd(sequence):
    """Conjoint Triad Descriptor: 343-dim (7^3 triad frequencies)."""
    groups = []
    for aa in sequence:
        g = PHYSCHEM_GROUPS.get(aa, -1)
        if g >= 0:
            groups.append(g)
    triads = np.zeros(343, dtype=np.float32)
    for i in range(len(groups) - 2):
        idx = groups[i] * 49 + groups[i+1] * 7 + groups[i+2]
        triads[idx] += 1
    total = max(len(groups) - 2, 1)
    return triads / total


PROT_FAST_DIM = 20 + 400 + 343  # 763


def featurize_protein_fast(sequence):
    """AAC (20) + DPC (400) + CTD (343) = 763-dim."""
    seq_clean = ''.join(c for c in sequence.upper() if c in AA_TO_IDX or c in ('X', 'U'))
    seq_clean = seq_clean.replace('X', 'A').replace('U', 'C')
    aac = protein_aac(seq_clean)
    dpc = protein_dpc(seq_clean)
    ctd = protein_ctd(seq_clean)
    return np.concatenate([aac, dpc, ctd])


# ---------------------------------------------------------------------------
# Protein featurization: ESM-2 (best mode, needs fair-esm + GPU)
# ---------------------------------------------------------------------------

ESM_DIM = 1280  # esm2_t33_650M_UR50D output dimension


def featurize_proteins_esm(sequences, device="cuda:0", batch_size=8, max_len=1022):
    """Batch ESM-2 inference. Returns (N, 1280) array."""
    import torch
    import esm

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)

    embeddings = []
    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start:start + batch_size]
        data = [(str(i), seq[:max_len]) for i, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[33])
        # Average over sequence length (exclude BOS/EOS)
        token_repr = results["representations"][33]
        for j, seq in enumerate(batch_seqs):
            seq_len = min(len(seq), max_len)
            emb = token_repr[j, 1:seq_len + 1].mean(dim=0).cpu().numpy()
            embeddings.append(emb)

        if (start // batch_size) % 10 == 0:
            print("  ESM-2 progress: {}/{}".format(start + len(batch_seqs), len(sequences)))

    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main: process dataset
# ---------------------------------------------------------------------------

def align_dimensions(drug_feats, prot_feats):
    """Zero-pad both to the same dimension (max of the two)."""
    d_dim = drug_feats.shape[1]
    p_dim = prot_feats.shape[1]
    target_dim = max(d_dim, p_dim)

    if d_dim < target_dim:
        pad = np.zeros((drug_feats.shape[0], target_dim - d_dim), dtype=np.float32)
        drug_feats = np.concatenate([drug_feats, pad], axis=1)
    if p_dim < target_dim:
        pad = np.zeros((prot_feats.shape[0], target_dim - p_dim), dtype=np.float32)
        prot_feats = np.concatenate([prot_feats, pad], axis=1)

    return drug_feats, prot_feats, target_dim


def normalize_features(feats):
    """Per-column standardization (zero mean, unit variance)."""
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std == 0] = 1.0
    return (feats - mean) / std


def main():
    parser = argparse.ArgumentParser(description="Generate pretrained node features")
    parser.add_argument("--dataset", required=True, help="Dataset name (biosnap, human, DrugBank, bindingdb)")
    parser.add_argument("--mode", default="fast", choices=["fast", "esm"],
                        help="fast: Morgan FP + AAC/DPC (no GPU). esm: Morgan FP + ESM-2 (GPU).")
    parser.add_argument("--device", default="cuda:0", help="Device for ESM-2 inference")
    parser.add_argument("--esm_batch_size", type=int, default=4, help="ESM-2 batch size")
    parser.add_argument("--no_backup", action="store_true", help="Skip backing up original CSV")
    args = parser.parse_args()

    data_dir = os.path.join("..", "data", args.dataset)
    full_path = os.path.join(data_dir, "full.csv")
    out_path = os.path.join(data_dir, "AllNodeAttribute_DrPr_pretrained.csv")
    orig_path = os.path.join(data_dir, "AllNodeAttribute_DrPr.csv")
    backup_path = os.path.join(data_dir, "AllNodeAttribute_DrPr_gcn_backup.csv")

    print("Dataset: {}, Mode: {}".format(args.dataset, args.mode))
    df = pd.read_csv(full_path)

    # Build ordered drug/protein lists (same order as data_proc.py)
    smiles_list = []
    protein_list = []
    for _, row in df.iterrows():
        if row['SMILES'] not in smiles_list:
            smiles_list.append(row['SMILES'])
        if row['Protein'] not in protein_list:
            protein_list.append(row['Protein'])

    drug_num = len(smiles_list)
    prot_num = len(protein_list)
    print("Drugs: {}, Proteins: {}".format(drug_num, prot_num))

    # Verify against num.pkl
    num_path = os.path.join(data_dir, "num.pkl")
    if os.path.exists(num_path):
        num = pkl.load(open(num_path, "rb"))
        assert num["drug_num"] == drug_num, "Drug count mismatch: {} vs {}".format(num["drug_num"], drug_num)
        assert num["prot_num"] == prot_num, "Protein count mismatch: {} vs {}".format(num["prot_num"], prot_num)
        print("num.pkl verified OK")

    # --- Drug features ---
    print("\nComputing drug features (Morgan FP {} + RDKit desc {})...".format(DRUG_FP_DIM, DRUG_DESC_DIM))
    drug_feats = []
    for i, smi in enumerate(smiles_list):
        drug_feats.append(featurize_drug(smi))
        if (i + 1) % 500 == 0:
            print("  Drug {}/{}".format(i + 1, drug_num))
    drug_feats = np.array(drug_feats, dtype=np.float32)
    print("Drug features: {}".format(drug_feats.shape))

    # --- Protein features ---
    if args.mode == "esm":
        print("\nComputing protein features (ESM-2, {}d)...".format(ESM_DIM))
        prot_feats = featurize_proteins_esm(
            protein_list, device=args.device, batch_size=args.esm_batch_size
        )
    else:
        print("\nComputing protein features (AAC+DPC+CTD, {}d)...".format(PROT_FAST_DIM))
        prot_feats = []
        for i, seq in enumerate(protein_list):
            prot_feats.append(featurize_protein_fast(seq))
            if (i + 1) % 500 == 0:
                print("  Protein {}/{}".format(i + 1, prot_num))
        prot_feats = np.array(prot_feats, dtype=np.float32)
    print("Protein features: {}".format(prot_feats.shape))

    # --- Normalize each modality separately before alignment ---
    drug_feats = normalize_features(drug_feats)
    prot_feats = normalize_features(prot_feats)

    # --- Align dimensions (zero-pad to max) ---
    drug_feats, prot_feats, final_dim = align_dimensions(drug_feats, prot_feats)
    print("\nAligned to {}d (drug: {}, protein: {})".format(final_dim, drug_feats.shape, prot_feats.shape))

    # --- Concatenate: drugs first, then proteins (same order as data_proc.py) ---
    all_feats = np.concatenate([drug_feats, prot_feats], axis=0)
    print("Final feature matrix: {}".format(all_feats.shape))

    # --- Save ---
    if not args.no_backup and os.path.exists(orig_path) and not os.path.exists(backup_path):
        import shutil
        shutil.copy2(orig_path, backup_path)
        print("Backed up original to: {}".format(backup_path))

    feat_df = pd.DataFrame(all_feats)
    feat_df.to_csv(out_path, index=True, header=False)
    print("Saved pretrained features to: {}".format(out_path))

    # Also overwrite the original so main.py/cold_bridge.py use them directly
    feat_df.to_csv(orig_path, index=True, header=False)
    print("Overwrote {} (original backed up)".format(orig_path))
    print("Done!")


if __name__ == "__main__":
    main()
