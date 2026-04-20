#!/usr/bin/env python3

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import torch
from scipy import sparse


BASE_ROOT = "/home/kirilb/orcd/pool/PRH_data"
TOPK_RE = re.compile(r"^topk_(?P<d>\d+)_(?P<model_key>.+)_k_(?P<k>\d+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute mean rowwise residuals ||X_i - ((G_i)A + b)||_2 for matched SAE folders."
    )
    parser.add_argument("--name", required=True, help="Dataset name suffix, e.g. coco, words, visual_genome")
    parser.add_argument("--chunk_size", type=int, default=8192, help="Rows per chunk")
    parser.add_argument("--device", type=str, default="cuda", help="torch device, e.g. cuda or cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--verbose", type=int, default=1, help="1 to print progress, 0 to stay quiet")
    parser.add_argument(
        "--use_normalized",
        type=int,
        default=1,
        help="1 => use *_normalized.npy, 0 => use plain *.npy"
    )
    return parser.parse_args()


def log(msg, verbose=1):
    if verbose:
        print(msg, flush=True)


def get_embedding_filenames(use_normalized):
    if use_normalized:
        return "text_embeddings_normalized.npy", "img_embeddings_normalized.npy"
    return "text_embeddings.npy", "img_embeddings.npy"


def find_embedding_entries(embedded_root, use_normalized=1, verbose=1):
    """
    Returns:
        entries: dict model_key -> path
        base_to_keys: dict base_folder -> list of model_keys
    """
    entries = {}
    base_to_keys = defaultdict(list)

    if not os.path.isdir(embedded_root):
        raise FileNotFoundError(f"Embedded root does not exist: {embedded_root}")

    text_file, img_file = get_embedding_filenames(use_normalized)

    for folder in sorted(os.listdir(embedded_root)):
        folder_path = os.path.join(embedded_root, folder)
        if not os.path.isdir(folder_path):
            continue

        text_path = os.path.join(folder_path, text_file)
        img_path = os.path.join(folder_path, img_file)

        if os.path.isfile(text_path):
            key = f"{folder}_text"
            entries[key] = text_path
            base_to_keys[folder].append(key)

        if os.path.isfile(img_path):
            key = f"{folder}_img"
            entries[key] = img_path
            base_to_keys[folder].append(key)

    log(f"Found {len(entries)} embedding entries in {embedded_root}", verbose)
    return entries, dict(base_to_keys)


def scan_sae_folders(sae_root, verbose=1):
    """
    Returns list of dicts with keys:
      folder, folder_path, d, k, model_key, X_features_path, b_dec_path, decoder_weight_path
    """
    if not os.path.isdir(sae_root):
        raise FileNotFoundError(f"SAE root does not exist: {sae_root}")

    out = []
    for folder in sorted(os.listdir(sae_root)):
        folder_path = os.path.join(sae_root, folder)
        if not os.path.isdir(folder_path):
            continue

        m = TOPK_RE.match(folder)
        if m is None:
            continue

        d = int(m.group("d"))
        k = int(m.group("k"))
        model_key = m.group("model_key")

        xfeat = os.path.join(folder_path, "X_features.npz")
        bdec = os.path.join(folder_path, "b_dec.npy")
        decw = os.path.join(folder_path, "decoder_weight.npy")

        if not (os.path.isfile(xfeat) and os.path.isfile(bdec) and os.path.isfile(decw)):
            continue

        out.append(
            {
                "folder": folder,
                "folder_path": folder_path,
                "d": d,
                "k": k,
                "model_key": model_key,
                "X_features_path": xfeat,
                "b_dec_path": bdec,
                "decoder_weight_path": decw,
            }
        )

    log(f"Found {len(out)} SAE folders in {sae_root}", verbose)
    return out


def resolve_embedding_key(model_key, embedding_entries, base_to_keys):
    if model_key in embedding_entries:
        return model_key

    if model_key.endswith("_text") or model_key.endswith("_img"):
        return None

    candidates = base_to_keys.get(model_key, [])
    if len(candidates) == 1:
        return candidates[0]

    return None


def load_decoder_gpu(b_path, a_path, x_dim, device, dtype):
    """
    Loads decoder and returns:
      A_t: [m, x_dim]
      b_t: [x_dim]
    """
    b = np.load(b_path)
    A = np.load(a_path)

    b = np.asarray(b, dtype=np.float32).reshape(-1)
    A = np.asarray(A, dtype=np.float32)

    if A.ndim != 2:
        raise ValueError(f"decoder_weight must be 2D, got shape {A.shape} at {a_path}")

    # We want A.shape == (m, x_dim), and final bias in dense space [x_dim].
    # In your corrected version, the intended reconstruction is Y = G @ A + b,
    # where b is already in dense coordinates.
    if b.shape[0] != x_dim:
        raise ValueError(
            f"b_dec has shape {b.shape}, but embedding dim is x_dim={x_dim}. "
            f"This script assumes b_dec is already a dense-space bias of shape [x_dim]."
        )

    if A.shape[1] == x_dim:
        pass
    elif A.shape[0] == x_dim:
        A = A.T
    else:
        raise ValueError(
            f"Could not reconcile decoder shape A.shape={A.shape} with x_dim={x_dim}"
        )

    A_t = torch.from_numpy(A).to(device=device, dtype=dtype)
    b_t = torch.from_numpy(b).to(device=device, dtype=dtype)

    return A_t, b_t


def csr_chunk_to_padded_arrays(G_chunk):
    """
    Convert CSR chunk into padded indices/values arrays.

    Returns:
      idx:   np.int64 [B, kmax]
      val:   np.float32 [B, kmax]
      mask:  np.bool_ [B, kmax]
    """
    indptr = G_chunk.indptr
    indices = G_chunk.indices
    data = G_chunk.data
    B = G_chunk.shape[0]

    lengths = np.diff(indptr)
    kmax = int(lengths.max()) if B > 0 else 0

    idx = np.zeros((B, kmax), dtype=np.int64)
    val = np.zeros((B, kmax), dtype=np.float32)
    mask = np.zeros((B, kmax), dtype=bool)

    for i in range(B):
        s = indptr[i]
        e = indptr[i + 1]
        ell = e - s
        if ell > 0:
            idx[i, :ell] = indices[s:e]
            val[i, :ell] = data[s:e]
            mask[i, :ell] = True

    return idx, val, mask


@torch.no_grad()
def compute_mean_row_l2_error_gpu(
    X_path,
    G_path,
    b_path,
    A_path,
    chunk_size=8192,
    device="cuda",
    dtype=torch.float32,
):
    X = np.load(X_path, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"Embedding array must be 2D, got shape {X.shape} at {X_path}")

    n_rows, x_dim = X.shape

    G = sparse.load_npz(G_path).tocsr()
    if G.shape[0] != n_rows:
        raise ValueError(
            f"Row mismatch: X has {n_rows} rows but G has {G.shape[0]} rows "
            f"for X={X_path}, G={G_path}"
        )

    A_t, b_t = load_decoder_gpu(
        b_path=b_path,
        a_path=A_path,
        x_dim=x_dim,
        device=device,
        dtype=dtype,
    )

    total = 0.0
    count = 0

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)

        X_chunk = np.asarray(X[start:end], dtype=np.float32)
        G_chunk = G[start:end]

        idx_np, val_np, mask_np = csr_chunk_to_padded_arrays(G_chunk)

        X_t = torch.from_numpy(X_chunk).to(device=device, dtype=dtype, non_blocking=True)
        idx_t = torch.from_numpy(idx_np).to(device=device, dtype=torch.long, non_blocking=True)
        val_t = torch.from_numpy(val_np).to(device=device, dtype=dtype, non_blocking=True)
        mask_t = torch.from_numpy(mask_np).to(device=device, non_blocking=True)

        # gathered: [B, kmax, x_dim]
        gathered = A_t[idx_t]
        gathered = gathered * val_t.unsqueeze(-1)
        gathered = gathered * mask_t.unsqueeze(-1)

        Y_t = gathered.sum(dim=1) + b_t.unsqueeze(0)

        row_norms = torch.linalg.norm(X_t - Y_t, dim=1)
        total += row_norms.sum().item()
        count += row_norms.shape[0]

        del X_t, idx_t, val_t, mask_t, gathered, Y_t, row_norms

    return total / max(count, 1)


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64

    embedded_root = os.path.join(BASE_ROOT, f"embedded_{args.name}")
    sae_root = os.path.join(BASE_ROOT, f"topk_sae_{args.name}")

    log(f"embedded_root = {embedded_root}", args.verbose)
    log(f"sae_root      = {sae_root}", args.verbose)
    log(f"device        = {args.device}", args.verbose)
    log(f"chunk_size    = {args.chunk_size}", args.verbose)

    embedding_entries, base_to_keys = find_embedding_entries(
        embedded_root,
        use_normalized=args.use_normalized,
        verbose=args.verbose,
    )
    sae_infos = scan_sae_folders(sae_root, verbose=args.verbose)

    if len(embedding_entries) == 0:
        raise RuntimeError(f"No embedding entries found in {embedded_root}")
    if len(sae_infos) == 0:
        raise RuntimeError(f"No valid SAE folders found in {sae_root}")

    row_keys = sorted(embedding_entries.keys())
    combos = sorted({(info["d"], info["k"]) for info in sae_infos})

    row_to_i = {rk: i for i, rk in enumerate(row_keys)}
    combo_to_j = {combo: j for j, combo in enumerate(combos)}

    residuals = np.full((len(row_keys), len(combos)), np.nan, dtype=np.float32)

    unmatched = []
    duplicates = set()
    seen_pairs = set()

    for info in sae_infos:
        resolved_key = resolve_embedding_key(info["model_key"], embedding_entries, base_to_keys)
        if resolved_key is None:
            unmatched.append(info["folder"])
            continue

        combo = (info["d"], info["k"])
        pair = (resolved_key, combo)

        if pair in seen_pairs:
            duplicates.add(pair)
            log(f"Warning: duplicate SAE entry for row={resolved_key}, combo={combo}; overwriting.", args.verbose)
        seen_pairs.add(pair)

        i = row_to_i[resolved_key]
        j = combo_to_j[combo]

        log(
            f"Computing row={resolved_key}, d={info['d']}, k={info['k']} from {info['folder']}",
            args.verbose,
        )

        val = compute_mean_row_l2_error_gpu(
            X_path=embedding_entries[resolved_key],
            G_path=info["X_features_path"],
            b_path=info["b_dec_path"],
            A_path=info["decoder_weight_path"],
            chunk_size=args.chunk_size,
            device=args.device,
            dtype=torch_dtype,
        )

        residuals[i, j] = val
        log(f"  mean residual = {val:.8f}", args.verbose)

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    out_path = os.path.join(sae_root, "residuals.npy")
    np.save(out_path, residuals)
    log(f"Saved residuals to {out_path} with shape {residuals.shape}", args.verbose)

    meta_path = os.path.join(sae_root, "residuals_metadata.npz")
    np.savez(
        meta_path,
        row_keys=np.array(row_keys, dtype=object),
        combos=np.array(combos, dtype=np.int64),
        unmatched_folders=np.array(unmatched, dtype=object),
    )
    log(f"Saved metadata to {meta_path}", args.verbose)

    if residuals.shape != (30, 6):
        log(
            f"Note: residual matrix shape is {residuals.shape}, not (30, 6).",
            args.verbose,
        )

    if unmatched:
        log(f"Unmatched folders: {len(unmatched)}", args.verbose)
        for x in unmatched:
            log(f"  {x}", args.verbose)

    if duplicates:
        log(f"Duplicate matched pairs encountered: {len(duplicates)}", args.verbose)


if __name__ == "__main__":
    main()