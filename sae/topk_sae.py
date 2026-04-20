#!/usr/bin/env python3
"""
Train a Top-K Sparse Autoencoder (TopKSAE) on normalized embeddings with
optional dead-neuron resampling and an optional cosine learning-rate scheduler.

Saving / naming convention:
  --output_dir is treated as a ROOT.
  We create:
    {output_dir}/topk_{hidden_dim}_{model_name}_{img|text}_k_{k}/

  And inside save:
    - X_features.npz          (CSR float32)
    - decoder_weight.npy      (d_sparse, d_model)
    - encoder_weight.npy      (d_sparse, d_model)
    - encoder_bias.npy        (d_sparse,)
    - b_dec.npy               (d_model,)
    - topk_sae_state_dict.pt  (torch state_dict)
    - config.json

Scheduler behavior:
  --scheduler 0   constant learning rate (old behavior)
  --scheduler 1   cosine annealing from lr down to 0.05 * lr
"""

import os
import argparse
import random
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.sparse as sp


# ============================================================
# Top-K Sparse Autoencoder
# ============================================================

class TopKSAE(nn.Module):
    def __init__(self, d_model: int, d_sparse: int, k: int, device=None):
        super().__init__()
        if k <= 0 or k > d_sparse:
            raise ValueError(f"k must satisfy 1 <= k <= d_sparse. Got k={k}, d_sparse={d_sparse}")

        self.d_model = d_model
        self.d_sparse = d_sparse
        self.k = k

        self.encoder = nn.Linear(d_model, d_sparse, device=device)
        nn.init.zeros_(self.encoder.bias)

        self.decoder = nn.Linear(d_sparse, d_model, bias=False, device=device)
        self.b_dec = nn.Parameter(torch.zeros(d_model, device=device))

        nn.init.kaiming_uniform_(self.decoder.weight)

    def forward(self, x: torch.Tensor):
        x_cent = x - self.b_dec
        f = F.relu(self.encoder(x_cent))

        topk_values, topk_indices = torch.topk(f, self.k, dim=-1)
        z = torch.zeros_like(f)
        z.scatter_(-1, topk_indices, topk_values)

        x_hat = self.decoder(z) + self.b_dec
        return x_hat, z

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_cent = x - self.b_dec
        f = F.relu(self.encoder(x_cent))
        topk_values, topk_indices = torch.topk(f, self.k, dim=-1)
        z = torch.zeros_like(f)
        z.scatter_(-1, topk_indices, topk_values)
        return z


# ============================================================
# Helpers & Resampling
# ============================================================

@torch.no_grad()
def resample_dead_neurons(model: TopKSAE, X: torch.Tensor, activity_counter: torch.Tensor):
    dead_indices = torch.where(activity_counter == 0)[0]
    num_dead = len(dead_indices)

    if num_dead == 0:
        return 0

    rand_idx = torch.randint(0, X.shape[0], (num_dead,), device=X.device)
    batch = X[rand_idx]
    x_hat, _ = model(batch)
    residuals = batch - x_hat

    new_dec_weights = residuals / (residuals.norm(dim=-1, keepdim=True) + 1e-8)

    model.decoder.weight.data[:, dead_indices] = new_dec_weights.T
    model.encoder.weight.data[dead_indices, :] = new_dec_weights * 0.1
    model.encoder.bias.data[dead_indices] = 0

    return num_dead


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_input_path(model_spec: str, embedded_root: str) -> str:
    if "/" not in model_spec:
        raise ValueError(f"--model_spec must look like '<model>/<img|text>', got: {model_spec}")
    model_name, dtype = model_spec.rsplit("/", 1)
    dtype = dtype.strip().lower()
    if dtype not in ("img", "text"):
        raise ValueError(f"model_spec dtype must be img or text, got: {dtype}")

    fname = "img_embeddings_normalized.npy" if dtype == "img" else "text_embeddings_normalized.npy"
    return os.path.join(embedded_root, model_name, fname)


def model_tag_from_spec(model_spec: str) -> str:
    model_name, dtype = model_spec.rsplit("/", 1)
    dtype = dtype.strip().lower()
    return f"{model_name}_{dtype}"


@torch.no_grad()
def renorm_decoder_columns_(model: TopKSAE, eps: float = 1e-12):
    W = model.decoder.weight
    norms = W.norm(dim=0, keepdim=True).clamp_min(eps)
    W.div_(norms)


def build_scheduler(optimizer, scheduler_type: int, num_steps: int, base_lr: float):
    if scheduler_type == 0:
        return None
    if scheduler_type == 1:
        eta_min = 0.05 * base_lr
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_steps,
            eta_min=eta_min,
        )
    raise ValueError(f"Unsupported scheduler={scheduler_type}. Use 0 or 1.")


# ============================================================
# Training Logic
# ============================================================

def train_model(
    model: TopKSAE,
    X: torch.Tensor,
    batch_size: int,
    num_steps: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    print_every: int,
    renorm_decoder: bool = False,
    resample_every: int = 2500,
    scheduler_type: int = 0,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, scheduler_type=scheduler_type, num_steps=num_steps, base_lr=lr)
    model.train()

    N = X.shape[0]
    activity_counter = torch.zeros(model.d_sparse, device=device)

    for step in range(1, num_steps + 1):
        idx = torch.randint(0, N, (batch_size,), device=device)
        batch = X[idx]

        optimizer.zero_grad(set_to_none=True)
        x_hat, z = model(batch)

        recon_loss = F.mse_loss(x_hat, batch)

        recon_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if renorm_decoder:
            renorm_decoder_columns_(model)

        with torch.no_grad():
            activity_counter += (z > 0).sum(dim=0)

        if step % resample_every == 0 and step < (num_steps * 0.8):
            num_resampled = resample_dead_neurons(model, X, activity_counter)
            if num_resampled > 0:
                print(f"Step {step}: Resampled {num_resampled} dead neurons.")
            activity_counter.zero_()

        if print_every is not None and (step % print_every == 0 or step == 1):
            l0 = (z > 0).float().sum(dim=-1).mean().item()
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Step {step}/{num_steps} | Loss: {recon_loss.item():.6f} | "
                f"L0: {l0:.1f} | LR: {current_lr:.8f} | scheduler={scheduler_type}"
            )

    return recon_loss


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_spec", type=str, required=True)
    p.add_argument("--embedded_root", type=str, default=None)
    p.add_argument("--embedded_coco_root", type=str, default=None)
    p.add_argument("--dataset", type=str, default="coco")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--hidden_dim", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--print_every", type=int, default=500)
    p.add_argument("--truncate_d", type=int, default=None)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--renorm_decoder", action="store_true")
    p.add_argument("--mmap", action="store_true")
    p.add_argument("--scheduler", type=int, default=0, choices=[0, 1])
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedded_root = args.embedded_root
    if embedded_root is None:
        embedded_root = args.embedded_coco_root
    if embedded_root is None:
        embedded_root = "/home/kirilb/orcd/pool/PRH_data/embedded_coco"

    model_tag = model_tag_from_spec(args.model_spec)
    run_dir = os.path.join(args.output_dir, f"topk_{args.hidden_dim}_{model_tag}_k_{args.k}")
    os.makedirs(run_dir, exist_ok=True)

    input_path = resolve_input_path(args.model_spec, embedded_root)
    load_kwargs = {"mmap_mode": "r"} if args.mmap else {}
    X_np = np.load(input_path, **load_kwargs)

    if X_np.dtype != np.float32:
        X_np = np.asarray(X_np, dtype=np.float32)

    if args.truncate_d is not None:
        U, S, Vt = np.linalg.svd(X_np, full_matrices=False)
        D = min(args.truncate_d, X_np.shape[1], int((S > 1e-6).sum()))
        X_np = X_np @ Vt[:D, :].T

    X = torch.from_numpy(X_np).to(device)
    input_dim = X.shape[1]

    model = TopKSAE(d_model=input_dim, d_sparse=args.hidden_dim, k=args.k, device=device).to(device)

    with torch.no_grad():
        model.b_dec.data = X.mean(dim=0)

    train_model(
        model=model,
        X=X,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        print_every=args.print_every,
        renorm_decoder=args.renorm_decoder,
        scheduler_type=args.scheduler,
    )

    model.eval()

    torch.save(model.state_dict(), os.path.join(run_dir, "topk_sae_state_dict.pt"))

    with torch.no_grad():
        decoder_weight = model.decoder.weight.detach().cpu().numpy().T.astype(np.float32, copy=False)
        encoder_weight = model.encoder.weight.detach().cpu().numpy().astype(np.float32, copy=False)
        encoder_bias = model.encoder.bias.detach().cpu().numpy().astype(np.float32, copy=False)
        b_dec = model.b_dec.detach().cpu().numpy().astype(np.float32, copy=False)

    np.save(os.path.join(run_dir, "decoder_weight.npy"), decoder_weight)
    np.save(os.path.join(run_dir, "encoder_weight.npy"), encoder_weight)
    np.save(os.path.join(run_dir, "encoder_bias.npy"), encoder_bias)
    np.save(os.path.join(run_dir, "b_dec.npy"), b_dec)

    with torch.no_grad():
        z_all = model.encode(X).cpu()

    rows, cols = torch.nonzero(z_all, as_tuple=True)
    data = z_all[rows, cols].numpy().astype(np.float32, copy=False)
    Z_csr = sp.csr_matrix((data, (rows.numpy(), cols.numpy())), shape=z_all.shape)
    sp.save_npz(os.path.join(run_dir, "X_features.npz"), Z_csr)

    cfg = vars(args).copy()
    cfg["run_dir"] = run_dir
    cfg["input_path"] = input_path
    cfg["model_tag"] = model_tag
    cfg["input_dim"] = int(input_dim)
    cfg["resolved_embedded_root"] = embedded_root
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    print(f"Training complete. Saved artifacts to:\n  {run_dir}")


if __name__ == "__main__":
    main()
