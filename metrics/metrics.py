import torch
from sklearn.cross_decomposition import CCA
import numpy as np
# import torchaudio.functional as TAF
from numpy.linalg import LinAlgError

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    Levenshtein = None

def get_kernel(X: torch.Tensor) -> torch.Tensor:
    # Safer for fp16/bf16 inputs
    Xf = X.float()
    return Xf @ Xf.T

# -------------------------
# CKA wrapper (HSIC / unbiased_HSIC)
# -------------------------
def cka(X, Y, f_name, is_kernel: bool = False, eps: float = 1e-6):
    if f_name == "HSIC":
        f = hsic_biased
    elif f_name == "unbiased_HSIC":
        f = hsic_unbiased
    else:
        raise ValueError(f"Invalid function name: {f_name}")

    Xk = X if is_kernel else get_kernel(X)
    Yk = Y if is_kernel else get_kernel(Y)

    f_xy = f(Xk, Yk)
    f_xx = f(Xk, Xk)
    f_yy = f(Yk, Yk)

    # Put eps in the denominator (and guard negatives from unbiased estimator)
    denom = torch.sqrt(torch.clamp(f_xx * f_yy, min=eps))
    val = (f_xy / denom).item()

    return {"value": val, "name": "CKA_" + f_name}

# -------------------------
# YOUR HSIC (unchanged per your request)
# -------------------------
def hsic_unbiased(K, L):
    """
    Copied from https://github.com/minyoungg/platonic-rep/blob/main/metrics.py
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC) as per Equation 5 in the paper.
    > Reference: https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K, L):
    """ 
    Copied from https://github.com/minyoungg/platonic-rep/blob/main/metrics.py
    Compute the biased HSIC (the original CKA) """
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)

# -------------------------
# TOP-K KNN overlap
# -------------------------
def top_k_knn(X, Y, k, is_kernel: bool = False):
    Xk = X if is_kernel else get_kernel(X)
    Yk = Y if is_kernel else get_kernel(Y)

    Xk_sorted = Xk.sort(dim=1, descending=True).indices[:, 1:k+1]
    Yk_sorted = Yk.sort(dim=1, descending=True).indices[:, 1:k+1]

    # Make GPU-safe: move indices to CPU for Python set ops
    Xk_sorted = Xk_sorted.cpu()
    Yk_sorted = Yk_sorted.cpu()

    common = torch.zeros(Xk_sorted.shape[0], dtype=torch.float32)
    for i in range(Xk_sorted.shape[0]):
        X_ind = set(Xk_sorted[i].tolist())
        Y_ind = set(Yk_sorted[i].tolist())
        common[i] = len(X_ind.intersection(Y_ind)) / k

    return {"value": common.mean().item(), "name": "TOP_K_KNN"}

# -------------------------
# KNN edit distance
# -------------------------
def knn_edit_distance_old(X, Y, k, is_kernel: bool = False):
    Xk = X if is_kernel else get_kernel(X)
    Yk = Y if is_kernel else get_kernel(Y)

    X_order = Xk.argsort(dim=1, descending=True)[:, 1:k+1]
    Y_order = Yk.argsort(dim=1, descending=True)[:, 1:k+1]
    # torchaudio edit_distance is happiest on CPU integer tensors
    X_order = X_order.cpu()
    Y_order = Y_order.cpu()

    distances = torch.zeros(X_order.shape[0], dtype=torch.float32)
    for i in range(X_order.shape[0]):
        d = TAF.edit_distance(X_order[i], Y_order[i])
        # d may be python int or tensor depending on version
        distances[i] = float(d)

    return {"value": distances.mean().item(), "name": "KNN_EDIT_DISTANCE"}


import torch

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    Levenshtein = None


def knn_edit_distance(X, Y, k, is_kernel: bool = False):
    """
    Uses RapidFuzz Levenshtein on the top-k neighbor index sequences.

    X, Y: either representations [N, D] (is_kernel=False) or kernels [N, N] (is_kernel=True)
    k: number of neighbors
    """
    if Levenshtein is None:
        raise ImportError(
            "rapidfuzz is not installed. Install with: pip install rapidfuzz "
            "or conda install -c conda-forge rapidfuzz"
        )

    Xk = X if is_kernel else (X.float() @ X.float().T)
    Yk = Y if is_kernel else (Y.float() @ Y.float().T)

    # Top-k neighbor indices (exclude self at position 0)
    X_order = Xk.argsort(dim=1, descending=True)[:, 1:k+1].detach().cpu()
    Y_order = Yk.argsort(dim=1, descending=True)[:, 1:k+1].detach().cpu()

    # RapidFuzz runs in C++ but we still loop rows to feed sequences
    N = X_order.shape[0]
    total = 0.0
    for i in range(N):
        # tuple tends to be a tiny bit faster / lighter than list
        a = tuple(int(x) for x in X_order[i])
        b = tuple(int(y) for y in Y_order[i])
        total += Levenshtein.distance(a, b)

    return {"value": total / N, "name": "KNN_EDIT_DISTANCE"}


def _corrcoef_cols(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Column-wise corr(A[:,i], B[:,i]) with protection against constant columns."""
    k = A.shape[1]
    out = np.empty((k,), dtype=np.float64)
    for i in range(k):
        a = A[:, i]
        b = B[:, i]
        sa = a.std()
        sb = b.std()
        if sa == 0.0 or sb == 0.0:
            out[i] = 0.0
        else:
            out[i] = np.corrcoef(a, b)[0, 1]
    return out


def _canonical_corrs_whiten_svd(U1: np.ndarray, U2: np.ndarray, k: int, eps: float = 1e-8) -> np.ndarray:
    """
    Stable CCA via whitening + SVD. Returns top-k canonical correlations.
    eps only stabilizes *degenerate* windows (the ones that would crash).
    """
    import scipy.linalg  # local import so you don't pay it unless fallback triggers

    # Center
    U1 = U1 - U1.mean(axis=0, keepdims=True)
    U2 = U2 - U2.mean(axis=0, keepdims=True)

    n = U1.shape[0]
    if n < 2:
        return np.full((k,), np.nan, dtype=np.float64)

    # Covariances
    C11 = (U1.T @ U1) / (n - 1)
    C22 = (U2.T @ U2) / (n - 1)
    C12 = (U1.T @ U2) / (n - 1)

    # tiny ridge for stability on singular/near-singular cases
    C11 = C11 + eps * np.eye(C11.shape[0], dtype=C11.dtype)
    C22 = C22 + eps * np.eye(C22.shape[0], dtype=C22.dtype)

    # Whitening via eigh (symmetric, stable)
    w1, V1 = np.linalg.eigh(C11)
    w2, V2 = np.linalg.eigh(C22)
    w1 = np.maximum(w1, eps)
    w2 = np.maximum(w2, eps)

    W1 = V1 @ (np.diag(1.0 / np.sqrt(w1)) @ V1.T)
    W2 = V2 @ (np.diag(1.0 / np.sqrt(w2)) @ V2.T)

    T = W1 @ C12 @ W2

    # More robust LAPACK driver than the default divide-and-conquer on some cases
    _, s, _ = scipy.linalg.svd(T, full_matrices=False, lapack_driver="gesvd")
    s = np.clip(s, 0.0, 1.0)
    return s[:k]


def svcca(X, Y, cca_dim: int = 10):
    """
    Same behavior as your original in the normal case,
    but robust to sklearn/scipy "SVD did not converge" failures.

    Returns: {'value': float,  'name': 'SVCCA'}
    """
    # -------------------------
    # 1) Compute SVD subspaces (same as you had, with a bit more guard-rails)
    # -------------------------
    try:
        # torch.svd_lowrank requires float and works best on CPU/GPU depending on shapes
        U1, _, _ = torch.svd_lowrank(X, q=cca_dim)
        U2, _, _ = torch.svd_lowrank(Y, q=cca_dim)
    except Exception:
        # Full SVD fallback (your original idea)
        U1_full, Sigma_1, _ = X.svd()
        U2_full, Sigma_2, _ = Y.svd()

        effective_dim_1 = int((Sigma_1 > 1e-6).sum().item())
        effective_dim_2 = int((Sigma_2 > 1e-6).sum().item())
        cca_dim = int(min(effective_dim_1, effective_dim_2, cca_dim))

        U1 = U1_full[:, :cca_dim]
        U2 = U2_full[:, :cca_dim]

    # Move to numpy float64 for numerical stability (this alone often fixes convergence)
    U1 = U1.detach().cpu().numpy().astype(np.float64, copy=False, order="F")
    U2 = U2.detach().cpu().numpy().astype(np.float64, copy=False, order="F")

    # Clean any NaN/Inf that slipped in (sklearn uses check_finite=False internally sometimes)
    if not (np.isfinite(U1).all() and np.isfinite(U2).all()):
        U1 = np.nan_to_num(U1, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        U2 = np.nan_to_num(U2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # sklearn CCA needs n_components <= n_samples - 1 and <= feature dims
    k = int(min(cca_dim, U1.shape[0] - 1, U1.shape[1], U2.shape[1]))
    if k < 1:
        return {"value": np.nan, "std": None, "name": "SVCCA"}

    # -------------------------
    # 2) Compute CCA robustly: default -> retry -> stable fallback
    # -------------------------
    def _run_sklearn_cca(algorithm=None):
        if algorithm is None:
            cca = CCA(n_components=k)
        else:
            cca = CCA(n_components=k, algorithm=algorithm)
        cca.fit(U1, U2)
        A, B = cca.transform(U1, U2)

        # keep your “avoid NaN” jitter
        A = A + 1e-10 * np.random.randn(*A.shape)
        B = B + 1e-10 * np.random.randn(*B.shape)

        corrs = _corrcoef_cols(A, B)
        return float(np.mean(corrs))

    try:
        svcca_similarity = _run_sklearn_cca(algorithm=None)
    except (LinAlgError, ValueError, FloatingPointError) as _:
        # Retry: more stable CCA solver
        try:
            svcca_similarity = _run_sklearn_cca(algorithm="svd")
        except Exception:
            # Final fallback: whitening + robust SVD
            corrs = _canonical_corrs_whiten_svd(U1, U2, k, eps=1e-8)
            svcca_similarity = float(np.nanmean(corrs))

    return {"value": svcca_similarity, "name": "SVCCA"}

