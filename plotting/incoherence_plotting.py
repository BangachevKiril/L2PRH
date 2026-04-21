import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUPPORTED_STAT_FILES = (
    "incoherence_statistics.npz",
    "img_incoherence_statistics.npy",
    "text_incoherence_statistics.npy",
    "img_incoherence_statistics.py",
    "text_incoherence_statistics.py",
)

VALID_STATS = {
    'G_mean_offdiag',
    'G_std_offdiag',
    'G_min_offdiag',
    'G_max_offdiag',
    'G_mean_abs_offdiag',
    'G_abs_offdiag_p50',
    'G_abs_offdiag_p75',
    'G_abs_offdiag_p90',
    'G_abs_offdiag_p95',
    'G_num_offdiag',
    'G_num_rows',
    'dense_dimension',
    'embedding_dimension',
    'num_vectors',
    'num_sampled_vectors',
    'sampled_indices',
    'representation_shape',
    'source_file',
    'modality',
    'sample_seed',
    'random_batch_subset',
    'statistics_scope',
    'compute_dtype',
    'chunk_size',
    'std_definition',
    'sparse_dimension_full',
    'sparse_dimension_truncated',
    'sparse_k_full',
    'sparse_k_truncated',
    'weight_shape',
    'idx_shape',
    'G_t_mean_offdiag',
    'G_t_std_offdiag',
    'G_t_min_offdiag',
    'G_t_max_offdiag',
    'G_t_mean_abs_offdiag',
    'G_t_num_offdiag',
    'G_t_num_rows',
}

STAT_ALIASES = {
    'dense_dimension': ('dense_dimension', 'embedding_dimension'),
    'embedding_dimension': ('embedding_dimension', 'dense_dimension'),
    'G_num_rows': ('G_num_rows', 'num_vectors', 'num_sampled_vectors'),
    'num_vectors': ('num_vectors', 'num_sampled_vectors', 'G_num_rows'),
    'num_sampled_vectors': ('num_sampled_vectors', 'num_vectors', 'G_num_rows'),
}

FOLDER_RE = re.compile(r'^topk_(?P<d>\d+?)_(?P<model>.+)_k_(?P<k>[^/]+)$')



def _normalize_scalar(x):
    if isinstance(x, np.ndarray):
        if np.ndim(x) != 0:
            raise TypeError(f"expected scalar, got array with shape {np.shape(x)}")
        x = x.item()
    return x



def _as_root_list(root_dir):
    if isinstance(root_dir, (str, os.PathLike, Path)):
        return [str(root_dir)]
    return [str(p) for p in root_dir]



def _load_stats_file(stat_path):
    stat_path = str(stat_path)

    if stat_path.endswith('.npz'):
        with np.load(stat_path, allow_pickle=True) as data:
            payload = {k: data[k] for k in data.files}
        return payload, 'dictionary'

    if stat_path.endswith('.npy') or stat_path.endswith('.py'):
        obj = np.load(stat_path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.shape == ():
            payload = obj.item()
        elif isinstance(obj, dict):
            payload = obj
        else:
            raise ValueError(f"Expected dict-like payload in {stat_path}, got type {type(obj)}")
        return payload, 'raw_embedding'

    raise ValueError(f"Unsupported stats file type: {stat_path}")



def _get_stat_value(payload, key):
    for candidate in STAT_ALIASES.get(key, (key,)):
        if candidate in payload:
            return payload[candidate]
    raise KeyError(key)



def _has_stat(payload, key):
    try:
        _get_stat_value(payload, key)
        return True
    except KeyError:
        return False



def _extract_folder_filter_info(folder_path, payload):
    if 'sparse_dimension_full' in payload and 'sparse_k_full' in payload:
        try:
            d = round(float(_normalize_scalar(payload['sparse_dimension_full'])))
            k = float(_normalize_scalar(payload['sparse_k_full']))
            return d, k
        except Exception:
            pass

    m = FOLDER_RE.match(Path(folder_path).name)
    if m is None:
        return None, None

    d = int(m.group('d'))
    k_raw = m.group('k')
    try:
        k = float(k_raw)
    except Exception:
        return d, None
    return d, k



def _iter_candidate_stat_files(root_dir):
    for root in _as_root_list(root_dir):
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in SUPPORTED_STAT_FILES:
                if name in filenames:
                    yield os.path.join(dirpath, name)



def _collect_points(
    root_dir,
    sparse_dimension_full,
    sparse_k_full,
    metric_1,
    dimension,
    *,
    metric_2=None,
    atol_k=1e-8,
    rtol_k=1e-8,
    verbose=True,
):
    xs = []
    ys = []
    matched_paths = []
    matched_sources = []

    for stat_path in _iter_candidate_stat_files(root_dir):
        try:
            payload, source_type = _load_stats_file(stat_path)

            this_sparse_dimension_full, this_sparse_k_full = _extract_folder_filter_info(
                os.path.dirname(stat_path), payload
            )

            if this_sparse_dimension_full is not None and this_sparse_k_full is not None:
                if round(this_sparse_dimension_full) != sparse_dimension_full:
                    continue
                if not np.isclose(this_sparse_k_full, sparse_k_full, atol=atol_k, rtol=rtol_k):
                    continue
            elif source_type == 'dictionary':
                if verbose:
                    print(f"[skip missing filter info] {stat_path}")
                continue

            if not _has_stat(payload, metric_1):
                if verbose:
                    print(f"[skip missing {metric_1}] {stat_path}")
                continue
            if not _has_stat(payload, dimension):
                if verbose:
                    print(f"[skip missing {dimension}] {stat_path}")
                continue
            if metric_2 is not None and not _has_stat(payload, metric_2):
                if verbose:
                    print(f"[skip missing {metric_2}] {stat_path}")
                continue

            x1 = _normalize_scalar(_get_stat_value(payload, metric_1))
            y = _normalize_scalar(_get_stat_value(payload, dimension))

            try:
                x1 = float(x1)
                y = float(y)
            except Exception:
                if verbose:
                    print(f"[skip non-float-convertible pair] {stat_path}")
                continue

            if metric_2 is None:
                x = x1
            else:
                x2 = _normalize_scalar(_get_stat_value(payload, metric_2))
                try:
                    x2 = float(x2)
                except Exception:
                    if verbose:
                        print(f"[skip non-float-convertible {metric_2}] {stat_path}")
                    continue
                x = x1 - x2

            if not np.isfinite(x) or not np.isfinite(y):
                if verbose:
                    print(f"[skip non-finite values] {stat_path}")
                continue

            xs.append(x)
            ys.append(y)
            matched_paths.append(stat_path)
            matched_sources.append(source_type)

        except Exception as e:
            if verbose:
                print(f"[error] {stat_path}: {e}")

    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        matched_paths,
        matched_sources,
    )



def scatter_incoherence_statistics(
    root_dir,
    sparse_dimension_full,
    sparse_k_full,
    metric_1,
    dimension,
    *,
    metric_2=None,
    atol_k=1e-8,
    rtol_k=1e-8,
    figsize=(6.5, 8.5),
    marker_size=22,
    alpha=0.8,
    title=None,
    partial_title=None,
    xlabel=None,
    ylabel=None,
    show_grid=True,
    verbose=True,
    print_names=False,
    save_path=None,
    dpi=200,
    close_after_plot=False,
):
    """
    Recursively search for incoherence stats files under one root or a list of roots.

    Accepted file types:
      - incoherence_statistics.npz
      - img_incoherence_statistics.npy / .py
      - text_incoherence_statistics.npy / .py

    Produces a two-panel figure with a shared x-axis:
      - top: dictionary incoherence
      - bottom: raw embedding incoherence

    The y-axes are independent and both use log scale.

    Returns
    -------
    xs : np.ndarray
        All collected x-values across both panels.
    ys : np.ndarray
        All collected y-values across both panels.
    matched_paths : list[str]
        Paths of all matched files.
    """
    if metric_1 not in VALID_STATS:
        raise ValueError(f"metric_1={metric_1!r} is not in VALID_STATS")
    if dimension not in VALID_STATS:
        raise ValueError(f"dimension={dimension!r} is not in VALID_STATS")
    if metric_2 is not None and metric_2 not in VALID_STATS:
        raise ValueError(f"metric_2={metric_2!r} is not in VALID_STATS")

    xs, ys, matched_paths, matched_sources = _collect_points(
        root_dir,
        sparse_dimension_full,
        sparse_k_full,
        metric_1,
        dimension,
        metric_2=metric_2,
        atol_k=atol_k,
        rtol_k=rtol_k,
        verbose=verbose,
    )

    if print_names:
        print("Files plotted:")
        for p in matched_paths:
            print(p)

    dict_mask = np.array([s == 'dictionary' for s in matched_sources], dtype=bool)
    raw_mask = np.array([s == 'raw_embedding' for s in matched_sources], dtype=bool)

    fig, (ax_dict, ax_raw) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={'hspace': 0.08, 'height_ratios': [1, 1]},
    )

    def _plot_panel(ax, xvals, yvals, *, marker, color, label, empty_message):
        positive_mask = np.isfinite(xvals) & np.isfinite(yvals) & (yvals > 0)
        xplot = xvals[positive_mask]
        yplot = yvals[positive_mask]

        if xplot.size > 0:
            ax.scatter(
                xplot,
                yplot,
                s=marker_size,
                alpha=alpha,
                marker=marker,
                color=color,
                label=label,
            )
            ax.legend(fontsize=12, loc='best')
        else:
            ax.text(
                0.5,
                0.5,
                empty_message,
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=12,
            )

        ax.set_yscale('log')
        if show_grid:
            ax.grid(True, alpha=0.3, which='both')

    _plot_panel(
        ax_dict,
        xs[dict_mask],
        ys[dict_mask],
        marker='o',
        color='C0',
        label='Dictionary',
        empty_message='No matching dictionary stats',
    )
    _plot_panel(
        ax_raw,
        xs[raw_mask],
        ys[raw_mask],
        marker='^',
        color='orange',
        label='Raw Embedding',
        empty_message='No matching raw embedding stats',
    )

    if xlabel is not None:
        final_xlabel = xlabel
    else:
        final_xlabel = metric_1 if metric_2 is None else f"{metric_1} - {metric_2}"

    if ylabel is not None:
        final_ylabel = ylabel
    else:
        final_ylabel = dimension

    if title is None and partial_title is None:
        if metric_2 is None:
            final_title = (
                f"{dimension} vs {metric_1}\n"
                f"(dictionary size = {sparse_dimension_full}, "
                f"sparsity = {sparse_k_full})"
            )
        else:
            final_title = (
                f"{dimension} vs ({metric_1} - {metric_2})\n"
                f"(dictionary size = {sparse_dimension_full}, "
                f"sparsity = {sparse_k_full})"
            )
    elif title is None:
        final_title = (
            f"{str(partial_title)}\n"
            f"(dictionary size = {sparse_dimension_full}, "
            f"sparsity = {sparse_k_full})"
        )
    else:
        final_title = title

    fig.suptitle(final_title, fontsize=20)
    ax_dict.set_ylabel(final_ylabel, fontsize=16)
    ax_raw.set_ylabel(final_ylabel, fontsize=16)
    ax_raw.set_xlabel(final_xlabel, fontsize=16)

    # Keep top panel clean since x is shared.
    ax_dict.tick_params(axis='x', labelbottom=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        root, ext = os.path.splitext(save_path)
        if ext.lower() != ".pdf":
            save_path = root + ".pdf"

        save_dir = os.path.dirname(os.path.abspath(save_path))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(save_path, format="pdf", bbox_inches="tight")

        if verbose:
            print(f"Saved figure to: {save_path}")

    plt.show()

    if close_after_plot:
        plt.close(fig)

    if verbose:
        num_dict = int(dict_mask.sum())
        num_raw = int(raw_mask.sum())
        print(f"Matched files: {len(matched_paths)}")
        print(f"Breakdown: {{'dictionary': {num_dict}, 'raw_embedding': {num_raw}}}")

    return xs, ys, matched_paths
