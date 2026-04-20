#!/usr/bin/env python3
"""
embed_coco_img_and_pooled_captions_fastio.py

Faster I/O version of your COCO image + pooled-captions embedder.

Semantics preserved:
- One row per image.
- Text: mean-pool all caption embeddings for that image, then L2-normalize.
- Sort by lexicographic image file name.
- Save:
    img_embeddings.npy   float32 memmap [M, D]
    text_embeddings.npy  float32 memmap [M, D]
    img_names.txt        M lines

Speedups:
- Parallel image decode via DataLoader workers (torchvision.io.read_image)
- pin_memory=True in DataLoader (main-process pin thread)
- non-blocking H2D transfer
- vectorized caption mean pooling via index_add_
- chunked text embedding

Notes:
- If an image cannot be decoded, we replace it with a black image (keeps alignment).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm


# -------------------------
# Helpers
# -------------------------
def safe_dirname(model_id: str) -> str:
    s = model_id.replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def infer_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def infer_autocast_dtype(dtype_str: str, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if dtype_str == "auto":
        return torch.float16
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp32":
        return None
    raise ValueError(f"Unknown dtype: {dtype_str}")


def open_or_create_memmap(path: str, shape: Tuple[int, int]) -> np.memmap:
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r+")
        if tuple(arr.shape) != tuple(shape):
            raise RuntimeError(f"Existing {path} has shape {arr.shape}, expected {shape}.")
        return arr
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def get_existing_dim(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    arr = np.load(path, mmap_mode="r")
    return int(arr.shape[1])


def set_hf_cache_dir(hf_cache_dir: str) -> None:
    """
    Prefer HF_HOME. TRANSFORMERS_CACHE is deprecated but leaving it set is harmless.
    """
    hf_cache_dir = os.path.abspath(hf_cache_dir)
    ensure_dir(hf_cache_dir)
    os.environ.setdefault("HF_HOME", hf_cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_cache_dir, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_cache_dir, "datasets"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def apply_mask_list(xs: List[torch.Tensor], mask: torch.Tensor) -> List[torch.Tensor]:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    keep = mask.nonzero(as_tuple=False).flatten().tolist()
    return [xs[i] for i in keep]


def sort_batch_by_indices(xs: List[torch.Tensor], idxs: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
    # DataLoader should already be ordered, but keep invariants.
    if idxs.numel() <= 1:
        return xs, idxs
    if torch.all(idxs[1:] >= idxs[:-1]):
        return xs, idxs
    order = torch.argsort(idxs)
    idxs2 = idxs[order]
    order_list = order.tolist()
    xs2 = [xs[i] for i in order_list]
    return xs2, idxs2


# -------------------------
# COCO per-image records
# -------------------------
@dataclass
class CocoImage:
    image_id: int
    file_name: str
    image_path: str
    captions: List[str]


def resolve_coco_img_dir(coco_root: str, split: str) -> str:
    split = split.strip().lower()
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got: {split!r}")

    cand1 = os.path.join(coco_root, f"{split}2017")
    if os.path.isdir(cand1):
        return cand1

    cand2 = os.path.join(coco_root, "images", f"{split}2017")
    if os.path.isdir(cand2):
        return cand2

    raise FileNotFoundError(
        "Could not find COCO image directory. Tried:\n"
        f"  1) {cand1}\n"
        f"  2) {cand2}"
    )


def load_coco_images(coco_root: str, split: str) -> List[CocoImage]:
    split = split.lower()
    ann_path = os.path.join(coco_root, "annotations", f"captions_{split}2017.json")
    img_dir = resolve_coco_img_dir(coco_root, split)

    data = load_json(ann_path)

    id_to_file: Dict[int, str] = {int(img["id"]): str(img["file_name"]) for img in data["images"]}
    caps_by_id: Dict[int, List[str]] = {img_id: [] for img_id in id_to_file.keys()}

    for ann in data["annotations"]:
        image_id = int(ann["image_id"])
        cap = str(ann["caption"]).replace("\n", " ").strip()
        if image_id in caps_by_id:
            caps_by_id[image_id].append(cap)

    images: List[CocoImage] = []
    for image_id, fn in id_to_file.items():
        images.append(
            CocoImage(
                image_id=image_id,
                file_name=fn,
                image_path=os.path.join(img_dir, fn),
                captions=caps_by_id.get(image_id, []),
            )
        )

    images.sort(key=lambda x: x.file_name)
    return images


# -------------------------
# Backends
# -------------------------
class EncoderBackend:
    def embed_images_pixels_uint8(self, pixel_batch: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def close(self) -> None:
        pass


class TransformersCLIPLikeBackend(EncoderBackend):
    """
    Uses AutoModel + AutoProcessor with get_image_features/get_text_features.
    Does NOT normalize internally.
    """

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        autocast_dtype: Optional[torch.dtype],
        revision: Optional[str] = None,
        use_fast_processor: Optional[bool] = None,
    ):
        try:
            import safetensors  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "The 'safetensors' package is required.\n"
                "Install:\n  pip install -U safetensors\n"
                f"Original error: {e}"
            )

        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id
        self.device = device
        self.autocast_dtype = autocast_dtype

        kwargs = {"use_safetensors": True}
        if revision is not None:
            kwargs["revision"] = revision

        self.model = AutoModel.from_pretrained(model_id, **kwargs)

        proc_kwargs = {}
        if revision is not None:
            proc_kwargs["revision"] = revision
        if use_fast_processor is not None:
            proc_kwargs["use_fast"] = bool(use_fast_processor)

        self.processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)

        self.model.eval().to(self.device)

        if not (hasattr(self.model, "get_image_features") and hasattr(self.model, "get_text_features")):
            raise RuntimeError(f"{self.model_id} does not expose get_image_features/get_text_features.")

        mt = str(getattr(getattr(self.model, "config", None), "model_type", "")).lower()
        mid = model_id.lower()
        self.is_siglip_family = (mt in ("siglip", "siglip2")) or ("siglip" in mid)
        self.siglip_text_max_len = None
        if self.is_siglip_family:
            try:
                self.siglip_text_max_len = int(self.model.config.text_config.max_position_embeddings)
            except Exception:
                self.siglip_text_max_len = 64

    @torch.no_grad()
    def embed_images_pixels_uint8(self, pixel_batch: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # processor handles resizing/cropping/normalization
        inputs = self.processor(images=pixel_batch, return_tensors="pt")
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        img_kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask", "token_type_ids")}

        if self.autocast_dtype is not None and self.device.type == "cuda":
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                feats = self.model.get_image_features(**img_kwargs)
        else:
            feats = self.model.get_image_features(**img_kwargs)

        return feats.detach().float().cpu()

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        if self.is_siglip_family:
            inputs = self.processor(
                text=texts,
                padding="max_length",
                truncation=True,
                max_length=self.siglip_text_max_len,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")

        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        txt_kwargs = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask", "token_type_ids")}

        if self.autocast_dtype is not None and self.device.type == "cuda":
            with torch.autocast("cuda", dtype=self.autocast_dtype):
                feats = self.model.get_text_features(**txt_kwargs)
        else:
            feats = self.model.get_text_features(**txt_kwargs)

        return feats.detach().float().cpu()


# -------------------------
# Image Dataset with parallel decode
# -------------------------
class CocoImageDataset(torch.utils.data.Dataset):
    def __init__(self, images: List[CocoImage], force_rgb: bool = True):
        self.images = images
        self.force_rgb = force_rgb
        try:
            from torchvision.io import read_image
            self._read_image = read_image
            self._has_torchvision = True
        except Exception:
            self._has_torchvision = False

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        rec = self.images[idx]
        path = rec.image_path

        if self._has_torchvision:
            try:
                x = self._read_image(path)  # uint8 [C,H,W]
            except Exception:
                x = torch.zeros((3, 32, 32), dtype=torch.uint8)

            if x.ndim == 2:
                x = x.unsqueeze(0)
            if self.force_rgb:
                if x.shape[0] == 1:
                    x = x.repeat(3, 1, 1)
                elif x.shape[0] == 4:
                    x = x[:3]
                elif x.shape[0] > 4:
                    x = x[:3]
            return x.contiguous(), idx

        # fallback PIL
        from PIL import Image
        try:
            img = Image.open(path)
            if self.force_rgb and img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        except Exception:
            x = torch.zeros((3, 32, 32), dtype=torch.uint8)
        return x, idx


def collate_list(batch):
    # IMPORTANT: do NOT call .pin_memory() here; collate runs in worker processes.
    xs = [b[0] for b in batch]
    idxs = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, idxs


# -------------------------
# Vectorized caption pooling
# -------------------------
def mean_pool_by_group(emb: torch.Tensor, group_ids: torch.Tensor, num_groups: int) -> torch.Tensor:
    N, D = emb.shape
    sums = torch.zeros((num_groups, D), dtype=emb.dtype)
    counts = torch.zeros((num_groups,), dtype=torch.long)
    sums.index_add_(0, group_ids, emb)
    counts.index_add_(0, group_ids, torch.ones((N,), dtype=torch.long))
    counts = counts.clamp_min(1).unsqueeze(1)
    return sums / counts


# -------------------------
# Main embedding logic
# -------------------------
def embed_model(
    model_id: str,
    images: List[CocoImage],
    out_root: str,
    batch_size: int,
    device: torch.device,
    dtype_str: str,
    normalize: bool,
    resume: bool,
    revision: Optional[str],
    num_workers: int,
    prefetch_factor: int,
    text_chunk_size: int,
    use_fast_processor: Optional[bool],
) -> None:
    out_dir = os.path.join(out_root, safe_dirname(model_id))
    ensure_dir(out_dir)

    progress_path = os.path.join(out_dir, "progress.json")
    meta_path = os.path.join(out_dir, "meta.json")

    img_embeds_path = os.path.join(out_dir, "img_embeddings.npy")
    txt_embeds_path = os.path.join(out_dir, "text_embeddings.npy")
    names_path = os.path.join(out_dir, "img_names.txt")

    M = len(images)
    autocast_dtype = infer_autocast_dtype(dtype_str, device)

    backend = TransformersCLIPLikeBackend(
        model_id=model_id,
        device=device,
        autocast_dtype=autocast_dtype,
        revision=revision,
        use_fast_processor=use_fast_processor,
    )

    # names
    if not os.path.exists(names_path):
        with open(names_path, "w", encoding="utf-8") as f:
            for im in images:
                f.write(im.file_name + "\n")

    # resume
    start_idx = 0
    if resume and os.path.exists(progress_path):
        try:
            prog = load_json(progress_path)
            start_idx = int(prog.get("next_index", 0))
            start_idx = max(0, min(start_idx, M))
        except Exception:
            start_idx = 0

    ds = CocoImageDataset(images, force_rgb=True)
    pin = (device.type == "cuda")

    dl_kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        collate_fn=collate_list,
    )
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = prefetch_factor

    loader = torch.utils.data.DataLoader(**dl_kwargs)

    # determine D
    D = get_existing_dim(img_embeds_path)
    if D is None:
        got = False
        for xs, idxs in loader:
            xs, idxs = sort_batch_by_indices(xs, idxs)
            if int(idxs[-1]) < start_idx:
                continue

            mask = idxs >= start_idx
            xs_f = apply_mask_list(xs, mask)
            idxs_f = idxs[mask]
            if len(xs_f) == 0:
                continue

            img_e = backend.embed_images_pixels_uint8(xs_f)

            probe_texts: List[str] = []
            for ii in idxs_f.tolist():
                caps = images[ii].captions
                probe_texts.append(caps[0] if caps else "")
            txt_e = backend.embed_texts(probe_texts)

            D = int(img_e.shape[1])
            if int(txt_e.shape[1]) != D:
                raise RuntimeError(f"Image/text dims differ: img D={D}, text D={int(txt_e.shape[1])}")

            got = True
            break

        if not got or D is None:
            raise RuntimeError("Could not probe embedding dimension (no images?).")

    img_mm = open_or_create_memmap(img_embeds_path, (M, D))
    txt_mm = open_or_create_memmap(txt_embeds_path, (M, D))

    if start_idx >= M:
        atomic_save_json(meta_path, {"model_id": model_id, "images": M, "done": True, "timestamp": time.time()})
        backend.close()
        return

    pbar = tqdm(total=(M - start_idx), desc=f"Embedding {model_id}", unit="images")

    for xs, idxs in loader:
        xs, idxs = sort_batch_by_indices(xs, idxs)
        b_start = int(idxs[0])
        b_end = int(idxs[-1]) + 1

        if b_end <= start_idx:
            continue

        if b_start < start_idx:
            mask = idxs >= start_idx
            xs = apply_mask_list(xs, mask)
            idxs = idxs[mask]
            if idxs.numel() == 0:
                continue
            b_start = int(idxs[0])
            b_end = int(idxs[-1]) + 1

        # ---- images ----
        img_e = backend.embed_images_pixels_uint8(xs)
        if normalize:
            img_e = l2_normalize(img_e)
        img_mm[b_start:b_end, :] = img_e.numpy()

        # ---- captions ----
        flat_texts: List[str] = []
        group_ids_list: List[int] = []
        idxs_list = idxs.tolist()

        for local_i, global_i in enumerate(idxs_list):
            caps = images[global_i].captions
            if not caps:
                caps = [""]
            flat_texts.extend(caps)
            group_ids_list.extend([local_i] * len(caps))

        group_ids = torch.tensor(group_ids_list, dtype=torch.long)

        txt_chunks: List[torch.Tensor] = []
        gi_chunks: List[torch.Tensor] = []
        n = len(flat_texts)
        pos = 0
        while pos < n:
            q = min(n, pos + text_chunk_size)
            txt_chunks.append(backend.embed_texts(flat_texts[pos:q]))
            gi_chunks.append(group_ids[pos:q])
            pos = q

        txt_all = torch.cat(txt_chunks, dim=0)
        gi_all = torch.cat(gi_chunks, dim=0)

        pooled = mean_pool_by_group(txt_all, gi_all, num_groups=idxs.numel())
        if normalize:
            pooled = l2_normalize(pooled)
        txt_mm[b_start:b_end, :] = pooled.numpy()

        # progress
        next_index = b_end
        atomic_save_json(progress_path, {"next_index": next_index})
        pbar.update(b_end - b_start)

    pbar.close()
    img_mm.flush()
    txt_mm.flush()

    atomic_save_json(
        meta_path,
        {
            "model_id": model_id,
            "images": M,
            "normalize_l2_final": normalize,
            "device": str(device),
            "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else "none",
            "revision": revision if revision is not None else "default",
            "use_fast_processor": use_fast_processor if use_fast_processor is not None else "default",
            "sorted_by": "lexicographic_file_name",
            "caption_pooling": "mean_then_l2_normalize",
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor if num_workers > 0 else 0,
            "text_chunk_size": text_chunk_size,
            "done": True,
            "timestamp": time.time(),
        },
    )

    backend.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--no_normalize", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--hf_cache_dir", type=str, default=None)
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--models", type=str, nargs="+", required=True)

    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--text_chunk_size", type=int, default=2048)

    ap.add_argument(
        "--use_fast_processor",
        type=int,
        default=-1,
        help="Set to 1/0 to force fast/slow processor. Default -1 leaves HF default.",
    )

    args = ap.parse_args()

    if args.hf_cache_dir:
        set_hf_cache_dir(args.hf_cache_dir)

    device = infer_device(args.device)
    normalize = not args.no_normalize

    if args.use_fast_processor == -1:
        use_fast_processor: Optional[bool] = None
    else:
        use_fast_processor = bool(args.use_fast_processor)

    images = load_coco_images(args.coco_root, args.split)
    print(f"Loaded {len(images):,} COCO {args.split}2017 images. Sorted lexicographically.")

    ensure_dir(args.out_root)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    for model_id in args.models:
        print(f"\n==> Embedding with: {model_id}")
        embed_model(
            model_id=model_id,
            images=images,
            out_root=args.out_root,
            batch_size=args.batch_size,
            device=device,
            dtype_str=args.dtype,
            normalize=normalize,
            resume=args.resume,
            revision=args.revision,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            text_chunk_size=args.text_chunk_size,
            use_fast_processor=use_fast_processor,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()