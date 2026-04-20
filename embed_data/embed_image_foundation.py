#!/usr/bin/env python3
"""
embed_image_foundation_fast.py

Fast image-only embedding with Hugging Face vision models.

Key speedups vs the slow version:
- Parallel JPEG decode + file I/O using DataLoader workers
- Pinned memory for faster H2D transfer
- Avoid per-image PIL open in the main process
- Single-pass streaming write into a memmap

Outputs per model:
  - img_embeddings.npy (float32, shape [N, D], in lexicographic name order)
  - img_names.txt      (N lines, lexicographic order)

Two ways to specify images:
1) COCO mode (unique images from captions_{split}2017.json):
     --coco_root /path/to/coco --split train|val
2) Plain folder mode (recursively collect images under --image_root):
     --image_root /path/to/images

Notes:
- We sort by lexicographic filename (relative path under the image root/dir).
- Embeddings are L2-normalized by default (disable with --no_l2norm).
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

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


def infer_model_load_dtype(dtype_str: str, device: torch.device) -> Optional[torch.dtype]:
    if dtype_str == "auto":
        return torch.float16 if device.type == "cuda" else None
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


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def set_hf_cache_dir(hf_cache_dir: str) -> None:
    hf_cache_dir = os.path.abspath(hf_cache_dir)
    ensure_dir(hf_cache_dir)
    os.environ.setdefault("HF_HOME", hf_cache_dir)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_cache_dir, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_cache_dir, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_cache_dir, "datasets"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# -------------------------
# Image listing
# -------------------------
@dataclass
class ImageItem:
    file_name: str   # for sorting + writing names
    path: str        # absolute path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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


def load_coco_images_sorted(coco_root: str, split: str) -> List[ImageItem]:
    split = split.lower()
    ann_path = os.path.join(coco_root, "annotations", f"captions_{split}2017.json")
    img_dir = resolve_coco_img_dir(coco_root, split)

    data = load_json(ann_path)
    images = data["images"]

    out: List[ImageItem] = []
    for img in images:
        fn = img["file_name"]
        out.append(ImageItem(file_name=fn, path=os.path.join(img_dir, fn)))

    out.sort(key=lambda x: x.file_name)
    return out


def load_images_from_folder_sorted(image_root: str, recursive: bool = True) -> List[ImageItem]:
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"--image_root does not exist: {image_root}")

    if root.is_file():
        raise ValueError(f"--image_root must be a directory, got file: {image_root}")

    if recursive:
        paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

    # file_name = relative path for stable sorting + traceability
    items = [ImageItem(file_name=str(p.relative_to(root)).replace("\\", "/"), path=str(p)) for p in paths]
    items.sort(key=lambda x: x.file_name)
    return items


# -------------------------
# Backend
# -------------------------
DINOV3_MODELS = {
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


class TransformersVisionBackend:
    """
    Vision-only embedding backend.
    - use_safetensors=True
    - trust_remote_code for DINOv3
    - pooling: auto|pooler|cls|mean
    """

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        autocast_dtype: Optional[torch.dtype],
        model_load_dtype: Optional[torch.dtype],
        pooling: str,
        revision: Optional[str],
        device_map: Optional[str],
    ):
        try:
            import safetensors  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "The 'safetensors' package is required.\n"
                "Install:\n  pip install -U safetensors\n"
                f"Original error: {e}"
            )

        from transformers import AutoImageProcessor, AutoModel

        self.model_id = model_id
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.pooling = pooling

        trust_remote_code = model_id in DINOV3_MODELS

        kwargs = {
            "use_safetensors": True,
            "trust_remote_code": trust_remote_code,
        }
        if revision is not None:
            kwargs["revision"] = revision
        if model_load_dtype is not None:
            kwargs["torch_dtype"] = model_load_dtype
        if device_map is not None:
            kwargs["device_map"] = device_map

        self.model = AutoModel.from_pretrained(model_id, **kwargs)

        proc_kwargs = {"trust_remote_code": trust_remote_code}
        if revision is not None:
            proc_kwargs["revision"] = revision
        self.processor = AutoImageProcessor.from_pretrained(model_id, **proc_kwargs)

        self.model.eval()
        if device_map is None:
            self.model.to(self.device)

    def _pool(self, outputs) -> torch.Tensor:
        if self.pooling in ("auto", "pooler") and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if self.pooling in ("auto", "cls") and hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state[:, 0, :]
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state.mean(dim=1)
        raise RuntimeError("Could not pool embeddings: no pooler_output or last_hidden_state found.")

    @torch.inference_mode()
    def embed_pixel_batch(self, pixel_batch: List[torch.Tensor]) -> torch.Tensor:
        """
        pixel_batch: list of uint8 tensors, typically [C,H,W] in RGB order.
        Returns: float32 CPU tensor [B, D]
        """
        inputs = self.processor(images=pixel_batch, return_tensors="pt")
        model_device = getattr(self.model, "device", self.device)
        inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}

        if self.autocast_dtype is not None and model_device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        emb = self._pool(outputs)
        return emb


# -------------------------
# Dataset / Loader
# -------------------------
class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[ImageItem], force_rgb: bool = True):
        self.items = items
        self.force_rgb = force_rgb
        try:
            from torchvision.io import read_image
            from torchvision.transforms.functional import rgb_to_grayscale
            self._read_image = read_image
            self._rgb_to_grayscale = rgb_to_grayscale
            self._has_torchvision = True
        except Exception:
            self._has_torchvision = False

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        path = it.path

        if self._has_torchvision:
            # torchvision.io.read_image -> uint8 tensor [C,H,W] in RGB for most formats
            x = self._read_image(path)  # uint8
            # Ensure 3 channels
            if x.ndim == 2:
                x = x.unsqueeze(0)
            if x.shape[0] == 1 and self.force_rgb:
                x = x.repeat(3, 1, 1)
            elif x.shape[0] == 4 and self.force_rgb:
                x = x[:3]  # drop alpha
            return x, idx
        else:
            # Fallback to PIL (slower but functional)
            from PIL import Image
            img = Image.open(path)
            if self.force_rgb and img.mode != "RGB":
                img = img.convert("RGB")
            # Convert to torch uint8 [C,H,W]
            x = torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()
            return x, idx


def collate_keep_order(batch):
    # batch: list of (tensor, idx)
    xs = [b[0] for b in batch]
    idxs = [b[1] for b in batch]
    return xs, idxs


# -------------------------
# Main embedding loop
# -------------------------
def embed_model(
    model_id: str,
    items_sorted: List[ImageItem],
    out_root: str,
    batch_size: int,
    device: torch.device,
    dtype_str: str,
    load_dtype_str: str,
    pooling: str,
    revision: Optional[str],
    device_map: Optional[str],
    num_workers: int,
    prefetch_factor: int,
    no_l2norm: bool,
) -> None:
    out_dir = os.path.join(out_root, safe_dirname(model_id))
    ensure_dir(out_dir)

    embeds_path = os.path.join(out_dir, "img_embeddings.npy")
    names_path = os.path.join(out_dir, "img_names.txt")

    N = len(items_sorted)
    if N == 0:
        raise RuntimeError("No images found to embed.")

    # Save names once per model (still cheap, but explicit)
    with open(names_path, "w", encoding="utf-8") as f:
        for it in items_sorted:
            f.write(it.file_name + "\n")

    autocast_dtype = infer_autocast_dtype(dtype_str, device)
    model_load_dtype = infer_model_load_dtype(load_dtype_str, device)

    backend = TransformersVisionBackend(
        model_id=model_id,
        device=device,
        autocast_dtype=autocast_dtype,
        model_load_dtype=model_load_dtype,
        pooling=pooling,
        revision=revision,
        device_map=device_map,
    )

    ds = ImagePathDataset(items_sorted, force_rgb=True)

    # Pin memory matters when device is CUDA
    pin = (device.type == "cuda")

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        collate_fn=collate_keep_order,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    # First batch to determine D
    first_xs, first_idxs = next(iter(loader))
    with torch.inference_mode():
        emb0 = backend.embed_pixel_batch(first_xs)
        if not no_l2norm:
            emb0 = l2_normalize(emb0)
        emb0_cpu = emb0.detach().float().cpu()

    D = int(emb0_cpu.shape[1])
    mm = open_or_create_memmap(embeds_path, (N, D))

    # Write first batch
    mm[first_idxs[0] : first_idxs[0] + len(first_idxs), :] = emb0_cpu.numpy()

    # Main loop over remaining batches
    pbar = tqdm(total=N, desc=f"Embedding {model_id}", unit="images")
    pbar.update(len(first_idxs))

    # Iterate again, but skip the first batch indices already written
    started = False
    for xs, idxs in loader:
        if not started:
            started = True
            continue  # skip the first batch we already handled

        with torch.inference_mode():
            emb = backend.embed_pixel_batch(xs)
            if not no_l2norm:
                emb = l2_normalize(emb)
            emb_cpu = emb.detach().float().cpu()

        # idxs are contiguous (because shuffle=False and dataset returns idx)
        start = idxs[0]
        mm[start : start + len(idxs), :] = emb_cpu.numpy()
        pbar.update(len(idxs))

    pbar.close()
    mm.flush()


def main() -> None:
    ap = argparse.ArgumentParser()

    # Either COCO mode OR folder mode
    ap.add_argument("--coco_root", type=str, default=None, help="COCO root containing annotations/ and train2017|val2017")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--image_root", type=str, default=None, help="Plain image folder mode (recursively loads images)")

    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--models", type=str, nargs="+", required=True)

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="auto")

    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--load_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--pooling", type=str, default="auto", choices=["auto", "pooler", "cls", "mean"])
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--device_map", type=str, default=None, help='Optional device_map, e.g. "auto".')

    ap.add_argument("--hf_cache_dir", type=str, default=None)

    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for decode/I/O")
    ap.add_argument("--prefetch_factor", type=int, default=4, help="Batches prefetched per worker")
    ap.add_argument("--no_l2norm", action="store_true", help="Disable L2 normalization")

    ap.add_argument("--no_tf32", action="store_true", help="Disable TF32 matmul on CUDA")

    args = ap.parse_args()

    if args.hf_cache_dir:
        set_hf_cache_dir(args.hf_cache_dir)

    device = infer_device(args.device)

    if device.type == "cuda":
        if not args.no_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    ensure_dir(args.out_root)

    # Load and sort image list once
    if args.image_root is not None:
        items = load_images_from_folder_sorted(args.image_root, recursive=True)
        print(f"Loaded {len(items):,} images from folder (sorted lexicographically): {args.image_root}")
    else:
        if args.coco_root is None:
            raise ValueError("Provide either --image_root OR --coco_root.")
        items = load_coco_images_sorted(args.coco_root, args.split)
        print(f"Loaded {len(items):,} COCO {args.split}2017 images (sorted lexicographically).")

    for model_id in args.models:
        print(f"\n==> Embedding with: {model_id}")
        embed_model(
            model_id=model_id,
            items_sorted=items,
            out_root=args.out_root,
            batch_size=args.batch_size,
            device=device,
            dtype_str=args.dtype,
            load_dtype_str=args.load_dtype,
            pooling=args.pooling,
            revision=args.revision,
            device_map=args.device_map,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            no_l2norm=args.no_l2norm,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()