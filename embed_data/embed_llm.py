#!/usr/bin/env python
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    p = argparse.ArgumentParser(
        "Embed COCO captions with an LLM (middle layer only), pool per image, sort, save."
    )
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument(
        "--captions_path",
        type=str,
        required=True,
        help="COCO captions JSON (must contain images + annotations).",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--hf_cache_dir", type=str, default=None)
    return p.parse_args()


def load_coco_image_caption_pairs(coco_json_path):
    """
    Returns:
      captions: List[str]               (one per annotation)
      img_indices: np.ndarray[int64]    (same length as captions; maps annotation -> image row index)
      img_names: List[str]              (length = num_images_with_captions; row index -> image file name)
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not (isinstance(data, dict) and "images" in data and "annotations" in data):
        raise ValueError("Expected COCO format JSON with keys: 'images' and 'annotations'.")

    imgid_to_name = {}
    for im in data["images"]:
        if "id" in im and "file_name" in im and isinstance(im["file_name"], str):
            imgid_to_name[int(im["id"])] = im["file_name"]

    captions = []
    ann_imgids = []
    for a in data["annotations"]:
        cap = a.get("caption", "")
        imgid = a.get("image_id", None)
        if imgid is None:
            continue
        if not isinstance(cap, str):
            continue
        cap = cap.strip()
        if not cap:
            continue
        imgid = int(imgid)
        if imgid not in imgid_to_name:
            continue
        captions.append(cap)
        ann_imgids.append(imgid)

    if len(captions) == 0:
        raise ValueError("No valid (image_id, caption) pairs found in COCO JSON.")

    used_imgids = sorted(set(ann_imgids))
    imgid_to_idx = {imgid: i for i, imgid in enumerate(used_imgids)}
    img_names = [imgid_to_name[i] for i in used_imgids]
    img_indices = np.array([imgid_to_idx[i] for i in ann_imgids], dtype=np.int64)

    return captions, img_indices, img_names


def mean_pool_tokens(h, attention_mask):
    m = attention_mask.unsqueeze(-1).to(h.dtype)
    denom = m.sum(dim=1).clamp(min=1.0)
    return (h * m).sum(dim=1) / denom


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def get_transformer_layers(model):
    candidate_paths = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        ("text_model", "layers"),
        ("model", "text_model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("backbone", "layers"),
    ]

    for path in candidate_paths:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, (list, nn.ModuleList)) and len(obj) > 0:
            return obj

    best = None
    best_score = -1
    for _name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            first = mod[0]
            looks_like_block = any(
                hasattr(first, a) for a in ["self_attn", "attn", "attention", "mlp", "feed_forward"]
            )
            score = (1000 if looks_like_block else 0) + len(mod)
            if score > best_score:
                best_score = score
                best = mod

    if best is not None:
        return best

    raise RuntimeError("Could not locate transformer layers ModuleList on this model.")


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.captions_path):
        raise FileNotFoundError(f"Captions JSON not found: {args.captions_path}")

    print(f"Loading COCO captions+images from {args.captions_path}...")
    captions, img_indices, img_names = load_coco_image_caption_pairs(args.captions_path)
    n_caps = len(captions)
    n_imgs = len(img_names)
    print(f"Loaded {n_caps} captions across {n_imgs} images.")

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.hf_cache_dir,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"

    dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_available()) else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.hf_cache_dir,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    layers = get_transformer_layers(model)
    n_layers = len(layers)
    mid_block_idx = n_layers // 2

    print(f"n_layers={n_layers} | using middle block index={mid_block_idx}")

    captured = {"mid": None}

    def hook_mid(_module, _inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        captured["mid"] = h

    h_mid = layers[mid_block_idx].register_forward_hook(hook_mid)

    sum_embs = None
    counts = np.zeros((n_imgs,), dtype=np.int32)

    print("Embedding captions (middle layer) and accumulating per-image means...")
    with torch.inference_mode():
        for start in range(0, n_caps, args.batch_size):
            end = min(start + args.batch_size, n_caps)
            batch_caps = captions[start:end]
            batch_img_idx = img_indices[start:end]

            batch = tokenizer(
                batch_caps,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            batch = {k: v.to(device) for k, v in batch.items()}

            captured["mid"] = None
            _ = model(**batch, use_cache=False)

            if captured["mid"] is None:
                raise RuntimeError("Hook did not capture middle-layer outputs.")

            attn = batch["attention_mask"]
            cap_vecs = mean_pool_tokens(captured["mid"], attn).detach().float().cpu().numpy()

            if sum_embs is None:
                D = cap_vecs.shape[1]
                sum_embs = np.zeros((n_imgs, D), dtype=np.float32)

            np.add.at(sum_embs, batch_img_idx, cap_vecs)
            np.add.at(counts, batch_img_idx, 1)

            if start % 5000 == 0:
                print(f"Processed {end}/{n_caps}")

    h_mid.remove()

    if sum_embs is None:
        raise RuntimeError("No embeddings were produced (unexpected).")

    denom = np.maximum(counts[:, None], 1)
    img_text_embs = sum_embs / denom
    img_text_embs = l2_normalize_rows(img_text_embs)

    img_names_arr = np.array(img_names, dtype=object)
    order = np.argsort(img_names_arr, kind="mergesort")
    img_names_sorted = img_names_arr[order].tolist()
    img_text_embs_sorted = img_text_embs[order]

    np.save(os.path.join(args.output_dir, "text_embeddings.npy"), img_text_embs_sorted)
    with open(os.path.join(args.output_dir, "img_names.txt"), "w", encoding="utf-8") as f:
        for name in img_names_sorted:
            f.write(f"{name}\n")

    print(f"Saved: {os.path.join(args.output_dir, 'text_embeddings.npy')}")
    print(f"Saved: {os.path.join(args.output_dir, 'img_names.txt')}")
    print("Done.")


if __name__ == "__main__":
    main()
