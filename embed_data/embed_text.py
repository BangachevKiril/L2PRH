#!/usr/bin/env python3
import os
import json
import argparse

import numpy as np
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed COCO captions JSON with a text embedding model (mean-pool per image)."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model ID for embeddings.",
    )
    parser.add_argument(
        "--captions_json",
        type=str,
        required=True,
        help="Path to COCO-style captions JSON (e.g. captions_train2017.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs: text_embeddings.npy and img_names.txt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for embedding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device for SentenceTransformer, e.g. "cuda", "cuda:0", or "cpu". '
             'If omitted, SentenceTransformer will auto-detect.',
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory (passed to SentenceTransformer).",
    )
    return parser.parse_args()


def load_coco_captions_and_names(json_path):
    """
    Returns:
      captions: List[str] length N_captions
      img_ids:  np.ndarray shape (N_captions,) of int image_ids aligned with captions
      imgid_to_name: dict[int,str] mapping COCO image id -> file_name
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "annotations" not in data:
        raise ValueError(
            f"{json_path} does not look like COCO captions JSON (missing 'annotations')."
        )
    if "images" not in data:
        raise ValueError(
            f"{json_path} does not look like COCO captions JSON (missing 'images')."
        )

    imgid_to_name = {}
    for im in data["images"]:
        img_id = im.get("id")
        fn = im.get("file_name")
        if img_id is None or fn is None:
            continue
        imgid_to_name[int(img_id)] = str(fn)

    captions = []
    img_ids = []

    for ann in data["annotations"]:
        cap = ann.get("caption", "")
        if not isinstance(cap, str):
            continue
        cap = cap.strip()
        if not cap:
            continue

        img_id = ann.get("image_id", None)
        if img_id is None:
            continue

        captions.append(cap)
        img_ids.append(int(img_id))

    return captions, np.asarray(img_ids, dtype=np.int64), imgid_to_name


def _is_stella_model_name(model_name: str) -> bool:
    """
    Heuristic: treat anything with 'stella' in the HF repo id / local path as a Stella model.
    """
    return "stella" in (model_name or "").lower()


def build_model(model_name, device, cache_dir):
    """
    Build SentenceTransformer, with a special case for Stella models:
      - use config_kwargs={"use_memory_efficient_attention": False} to bypass xformers check
      - default device to "cuda" if not explicitly provided (matches your requested pattern)
    """
    # trust_remote_code=True needed for Nomic & often safe for others
    kwargs = {"trust_remote_code": True}

    # Optional HF cache dir (SentenceTransformer uses cache_folder kwarg)
    if cache_dir is not None:
        kwargs["cache_folder"] = cache_dir

    if _is_stella_model_name(model_name):
        # Stella-specific: bypass xformers check
        kwargs["config_kwargs"] = {"use_memory_efficient_attention": False}

        # If user didn't specify a device, follow your desired behavior (cuda by default).
        # If they DID specify, respect it (cpu/cuda/cuda:0/etc.).
        if device is None:
            kwargs["device"] = "cuda"
        else:
            kwargs["device"] = device
    else:
        # Non-stella: keep existing behavior
        if device is not None:
            kwargs["device"] = device

    return SentenceTransformer(model_name, **kwargs)


def embed_texts(model, model_name, texts, batch_size):
    """
    Handle model-specific prompting / API quirks.
    """
    if "nomic-ai/nomic-embed-text-v2-moe" in model_name:
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            prompt_name="passage",
        )

    if "nomic-ai/nomic-embed-text-v1.5" in model_name:
        prefixed = [f"clustering: {t}" for t in texts]
        return model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=True,
        )

    if "google/embeddinggemma-300m" in model_name:
        if hasattr(model, "encode_document"):
            return model.encode_document(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
            )
        doc_texts = [f'title: "none" | text: {t}' for t in texts]
        return model.encode(
            doc_texts,
            batch_size=batch_size,
            show_progress_bar=True,
        )

    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )


def mean_pool_by_image(embeddings, img_ids):
    """
    embeddings: (N_captions, D) float32
    img_ids:    (N_captions,) int64

    Returns:
      pooled:         (N_images, D) float32  mean over captions per image_id
      unique_img_ids: (N_images,) int64      image ids aligned with pooled
    """
    unique_img_ids, inv = np.unique(img_ids, return_inverse=True)
    n_imgs = unique_img_ids.shape[0]
    d = embeddings.shape[1]

    sums = np.zeros((n_imgs, d), dtype=np.float32)
    np.add.at(sums, inv, embeddings.astype(np.float32, copy=False))

    counts = np.bincount(inv).astype(np.float32)  # (n_imgs,)
    pooled = sums / (counts[:, None] + 1e-12)

    # L2 normalize to norm 1
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    pooled = pooled / (norms + 1e-12)

    return pooled.astype(np.float32, copy=False), unique_img_ids


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading COCO captions from {args.captions_json} ...")
    captions, img_ids, imgid_to_name = load_coco_captions_and_names(args.captions_json)
    print(f"Loaded {len(captions)} captions.")

    print(f"Loading embedding model: {args.model_name}")
    model = build_model(args.model_name, args.device, args.hf_cache_dir)

    print("Embedding captions...")
    cap_emb = embed_texts(
        model,
        args.model_name,
        captions,
        batch_size=args.batch_size,
    )
    cap_emb = np.asarray(cap_emb, dtype=np.float32)
    print(f"Caption embeddings shape: {cap_emb.shape}")

    print("Mean-pooling per image and normalizing...")
    pooled, unique_img_ids = mean_pool_by_image(cap_emb, img_ids)
    print(f"Pooled text embeddings shape (num_images, dim): {pooled.shape}")

    # Build image names aligned with pooled embeddings
    img_names = []
    missing = 0
    for iid in unique_img_ids.tolist():
        name = imgid_to_name.get(int(iid))
        if name is None:
            missing += 1
            name = str(iid)  # fallback, still sortable
        img_names.append(name)

    if missing:
        print(f"Warning: {missing} image_ids had no file_name in JSON 'images' list. Using image_id as name.")

    # Sort by lexicographic order of image names
    sort_idx = np.argsort(np.asarray(img_names, dtype=object), kind="mergesort")
    pooled = pooled[sort_idx]
    img_names = [img_names[i] for i in sort_idx]

    # Save exactly the two requested files
    out_emb = os.path.join(args.output_dir, "text_embeddings.npy")
    out_names = os.path.join(args.output_dir, "img_names.txt")

    np.save(out_emb, pooled.astype(np.float32, copy=False))
    with open(out_names, "w", encoding="utf-8") as f:
        for name in img_names:
            f.write(name + "\n")

    print("Saved:")
    print(f"  {out_emb}")
    print(f"  {out_names}")
    print("Done.")


if __name__ == "__main__":
    main()