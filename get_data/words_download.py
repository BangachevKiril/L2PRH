#!/usr/bin/env python3
"""
wordfreq_train_as_coco.py

Render the top-N most frequent English words (wordfreq) as a COCO captions dataset
with COCO-like folder structure:

  OUT_ROOT/
    images/train2017/                (empty; no images)
    annotations/captions_train2017.json
    annotations/top_words.csv        (optional)
    frequencies.npy                  (raw wordfreq frequencies, length N)
    noun_indices.txt                 (zero-based row indices)
    verb_indices.txt                 (zero-based row indices)
    adjective_indices.txt            (zero-based row indices)

Each word becomes the caption for a distinct dummy image entry.

The three *_indices.txt files contain one zero-based integer per line, so they are
compatible with scripts that load an index list and then subselect as arr[L].

POS labeling policy
-------------------
The default is intentionally conservative:
  * use NLTK's Penn POS tagger on each word as an isolated token;
  * if POS_INCLUDE_AMBIGUOUS=0, only the primary Penn tag is used;
  * closed-class tokens such as pronouns, determiners, prepositions, conjunctions,
    and particles are not rescued by WordNet, so words like "I", "this", and "as"
    do not enter noun/verb/adjective lists;
  * auxiliary verb forms such as "was" are tagged as verbs because their Penn tag is VBD/VBZ/etc.

This is designed for the POS-composition ablation, where the goal is clean noun,
verb, and adjective buckets rather than every possible dictionary sense of a word.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from wordfreq import top_n_list, word_frequency


# -----------------------------------------------------------------------------
# Original CLI + POS index options
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, required=True)

    p.add_argument("--N", type=int, default=50_000)
    p.add_argument("--lang", type=str, default="en")
    p.add_argument("--wordlist", type=str, default="best")

    # COCO-ish naming, defaults match the original script.
    p.add_argument("--split_dir", type=str, default="train2017")
    p.add_argument("--ann_name", type=str, default="captions_train2017.json")

    # Optional CSV manifest.
    p.add_argument("--write_csv", type=int, default=1)
    p.add_argument("--csv_name", type=str, default="top50k_words.csv")

    # IDs.
    p.add_argument("--start_id", type=int, default=1)

    # If you want extra metadata in captions, off by default.
    p.add_argument(
        "--include_freq_in_caption",
        type=int,
        default=0,
        help="If 1, caption becomes 'word\\t<raw_freq>\\t<normalized_freq>' (default: 0).",
    )

    # New POS-index files.
    p.add_argument(
        "--write_pos_indices",
        type=int,
        default=1,
        help="If 1, write noun_indices.txt, verb_indices.txt, adjective_indices.txt into out_root.",
    )
    p.add_argument("--noun_indices_name", type=str, default="noun_indices.txt")
    p.add_argument("--verb_indices_name", type=str, default="verb_indices.txt")
    p.add_argument("--adjective_indices_name", type=str, default="adjective_indices.txt")

    p.add_argument(
        "--pos_include_ambiguous",
        type=int,
        default=0,
        help=(
            "If 0, use only the primary isolated-token Penn POS tag. "
            "If 1, add WordNet noun/verb/adjective senses for open-class words only. "
            "Default 0 is cleaner for POS ablations."
        ),
    )
    p.add_argument(
        "--strict_nltk_tagger",
        type=int,
        default=0,
        help=(
            "If 1, fail if NLTK Penn POS tagging is unavailable. "
            "Recommended for clean experiments. If 0, use a conservative fallback heuristic."
        ),
    )
    p.add_argument(
        "--try_download_nltk_data",
        type=int,
        default=0,
        help=(
            "If 1, try nltk.download for the POS tagger data if missing. "
            "Default 0 avoids network attempts on clusters."
        ),
    )
    p.add_argument(
        "--print_pos_examples",
        type=int,
        default=1,
        help="If 1, print the first few noun/verb/adjective indices and words for sanity checking.",
    )

    return p.parse_args()


# -----------------------------------------------------------------------------
# Original dataset rendering
# -----------------------------------------------------------------------------

def ensure_dirs(out_root: str, split_dir: str):
    img_dir = os.path.join(out_root, "images", split_dir)
    ann_dir = os.path.join(out_root, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    return img_dir, ann_dir


def build_coco_json(words: List[str], freqs: List[float], rel_freqs: List[float], *, args) -> Dict[str, Any]:
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    images = []
    annotations = []

    img_id = args.start_id
    ann_id = args.start_id

    for w, f_raw, f_norm in zip(words, freqs, rel_freqs):
        # No images exist: keep placeholders, but preserve COCO structure.
        images.append(
            {
                "id": img_id,
                "file_name": "",  # intentionally blank
                "width": 0,
                "height": 0,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": "",
            }
        )

        if args.include_freq_in_caption:
            caption = f"{w}\t{f_raw:.10g}\t{f_norm:.10g}"
        else:
            caption = w

        annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "caption": caption,
            }
        )

        img_id += 1
        ann_id += 1

    return {
        "info": {
            "description": f"Top-{len(words)} wordfreq words as COCO captions (no images).",
            "version": "1.0",
            "year": datetime.utcnow().year,
            "date_created": now,
            "source": "wordfreq",
            "lang": args.lang,
            "wordlist": args.wordlist,
        },
        "licenses": [],
        "type": "captions",
        "images": images,
        "annotations": annotations,
    }


def write_csv(path: str, words: List[str], freqs: List[float], rel_freqs: List[float]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "word", "wordfreq_freq", "normalized_freq"])
        for i, (word, f_raw, f_norm) in enumerate(zip(words, freqs, rel_freqs), start=1):
            w.writerow([i, word, f_raw, f_norm])


# -----------------------------------------------------------------------------
# Conservative POS labeling
# -----------------------------------------------------------------------------

# Tokens that should not be reclassified as noun/verb/adjective based on odd
# dictionary senses. We preserve auxiliaries such as "was" by letting the POS
# tagger tag them VB*.
CLOSED_CLASS_WORDS = {
    # pronouns
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "who", "whom", "whose", "which", "what", "whatever",
    "whoever", "whomever", "something", "anything", "nothing", "everything", "someone",
    "anyone", "everyone", "noone", "nobody", "somebody", "anybody", "everybody",
    # determiners and quantifiers
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "no", "each",
    "every", "either", "neither", "both", "all", "many", "much", "few", "several", "such",
    # prepositions, conjunctions, particles, infinitive marker
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about", "as", "into", "like",
    "through", "after", "over", "between", "out", "against", "during", "without", "before",
    "under", "around", "among", "and", "or", "but", "if", "because", "while", "although",
    "though", "until", "unless", "than", "whether", "nor", "so", "yet", "up", "down", "off",
    # modals and adverbs that should not become verbs/adjectives in the clean bucket
    "can", "could", "may", "might", "must", "shall", "should", "will", "would", "not", "n't",
    "very", "too", "also", "just", "only", "even", "then", "there", "here", "when", "where", "why",
    "how",
}

# A small fallback list used only when the NLTK tagger is unavailable and strict_nltk_tagger=0.
COMMON_NOUNS = {
    "time", "year", "people", "way", "day", "man", "woman", "thing", "world", "life", "hand",
    "part", "child", "eye", "place", "work", "week", "case", "point", "government", "company",
    "number", "group", "problem", "fact", "home", "water", "room", "mother", "area", "money",
    "story", "issue", "side", "kind", "head", "house", "service", "friend", "father", "power",
}
COMMON_VERBS = {
    "be", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "say", "says", "said", "go", "goes", "went", "gone", "get",
    "gets", "got", "make", "makes", "made", "know", "knows", "knew", "think", "take", "takes",
    "took", "see", "saw", "come", "came", "want", "use", "used", "find", "give", "tell", "work",
    "call", "try", "ask", "need", "feel", "become", "leave", "put", "mean", "keep", "let", "begin",
}
COMMON_ADJECTIVES = {
    "good", "new", "first", "last", "long", "great", "little", "own", "other", "old", "right",
    "big", "high", "different", "small", "large", "next", "early", "young", "important", "bad",
    "same", "able", "best", "better", "local", "sure", "clear", "white", "black",
}

NOUN_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ism", "ist", "ship", "age", "ance", "ence", "hood",
    "dom", "ery", "ry", "ure", "acy", "logy", "graphy", "scope", "meter",
)
VERB_SUFFIXES = ("ate", "ify", "fy", "ize", "ise", "en")
ADJ_SUFFIXES = (
    "able", "ible", "al", "ial", "ical", "ic", "ive", "ous", "eous", "ious", "ful", "less", "ish",
    "ary", "ory", "ant", "ent",
)


def penn_to_coarse(tag: str) -> Optional[str]:
    tag = (tag or "").upper()
    if tag.startswith("NN"):
        return "noun"
    if tag.startswith("VB"):
        return "verb"
    if tag.startswith("JJ"):
        return "adjective"
    return None


def try_load_nltk_pos_tagger(try_download: bool = False):
    """Return (tag_one_word_function_or_None, status_message)."""
    try:
        import nltk  # type: ignore
    except Exception as e:
        return None, f"NLTK import failed: {type(e).__name__}: {e}"

    def _probe():
        # NLTK versions differ in which resource name they use internally.
        return nltk.pos_tag(["was"], lang="eng")

    try:
        _probe()
    except LookupError as e:
        if not try_download:
            return None, f"NLTK POS tagger data not found: {e}"
        for package in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass
        try:
            _probe()
        except Exception as e2:
            return None, f"NLTK POS tagger download/probe failed: {type(e2).__name__}: {e2}"
    except Exception as e:
        return None, f"NLTK POS tagger probe failed: {type(e).__name__}: {e}"

    def tag_one(word: str) -> str:
        tagged = nltk.pos_tag([word], lang="eng")
        return str(tagged[0][1])

    return tag_one, "NLTK Penn POS tagger available"


def try_load_wordnet(try_download: bool = False):
    """Return (wordnet_or_None, status_message). Used only when pos_include_ambiguous=1."""
    try:
        import nltk  # type: ignore
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception as e:
        return None, f"NLTK/WordNet import failed: {type(e).__name__}: {e}"

    def _probe():
        _ = wn.synsets("dog", pos=wn.NOUN)

    try:
        _probe()
    except LookupError as e:
        if not try_download:
            return None, f"NLTK WordNet data not found: {e}"
        for package in ("wordnet", "omw-1.4"):
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass
        try:
            _probe()
        except Exception as e2:
            return None, f"NLTK WordNet download/probe failed: {type(e2).__name__}: {e2}"
    except Exception as e:
        return None, f"NLTK WordNet probe failed: {type(e).__name__}: {e}"

    return wn, "NLTK WordNet available"


def wordnet_open_class_pos(word: str, wn) -> Set[str]:
    """Return possible noun/verb/adjective POS from WordNet."""
    x = word.strip().lower().replace(" ", "_")
    if not x:
        return set()

    out: Set[str] = set()
    pos_specs = [
        ("noun", wn.NOUN),
        ("verb", wn.VERB),
        ("adjective", wn.ADJ),
        ("adjective", wn.ADJ_SAT),
    ]
    for coarse, pos in pos_specs:
        synsets = list(wn.synsets(x, pos=pos))
        try:
            lemma = wn.morphy(x, pos=pos)
        except Exception:
            lemma = None
        if lemma and lemma != x:
            synsets.extend(list(wn.synsets(lemma, pos=pos)))
        if synsets:
            out.add(coarse)
    return out


def fallback_heuristic_pos(word: str) -> Set[str]:
    """Conservative fallback. Used only if the NLTK tagger is unavailable."""
    x = word.strip().lower().strip("'\".,;:!?()[]{}")
    if not x or not any(ch.isalpha() for ch in x):
        return set()
    if x in CLOSED_CLASS_WORDS:
        # Do not classify pronouns/determiners/prepositions/modal particles as open-class words.
        return set()

    out: Set[str] = set()
    if x in COMMON_NOUNS:
        out.add("noun")
    if x in COMMON_VERBS:
        out.add("verb")
    if x in COMMON_ADJECTIVES:
        out.add("adjective")

    if x.endswith(NOUN_SUFFIXES):
        out.add("noun")
    if x.endswith(VERB_SUFFIXES):
        out.add("verb")
    if x.endswith(ADJ_SUFFIXES):
        out.add("adjective")
    if len(x) > 4 and (x.endswith("ing") or x.endswith("ed")):
        out.add("verb")

    return out


def classify_words(words: Sequence[str], *, args) -> Tuple[List[Set[str]], List[str], str]:
    """Classify words as noun/verb/adjective using primary isolated Penn POS tags."""
    tag_one, tagger_status = try_load_nltk_pos_tagger(try_download=bool(args.try_download_nltk_data))
    if tag_one is None:
        if args.strict_nltk_tagger:
            raise RuntimeError(
                f"{tagger_status}\n"
                "Install/download NLTK POS data or run with --try_download_nltk_data 1. "
                "For a clean POS ablation, I recommend keeping --strict_nltk_tagger 1."
            )
        print(f"[warn] {tagger_status}")
        print("[warn] Falling back to conservative built-in POS heuristics. This is weaker than NLTK tagging.")
        pos_sets = [fallback_heuristic_pos(w) for w in words]
        tags = ["HEUR" if ps else "" for ps in pos_sets]
        return pos_sets, tags, "heuristic_fallback"

    print(f"[info] {tagger_status}")
    wn = None
    if args.pos_include_ambiguous:
        wn, wn_status = try_load_wordnet(try_download=bool(args.try_download_nltk_data))
        if wn is None:
            print(f"[warn] {wn_status}")
            print("[warn] POS_INCLUDE_AMBIGUOUS=1 but WordNet is unavailable; using primary Penn tags only.")
        else:
            print(f"[info] {wn_status}")

    pos_sets: List[Set[str]] = []
    penn_tags: List[str] = []

    for w in words:
        raw = str(w)
        x = raw.strip().lower()
        tag = tag_one(raw)
        penn_tags.append(tag)
        primary = penn_to_coarse(tag)

        if primary is None:
            # Closed-class and other non-open-class tokens stay out of the clean lists.
            # This is the crucial guard that prevents I/this/as from entering via WordNet.
            pos_sets.append(set())
            continue

        if not args.pos_include_ambiguous:
            pos_sets.append({primary})
            continue

        expanded = {primary}
        # Add dictionary senses only for words that are already open-class by Penn tag.
        # This keeps genuine lexical ambiguities such as "run", but avoids closed-class leakage.
        if wn is not None and x not in CLOSED_CLASS_WORDS:
            expanded |= wordnet_open_class_pos(raw, wn)
        pos_sets.append(expanded)

    return pos_sets, penn_tags, "nltk_penn"


def write_indices_txt(path: str, indices: Iterable[int]):
    # No header, one zero-based integer per line. This is friendly to np.loadtxt(dtype=int64).
    with open(path, "w", encoding="utf-8") as f:
        for idx in indices:
            f.write(f"{int(idx)}\n")


def _preview_indices(name: str, indices: List[int], words: Sequence[str], tags: Sequence[str], k: int = 12):
    preview = [(int(i), str(words[i]), str(tags[i])) for i in indices[:k]]
    print(f"[sanity] first {min(k, len(indices))} {name} entries: {preview}")


def write_pos_index_files(out_root: str, words: List[str], *, args):
    all_pos, penn_tags, backend = classify_words(words, args=args)

    noun_idx = [i for i, ps in enumerate(all_pos) if "noun" in ps]
    verb_idx = [i for i, ps in enumerate(all_pos) if "verb" in ps]
    adj_idx = [i for i, ps in enumerate(all_pos) if "adjective" in ps]

    noun_path = os.path.join(out_root, args.noun_indices_name)
    verb_path = os.path.join(out_root, args.verb_indices_name)
    adj_path = os.path.join(out_root, args.adjective_indices_name)

    write_indices_txt(noun_path, noun_idx)
    write_indices_txt(verb_path, verb_idx)
    write_indices_txt(adj_path, adj_idx)

    print(f"[info] POS backend used: {backend}")
    print(f"[info] POS_INCLUDE_AMBIGUOUS={int(args.pos_include_ambiguous)}")
    print(f"[done] wrote noun indices:      {noun_path} ({len(noun_idx):,} indices)")
    print(f"[done] wrote verb indices:      {verb_path} ({len(verb_idx):,} indices)")
    print(f"[done] wrote adjective indices: {adj_path} ({len(adj_idx):,} indices)")

    if args.print_pos_examples:
        _preview_indices("noun", noun_idx, words, penn_tags)
        _preview_indices("verb", verb_idx, words, penn_tags)
        _preview_indices("adjective", adj_idx, words, penn_tags)

    # Explicit sanity check for the problematic early words the user mentioned.
    check_words = {"i", "this", "was", "as"}
    found = []
    for i, w in enumerate(words[:100]):
        if str(w).strip().lower() in check_words:
            found.append((i, str(w), penn_tags[i], sorted(all_pos[i])))
    if found:
        print(f"[sanity] selected early closed-class/auxiliary words: {found}")

    if len(noun_idx) == 0 or len(verb_idx) == 0 or len(adj_idx) == 0:
        print("[warn] At least one POS index file is empty. Check NLTK availability and the word list.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    img_dir, ann_dir = ensure_dirs(args.out_root, args.split_dir)

    ann_path = os.path.join(ann_dir, args.ann_name)
    csv_path = os.path.join(ann_dir, args.csv_name)
    freq_path = os.path.join(args.out_root, "frequencies.npy")

    print(f"[info] out_root: {args.out_root}")
    print(f"[info] images dir (empty by design): {img_dir}")
    print(f"[info] annotations: {ann_path}")
    print(f"[info] frequencies: {freq_path}")
    print(f"[info] N={args.N} lang={args.lang} wordlist={args.wordlist}")

    words = top_n_list(args.lang, args.N, wordlist=args.wordlist)
    freqs = [word_frequency(w, args.lang) for w in words]
    total = sum(freqs)
    if total == 0:
        raise ValueError("Sum of frequencies is zero; something is wrong with wordfreq results.")
    rel_freqs = [f / total for f in freqs]

    coco = build_coco_json(words, freqs, rel_freqs, args=args)

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    if args.write_csv:
        write_csv(csv_path, words, freqs, rel_freqs)
        print(f"[info] wrote csv: {csv_path}")

    # Save raw freqs as .npy (float32 for compactness), exactly as before.
    np.save(freq_path, np.asarray(freqs, dtype=np.float32))

    if args.write_pos_indices:
        write_pos_index_files(args.out_root, words, args=args)
    else:
        print("[info] --write_pos_indices=0, skipping POS index files.")

    print(f"[done] images entries: {len(coco['images']):,}")
    print(f"[done] annotations entries: {len(coco['annotations']):,}")
    print(f"[done] saved frequencies: {freq_path} (shape=({len(freqs)},), dtype=float32)")


if __name__ == "__main__":
    main()
