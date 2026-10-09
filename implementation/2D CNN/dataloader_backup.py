"""
DataLoader za DoTA dataset – next-frame predviđanje opasnosti.

Nova strategija:
    Ulaz:  24 framea + bounding box anotacije (pozicija, kategorija, brzina objekata)
    Izlaz: danger score za frame 25

    Za svaki frame u sekvenci učitavaju se:
    - RGB slika (za CNN)
    - Bounding boxovi objekata (za ROI Align)
    - Numeričke značajke bbox-ova (pozicija, veličina, brzina, kategorija)

    Bounding box značajke po objektu (17 dimenzija):
    [cx, cy, w, h, area, aspect_ratio, dx, dy, dw, dh, conf, approach_rate, expansion_rate, ttc,
     flow_mag, flow_angle, flow_std]
    - cx, cy:   normalizirana pozicija centra [0, 1]
    - w, h:     normalizirana veličina [0, 1]
    - area:     normalizirana površina bbox-a
    - aspect:   omjer širine i visine
    - dx, dy:   brzina centra (razlika između frameova, iz obj_track_id)
    - dw, dh:   promjena veličine (približavanje = bbox raste)
    - conf:     YOLO detection confidence (1.0 za DoTA anotacije)
    - approach_rate: brzina približavanja centru slike (negativno = približava se)
    - expansion_rate: relativna brzina rasta bbox-a (dw/w + dh/h)
    - ttc:      Time-to-Collision proxy = area / max(expansion_rate, eps)
    - flow_mag, flow_angle, flow_std: Optical flow statistike unutar bbox-a

    Kategorije objekata: car, truck, bus, person, rider, bike, motor
"""

import os
import sys
import glob
import json
import random
from functools import lru_cache
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import utility modula iz utils paketa
from utils import (
    CATEGORY_MAP, NUM_CATEGORIES, IMAGENET_MEAN, IMAGENET_STD,
    BBOX_FEATURE_DIM, FRAME_CACHE_MAX_SIZE, CNN_CACHE_MAX_SIZE,
    INV_255, extract_bbox_features, apply_horizontal_flip_to_bbox,
    AugmentationParams, apply_temporal_jitter, apply_color_jitter,
    normalize_image, create_random_erasing_transform,
    apply_horizontal_flip_to_flow, sample_temporal_mask_indices,
    process_flow_sequence, extract_per_object_flow_patches, add_flow_features_to_bbox
)


# ── LRU keš za frame čitanje (stride=1 → 23/24 frameova dijeljeno) ── #
@lru_cache(maxsize=2048)
def _cached_imread(path: str) -> Optional[np.ndarray]:
    """Učitaj sliku s diska i konvertiraj u RGB (keširano)."""
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else img


# Konstante za numpy normalizaciju (izbjegava re-kreaciju u __getitem__)
_NORM_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_NORM_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)


# LRU keš za CNN cache fajlove (per-video .pt)
@lru_cache(maxsize=CNN_CACHE_MAX_SIZE)
def _cached_load_cnn_cache(path: str) -> Dict:
    """Učitaj pre-extracted CNN features za jedan video (keširano)."""
    return torch.load(path, map_location="cpu", weights_only=True)


class DoTANextFrameDataset(Dataset):
    """
    Dataset za next-frame predviđanje opasnosti.

    Svaki uzorak sadrži 24 framea konteksta. Target je danger score
    za frame 25 (sljedeći frame nakon sekvence).

    Args:
        frames_root:     putanja do dataset/frames
        annotations_dir: putanja do dataset/annotations (JSON s bboxovima)
        metadata:        dict iz metadata_*.json
        video_list:      lista video naziva
        seq_len:         broj frameova konteksta (input)
        img_size:        (H, W) resize dimenzija
        stride:          korak kliznog prozora
        danger_horizon:  frameova rampe prije anomaly_start
        max_objects:     max objekata po frameu
        orig_img_size:   originalna veličina slike (za normalizaciju bbox-a)
        phase:           'train' | 'val' | 'test'
        augment:         koristiti augmentaciju
    """

    def __init__(
        self,
        frames_root,
        annotations_dir,
        metadata,
        video_list,
        seq_len=24,
        img_size=(128, 128),
        stride=12,
        stride_safe=None,
        stride_danger=None,
        danger_horizon=10,
        max_objects=10,
        orig_img_size=(1280, 720),
        phase="train",
        augment=True,
        yolo_detections_dir=None,
        temporal_mask_prob=0.0,
        optical_flow_dir=None,
        use_per_object_flow=False,
        obj_flow_pool_size=5,
        cnn_cache_dir=None,
        no_ramp=False,
    ):
        self.frames_root = frames_root
        self.annotations_dir = annotations_dir
        self.yolo_detections_dir = yolo_detections_dir
        self.optical_flow_dir = optical_flow_dir
        self.use_per_object_flow = use_per_object_flow
        self.obj_flow_pool_size = obj_flow_pool_size
        self.cnn_cache_dir = cnn_cache_dir
        self.seq_len = seq_len
        self.img_size = img_size
        self.max_objects = max_objects
        self.orig_img_size = orig_img_size
        self.danger_horizon = danger_horizon
        self.phase = phase
        self.no_ramp = no_ramp
        self.temporal_mask_prob = temporal_mask_prob if phase == "train" else 0.0

        # Hybrid stride: stride_safe za sigurne zone, stride_danger za opasne
        self.stride_safe = stride_safe or stride
        self.stride_danger = stride_danger or stride

        self.video_frames = {}   # video_name → [frame_paths]
        self.video_anno = {}     # video_name → parsed annotation
        self.samples = []        # (video_name, start_idx, target_idx)

        use_hybrid = (self.stride_safe != self.stride_danger) and phase == "train"
        print(f"\n{'=' * 60}")
        print(f"[{phase.upper()}] Next-frame dataset – CNN+LSTM+BBox")
        if use_hybrid:
            print(f"  seq_len={seq_len}  stride_safe={self.stride_safe}  stride_danger={self.stride_danger}  max_obj={max_objects}")
        else:
            print(f"  seq_len={seq_len}  stride={stride}  max_obj={max_objects}")
        print(f"{'=' * 60}")

        if use_hybrid:
            self._build_samples_hybrid(metadata, video_list)
        else:
            self._build_samples(metadata, video_list, stride)

        # Validiraj CNN cache: moraju postojati SVE video datoteke
        if self.cnn_cache_dir:
            if not os.path.isdir(self.cnn_cache_dir):
                print(f"  UPOZORENJE: CNN cache dir '{self.cnn_cache_dir}' ne postoji!")
                self.cnn_cache_dir = None
            else:
                missing = [v for v in self.video_frames
                           if not os.path.isfile(os.path.join(self.cnn_cache_dir, f"{v}.pt"))]
                if missing:
                    print(f"  UPOZORENJE: {len(missing)}/{len(self.video_frames)} videa nema CNN cache!")
                    self.cnn_cache_dir = None
                else:
                    print(f"  CNN cache: {len(self.video_frames)} videa iz {self.cnn_cache_dir}/")

        self.transform = self._build_transform(augment and phase == "train")

        # Keš za transformirane frame tensore (samo val/test — deterministički transform)
        # OrderedDict za LRU ponašanje; max _FRAME_CACHE_MAX unosa
        self._frame_tensor_cache = OrderedDict() if phase != "train" else None

        self._print_stats()

    # ------------------------------------------------------------------ #
    #  Parsiranje anotacija                                                #
    # ------------------------------------------------------------------ #

    def _load_annotation(self, video_name):
        """Učitaj bbox anotaciju — prioritet: YOLO detekcije, fallback: DoTA anotacije."""
        # Prioritet: YOLO detekcije
        if self.yolo_detections_dir:
            yolo = self._load_yolo_detections(video_name)
            if yolo is not None:
                return yolo

        # Fallback: DoTA anotacije
        anno_path = os.path.join(self.annotations_dir, f"{video_name}.json")
        if not os.path.isfile(anno_path):
            return None

        with open(anno_path, encoding="utf-8") as f:
            data = json.load(f)

        frames = {}
        for label in data.get("labels", []):
            fid = label["frame_id"]
            objs = []
            for obj in label.get("objects", []):
                cat_name = obj.get("category", "")
                objs.append({
                    "track_id": obj.get("obj_track_id", -1),
                    "bbox": obj["bbox"],  # [x1, y1, x2, y2] u originalnim pixelima
                    "category": CATEGORY_MAP.get(cat_name, 0),
                })
            frames[fid] = {
                "objects": objs,
                "accident_id": label.get("accident_id", 0),
            }

        return {
            "frames": frames,
            "anomaly_start": data.get("anomaly_start", 9999),
            "anomaly_end": data.get("anomaly_end", 9999),
        }

    def _load_yolo_detections(self, video_name):
        """Učitaj YOLO detekcije za video (svi frameovi imaju objekte)."""
        yolo_path = os.path.join(self.yolo_detections_dir, f"{video_name}.json")
        if not os.path.isfile(yolo_path):
            return None

        with open(yolo_path, encoding="utf-8") as f:
            data = json.load(f)

        frames = {}
        for fid_str, objects in data.get("detections", {}).items():
            fid = int(fid_str)
            objs = []
            for obj in objects:
                objs.append({
                    "track_id": obj.get("track_id", -1),
                    "bbox": obj["bbox"],
                    "category": obj.get("category", 0),
                    "confidence": obj.get("confidence", 1.0),
                })
            frames[fid] = {
                "objects": objs,
                "accident_id": 0,
            }

        # YOLO nema anomaly info — to dolazi iz metadata
        return {
            "frames": frames,
            "anomaly_start": 9999,  # Popunjava se iz metadata u _build_samples
            "anomaly_end": 9999,
        }

    # ------------------------------------------------------------------ #
    #  Generiranje uzoraka                                                 #
    # ------------------------------------------------------------------ #

    def _build_samples(self, metadata, video_list, stride):
        skipped = 0

        for video_name in tqdm(video_list, desc=f"  {self.phase}"):
            if video_name not in metadata:
                skipped += 1
                continue

            meta = metadata[video_name]

            # Pronađi frameove
            video_dir = os.path.join(self.frames_root, video_name, "images")
            if not os.path.isdir(video_dir):
                video_dir = os.path.join(self.frames_root, video_name)
                if not os.path.isdir(video_dir):
                    skipped += 1
                    continue

            frame_files = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
            if not frame_files:
                frame_files = sorted(glob.glob(os.path.join(video_dir, "*.png")))
            n_frames = len(frame_files)

            # Treba barem seq_len+1 frameova (24 ulaz + 1 target)
            if n_frames < self.seq_len + 1:
                skipped += 1
                continue

            self.video_frames[video_name] = frame_files

            # Učitaj bbox anotacije (YOLO ili DoTA)
            anno = self._load_annotation(video_name)
            if anno is None:
                anno = {
                    "frames": {},
                    "anomaly_start": meta["anomaly_start"],
                    "anomaly_end": meta["anomaly_end"],
                }
            # Anomaly info uvijek iz metadata (YOLO ih nema)
            anno["anomaly_start"] = meta["anomaly_start"]
            anno["anomaly_end"] = meta["anomaly_end"]
            self.video_anno[video_name] = anno

            # Klizni prozor: ulaz = [start..start+seq_len-1], target = start+seq_len
            for start in range(0, n_frames - self.seq_len, stride):
                target_idx = start + self.seq_len
                if target_idx < n_frames:
                    self.samples.append((video_name, start, target_idx))

        if self.phase == "train":
            random.shuffle(self.samples)

        print(f"  Preskočeno videa: {skipped}")

    def _build_samples_hybrid(self, metadata, video_list):
        """Hybrid stride: gusti uzorci oko anomalije, rijetki u sigurnoj zoni."""
        skipped = 0
        stride_s = self.stride_safe
        stride_d = self.stride_danger
        dh = self.danger_horizon

        for video_name in tqdm(video_list, desc=f"  {self.phase}"):
            if video_name not in metadata:
                skipped += 1
                continue

            meta = metadata[video_name]
            video_dir = os.path.join(self.frames_root, video_name, "images")
            if not os.path.isdir(video_dir):
                video_dir = os.path.join(self.frames_root, video_name)
                if not os.path.isdir(video_dir):
                    skipped += 1
                    continue

            frame_files = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
            if not frame_files:
                frame_files = sorted(glob.glob(os.path.join(video_dir, "*.png")))
            n_frames = len(frame_files)

            if n_frames < self.seq_len + 1:
                skipped += 1
                continue

            self.video_frames[video_name] = frame_files

            anno = self._load_annotation(video_name)
            if anno is None:
                anno = {
                    "frames": {},
                    "anomaly_start": meta["anomaly_start"],
                    "anomaly_end": meta["anomaly_end"],
                }
            anno["anomaly_start"] = meta["anomaly_start"]
            anno["anomaly_end"] = meta["anomaly_end"]
            self.video_anno[video_name] = anno

            a_start = meta["anomaly_start"]
            a_end = meta["anomaly_end"]

            # Danger zona: [anomaly_start - danger_horizon, anomaly_end]
            danger_zone_start = max(0, a_start - dh)
            danger_zone_end = min(n_frames - 1, a_end)

            # Set za brzo provjeru duplikata
            added_targets = set()

            # 1) Gusti uzorci oko anomalije — anchored od anomaly_start
            #    Unatrag od anomaly_start
            t = a_start
            while t >= danger_zone_start:
                start = t - self.seq_len
                if start >= 0 and t < n_frames:
                    added_targets.add(t)
                    self.samples.append((video_name, start, t))
                t -= stride_d
            #    Unaprijed od anomaly_start
            t = a_start + stride_d
            while t <= danger_zone_end:
                start = t - self.seq_len
                if start >= 0 and t < n_frames:
                    added_targets.add(t)
                    self.samples.append((video_name, start, t))
                t += stride_d

            # 2) Rijetki uzorci u sigurnoj zoni — stride_safe
            for start in range(0, n_frames - self.seq_len, stride_s):
                target_idx = start + self.seq_len
                if target_idx < n_frames and target_idx not in added_targets:
                    # Provjeri da target NIJE u danger zoni (već pokriveno)
                    if target_idx < danger_zone_start or target_idx > danger_zone_end:
                        added_targets.add(target_idx)
                        self.samples.append((video_name, start, target_idx))

        if self.phase == "train":
            random.shuffle(self.samples)

        print(f"  Preskočeno videa: {skipped}")

    # ------------------------------------------------------------------ #
    #  Danger label za target frame                                        #
    # ------------------------------------------------------------------ #

    def _compute_danger(self, frame_idx, anomaly_start, anomaly_end):
        """
        Danger score za frame_idx.
        0.0 = sigurno, 0.x = rampa (linearna), 1.0 = anomalija
        Ako je no_ramp=True, vraća samo 0.0 ili 1.0 (binarno).
        """
        if anomaly_start <= frame_idx <= anomaly_end:
            return 1.0
        elif not self.no_ramp and frame_idx < anomaly_start:
            dist = anomaly_start - frame_idx
            if dist <= self.danger_horizon:
                return (self.danger_horizon - dist) / self.danger_horizon
        return 0.0

    # ------------------------------------------------------------------ #
    #  Transforms (position-preserving – ne mijenjaju bbox koordinate)    #
    # ------------------------------------------------------------------ #

    def _build_transform(self, augment):
        self._augment = augment
        if augment:
            # RandomErasing radi na tensorima — koristimo ga zasebno
            self._random_erasing = transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
        return None  # Augmentacija se radi ručno u numpy/torch za brzinu

    # ------------------------------------------------------------------ #
    #  Statistike                                                          #
    # ------------------------------------------------------------------ #

    def _print_stats(self):
        n_safe, n_ramp, n_danger = 0, 0, 0
        sample_n = min(500, len(self.samples))
        for i in range(sample_n):
            vname, _, target_idx = self.samples[i]
            anno = self.video_anno[vname]
            d = self._compute_danger(target_idx, anno["anomaly_start"], anno["anomaly_end"])
            if d == 0.0:
                n_safe += 1
            elif d < 1.0:
                n_ramp += 1
            else:
                n_danger += 1

        total = max(n_safe + n_ramp + n_danger, 1)
        print(f"\n  Ukupno uzoraka:   {len(self.samples)}")
        print(f"  Target distribucija (uzorak {sample_n}):")
        print(f"    Sigurno (0.0):  {n_safe:5d}  ({n_safe / total * 100:.1f}%)")
        print(f"    Rampa (0-1):    {n_ramp:5d}  ({n_ramp / total * 100:.1f}%)")
        print(f"    Opasno (1.0):   {n_danger:5d}  ({n_danger / total * 100:.1f}%)")

    # ------------------------------------------------------------------ #
    #  __getitem__                                                         #
    # ------------------------------------------------------------------ #

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_name, start_idx, target_idx = self.samples[idx]
        frame_files = self.video_frames[video_name]
        anno = self.video_anno[video_name]
        orig_w, orig_h = self.orig_img_size

        # Temporal jitter: nasumičan pomak prozora ±3 framera (samo trening)
        # Daje drugačiji "pogled" na istu scenu → povećava efektivni dataset
        if self._augment:
            n_frames = len(frame_files)
            lo = -min(3, start_idx)
            hi = min(3, n_frames - self.seq_len - 1 - start_idx)
            if lo < hi:
                j = random.randint(lo, hi)
                start_idx += j
                target_idx = start_idx + self.seq_len

        # --- CNN Cache: učitaj pre-extracted features ako postoje ---
        use_cnn_cache = False
        cached_feat_maps = None
        cached_global_feats = None
        if self.cnn_cache_dir:
            cache_path = os.path.join(self.cnn_cache_dir, f"{video_name}.pt")
            if os.path.isfile(cache_path):
                cache_data = _cached_load_cnn_cache(cache_path)
                cached_feat_maps = cache_data["feat_maps"][start_idx:start_idx + self.seq_len].clone()  # (T, 256, 14, 14) f16
                cached_global_feats = cache_data["global_feats"][start_idx:start_idx + self.seq_len].clone()  # (T, 512) f16
                use_cnn_cache = True

        frames = []
        all_bboxes = []
        all_categories = []
        all_features = []
        all_masks = []

        prev_objects_by_id = {}   # track_id → {cx, cy, w, h} za velocity (DoTA)
        prev_objects_list = []    # lista objekata za NN matching (YOLO)

        _cache = self._frame_tensor_cache
        _use_cache = _cache is not None
        inv_w = 1.0 / orig_w
        inv_h = 1.0 / orig_h

        # --- Uzorkuj augmentacijske parametre jednom za cijelu sekvencu ---
        # Konzistentnost kroz frameove sprječava lažne temporalne varijacije
        if self._augment:
            _aug_bright = 1.0 + random.uniform(-0.4, 0.4)
            _aug_contrast = 1.0 + random.uniform(-0.4, 0.4)
            _aug_sat = 1.0 + random.uniform(-0.3, 0.3)
            _aug_do_gray = random.random() < 0.05  # 5% šansa za grayscale
        else:
            _aug_bright = _aug_contrast = _aug_sat = 1.0
            _aug_do_gray = False

        for t in range(start_idx, start_idx + self.seq_len):
            # --- Učitaj frame (preskoči kad koristimo CNN cache) ---
            if not use_cnn_cache:
                _fpath = frame_files[t]
                if _use_cache and _fpath in _cache:
                    _cache.move_to_end(_fpath)  # LRU: ažuriraj pristup
                    img = _cache[_fpath]
                else:
                    img_rgb = _cached_imread(_fpath)  # vraća RGB (keširano)
                    if img_rgb.shape[:2] != (self.img_size[0], self.img_size[1]):
                        img_rgb = cv2.resize(img_rgb, (self.img_size[1], self.img_size[0]),
                                             interpolation=cv2.INTER_AREA)
                    # Brzi numpy→tensor put (izbjegava PIL overhead)
                    img_f = img_rgb.transpose(2, 0, 1).astype(np.float32) * _INV_255

                    if self._augment:
                        # Color jitter s konzistentnim parametrima kroz cijelu sekvencu
                        img_f = img_f * np.float32(_aug_bright)
                        mean_val = img_f.mean()
                        img_f = np.float32(_aug_contrast) * (img_f - mean_val) + mean_val
                        gray = img_f[0:1] * 0.299 + img_f[1:2] * 0.587 + img_f[2:3] * 0.114
                        img_f = np.float32(_aug_sat) * (img_f - gray) + gray
                        if _aug_do_gray:
                            img_f[:] = gray  # grayscale konverzija
                        np.clip(img_f, 0.0, 1.0, out=img_f)

                    # Normalize
                    img_f = (img_f - _NORM_MEAN) / _NORM_STD
                    img = torch.from_numpy(np.ascontiguousarray(img_f))

                    if self._augment:
                        img = self._random_erasing(img)

                    if _use_cache:
                        if len(_cache) >= _FRAME_CACHE_MAX:
                            _cache.popitem(last=False)  # LRU: izbaci najstariji
                        _cache[_fpath] = img
                frames.append(img)

            # --- Dohvati objekte za ovaj frame ---
            frame_data = anno["frames"].get(t, {"objects": [], "accident_id": 0})
            objects = frame_data["objects"]

            # numpy umjesto torch — izbjegava stotine malih tensor alokacija
            f_bboxes = np.zeros((self.max_objects, 4), dtype=np.float32)
            f_cats = np.zeros(self.max_objects, dtype=np.int64)
            f_feats = np.zeros((self.max_objects, 17), dtype=np.float32)
            f_mask = np.zeros(self.max_objects, dtype=np.bool_)

            curr_objects = []

            for i, obj in enumerate(objects[: self.max_objects]):
                x1, y1, x2, y2 = obj["bbox"]

                x1n = max(0.0, min(1.0, x1 * inv_w))
                y1n = max(0.0, min(1.0, y1 * inv_h))
                x2n = max(0.0, min(1.0, x2 * inv_w))
                y2n = max(0.0, min(1.0, y2 * inv_h))

                f_bboxes[i] = (x1n, y1n, x2n, y2n)
                f_cats[i] = obj["category"]
                f_mask[i] = True

                cx = (x1n + x2n) * 0.5
                cy = (y1n + y2n) * 0.5
                w = max(x2n - x1n, 1e-6)
                h = max(y2n - y1n, 1e-6)
                area = w * h
                aspect = w / h

                # Velocity: tracking po ID-u, ili NN matching po poziciji
                dx = dy = dw = dh = 0.0
                tid = obj["track_id"]
                if tid >= 0 and tid in prev_objects_by_id:
                    prev = prev_objects_by_id[tid]
                    dx = cx - prev["cx"]
                    dy = cy - prev["cy"]
                    dw = w - prev["w"]
                    dh = h - prev["h"]
                elif prev_objects_list:
                    cat = obj["category"]
                    best_d2 = 0.01  # 0.1² — max udaljenost²
                    best_prev = None
                    for p in prev_objects_list:
                        if p["cat"] != cat or p["matched"]:
                            continue
                        d2 = (cx - p["cx"]) ** 2 + (cy - p["cy"]) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best_prev = p
                    if best_prev is not None:
                        best_prev["matched"] = True
                        dx = cx - best_prev["cx"]
                        dy = cy - best_prev["cy"]
                        dw = w - best_prev["w"]
                        dh = h - best_prev["h"]

                conf = obj.get("confidence", 1.0)
                offset_x = cx - 0.5
                offset_y = cy - 0.5
                approach_rate = -(dx * offset_x + dy * offset_y) / max((offset_x**2 + offset_y**2)**0.5, 1e-6)
                expansion_rate = dw / max(w, 1e-6) + dh / max(h, 1e-6)
                ttc = min(area / expansion_rate, 5.0) if expansion_rate > 1e-4 else 5.0

                f_feats[i, :14] = (
                    cx, cy, w, h, area, aspect, dx, dy, dw, dh,
                    conf, approach_rate, expansion_rate, ttc,
                )
                curr_objects.append({
                    "cx": cx, "cy": cy, "w": w, "h": h,
                    "cat": obj["category"], "matched": False,
                    "track_id": tid,
                })

            # Pripremi za sljedeći frame
            prev_objects_list = curr_objects
            prev_objects_by_id = {
                o["track_id"]: o for o in curr_objects if o["track_id"] >= 0
            }

            all_bboxes.append(torch.from_numpy(f_bboxes))
            all_categories.append(torch.from_numpy(f_cats))
            all_features.append(torch.from_numpy(f_feats))
            all_masks.append(torch.from_numpy(f_mask))

        # --- Temporal augmentacija: nasumično zamijeni 1-2 framea duplikatom prethodnog ---
        temporal_masked = set()
        n_seq = len(frames) if not use_cnn_cache else self.seq_len
        if self.temporal_mask_prob > 0 and random.random() < self.temporal_mask_prob and n_seq > 2:
            n_mask = random.randint(1, 2)
            # Ne maskiraj prvi ni zadnji frame
            candidates = list(range(1, n_seq - 1))
            to_mask = random.sample(candidates, min(n_mask, len(candidates)))
            for m in to_mask:
                # Zamijeni s prethodnim frameom (ne crnom slikom!)
                if use_cnn_cache:
                    cached_feat_maps[m] = cached_feat_maps[m - 1].clone()
                    cached_global_feats[m] = cached_global_feats[m - 1].clone()
                else:
                    frames[m] = frames[m - 1].clone()
                all_bboxes[m] = all_bboxes[m - 1].clone()
                all_features[m] = all_features[m - 1].clone()
                all_categories[m] = all_categories[m - 1].clone()
                all_masks[m] = all_masks[m - 1].clone()
                temporal_masked.add(m)

        # --- Optical Flow: učitaj pre-izračunati flow ---
        all_flows = []
        all_obj_flows = []
        if self.optical_flow_dir:
            for t in range(start_idx, start_idx + self.seq_len):
                if t == start_idx:
                    # Prvi frame: nema prethodnog → nula flow
                    # Dimenzije doznajemo iz prvog postojećeg flow filea
                    flow = None
                else:
                    flow_path = os.path.join(
                        self.optical_flow_dir, video_name, f"flow_{t - 1:06d}.npy",
                    )
                    if os.path.isfile(flow_path):
                        flow = _cached_load_flow(flow_path)  # (2, H, W)
                    else:
                        flow = None
                all_flows.append(flow)

            # Kanonska veličina flow mape: img_size / 4 (npr. 224→56, 112→28)
            # MORA biti fiksna za sve uzorke u batchu — flow_hd ima miješane 56x56 i 112x112
            target_h = self.img_size[0] // 4
            target_w = self.img_size[1] // 4
            flow_shape = (2, target_h, target_w)

            # Zamijeni None s nulama; resizaj sve neskladne na kanonsku veličinu
            for i in range(len(all_flows)):
                if all_flows[i] is None or i in temporal_masked:
                    all_flows[i] = np.zeros(flow_shape, dtype=np.float32)
                else:
                    fl = all_flows[i]
                    if fl.shape[1] != target_h or fl.shape[2] != target_w:
                        all_flows[i] = np.stack([
                            cv2.resize(fl[0], (target_w, target_h), interpolation=cv2.INTER_AREA),
                            cv2.resize(fl[1], (target_w, target_h), interpolation=cv2.INTER_AREA),
                        ], axis=0)
                all_flows[i] = torch.from_numpy(all_flows[i])

            # --- Per-object flow features: crop flow unutar svakog bbox-a ---
            flow_h, flow_w = flow_shape[1], flow_shape[2]
            for i in range(len(all_flows)):
                flow_map = all_flows[i]  # (2, Hf, Wf)
                for j in range(self.max_objects):
                    if not all_masks[i][j]:
                        continue
                    # Bbox u [0,1] → flow pixel koordinate
                    x1f = int(all_bboxes[i][j, 0].item() * flow_w)
                    y1f = int(all_bboxes[i][j, 1].item() * flow_h)
                    x2f = max(x1f + 1, int(all_bboxes[i][j, 2].item() * flow_w))
                    y2f = max(y1f + 1, int(all_bboxes[i][j, 3].item() * flow_h))
                    x1f = max(0, min(x1f, flow_w - 1))
                    y1f = max(0, min(y1f, flow_h - 1))
                    x2f = max(x1f + 1, min(x2f, flow_w))
                    y2f = max(y1f + 1, min(y2f, flow_h))
                    crop = flow_map[:, y1f:y2f, x1f:x2f]  # (2, h, w)
                    dx_crop = crop[0]
                    dy_crop = crop[1]
                    mag = torch.sqrt(dx_crop**2 + dy_crop**2 + 1e-8)
                    all_features[i][j, 14] = mag.mean()               # flow_mag
                    all_features[i][j, 15] = torch.atan2(
                        dy_crop.mean(), dx_crop.mean(),
                    )                                                  # flow_angle
                    all_features[i][j, 16] = mag.std() if mag.numel() > 1 else 0.0  # flow_std

            # --- Residual flow: oduzmi ego-motion (medijan) ---
            # Konvertiraj u numpy za brže operacije u dataloaderu
            for i in range(len(all_flows)):
                flow_np = all_flows[i].numpy()  # (2, Hf, Wf)
                ego_dx = np.median(flow_np[0])
                ego_dy = np.median(flow_np[1])
                flow_np[0] -= ego_dx
                flow_np[1] -= ego_dy
                all_flows[i] = flow_np  # ostaje numpy do stackanja

            # --- Per-object flow patches: crop residual flow per bbox → fixed size ---
            all_obj_flows = []
            if self.use_per_object_flow:
                ps = self.obj_flow_pool_size
                for i in range(len(all_flows)):
                    flow_np = all_flows[i]  # (2, Hf, Wf) numpy
                    flow_h, flow_w = flow_np.shape[1], flow_np.shape[2]
                    obj_flow = np.zeros((self.max_objects, 2, ps, ps), dtype=np.float32)
                    for j in range(self.max_objects):
                        if not all_masks[i][j]:
                            continue
                        # Bbox [0,1] → flow pixel coordinates
                        x1f = int(all_bboxes[i][j, 0].item() * flow_w)
                        y1f = int(all_bboxes[i][j, 1].item() * flow_h)
                        x2f = max(x1f + 1, int(all_bboxes[i][j, 2].item() * flow_w))
                        y2f = max(y1f + 1, int(all_bboxes[i][j, 3].item() * flow_h))
                        x1f = max(0, min(x1f, flow_w - 1))
                        y1f = max(0, min(y1f, flow_h - 1))
                        x2f = max(x1f + 1, min(x2f, flow_w))
                        y2f = max(y1f + 1, min(y2f, flow_h))
                        # Crop i resize u numpy (cv2.resize je 10x brži od F.adaptive_avg_pool2d na CPU)
                        crop_dx = flow_np[0, y1f:y2f, x1f:x2f]
                        crop_dy = flow_np[1, y1f:y2f, x1f:x2f]
                        obj_flow[j, 0] = cv2.resize(crop_dx, (ps, ps), interpolation=cv2.INTER_AREA)
                        obj_flow[j, 1] = cv2.resize(crop_dy, (ps, ps), interpolation=cv2.INTER_AREA)
                    all_obj_flows.append(torch.from_numpy(obj_flow))

            # Konvertiraj flow natrag u torch za stackanje
            for i in range(len(all_flows)):
                if isinstance(all_flows[i], np.ndarray):
                    all_flows[i] = torch.from_numpy(all_flows[i])

        # --- Horizontal Flip augmentacija (konzistentna kroz cijelu sekvencu) ---
        if self.phase == "train" and random.random() < 0.5:
            if use_cnn_cache:
                cached_feat_maps = torch.flip(cached_feat_maps, dims=[-1])  # flip W feat mapa
            else:
                for i in range(len(frames)):
                    frames[i] = torch.flip(frames[i], dims=[-1])  # flip W dimenziju
            for i in range(len(all_bboxes)):
                x1_old = all_bboxes[i][:, 0].clone()
                x2_old = all_bboxes[i][:, 2].clone()
                all_bboxes[i][:, 0] = 1.0 - x2_old  # novi x1 = 1 - stari x2
                all_bboxes[i][:, 2] = 1.0 - x1_old  # novi x2 = 1 - stari x1
                # Flip cx i dx u bbox features
                # [cx, cy, w, h, area, aspect, dx, dy, dw, dh, conf, approach, expansion, ttc, flow_mag, flow_angle, flow_std]
                all_features[i][:, 0] = 1.0 - all_features[i][:, 0]  # cx
                all_features[i][:, 6] = -all_features[i][:, 6]       # dx (smjer se obrće)
                all_features[i][:, 11] = -all_features[i][:, 11]     # approach_rate (smjer se obrće)
                all_features[i][:, 15] = -all_features[i][:, 15]     # flow_angle (horizontalni smjer se obrće)
            # Flip optical flow: horizontalni flip + negiraj dx
            for i in range(len(all_flows)):
                all_flows[i] = torch.flip(all_flows[i], dims=[-1])  # flip W
                all_flows[i][0] = -all_flows[i][0]                  # negate dx
            # Flip per-object flow patches
            if all_obj_flows:
                for i in range(len(all_obj_flows)):
                    all_obj_flows[i] = torch.flip(all_obj_flows[i], dims=[-1])  # flip W
                    all_obj_flows[i][:, 0] = -all_obj_flows[i][:, 0]           # negate dx

        # Target: danger score za sljedeći frame (glavni) + per-frame targets
        target = self._compute_danger(
            target_idx, anno["anomaly_start"], anno["anomaly_end"]
        )

        # Per-frame targets: za svaki timestep t, danger score framea t+1
        # Frame t u sekvenci je na poziciji start_idx + t
        # Njegov "sljedeći frame" je start_idx + t + 1
        per_frame_targets = []
        for t in range(self.seq_len):
            next_frame_idx = start_idx + t + 1
            per_frame_targets.append(
                self._compute_danger(next_frame_idx, anno["anomaly_start"], anno["anomaly_end"])
            )

        result = {
            "bboxes": torch.stack(all_bboxes),         # (T, max_obj, 4)
            "bbox_features": torch.stack(all_features), # (T, max_obj, 17)
            "categories": torch.stack(all_categories),  # (T, max_obj)
            "obj_mask": torch.stack(all_masks),         # (T, max_obj)
            "target": torch.tensor(target, dtype=torch.float32),
            "per_frame_targets": torch.tensor(per_frame_targets, dtype=torch.float32),  # (T,)
            "video_name": video_name,
        }
        if use_cnn_cache:
            result["feat_maps"] = cached_feat_maps       # (T, 256, 14, 14) f16
            result["global_feats"] = cached_global_feats  # (T, 512) f16
        else:
            result["frames"] = torch.stack(frames)       # (T, 3, H, W)
        if all_flows:
            result["flow"] = torch.stack(all_flows)    # (T, 2, Hf, Wf)
        if all_obj_flows:
            result["obj_flow"] = torch.stack(all_obj_flows)  # (T, max_obj, 2, ps, ps)
        return result


# ========================================================================= #
#  Kreiranje DataLoadera                                                      #
# ========================================================================= #

def create_dataloaders(config, train_vids=None, val_vids=None):
    """Stvara train / val / test DataLoader-e."""
    frames_root = config["frames_root"]
    metadata_dir = config["metadata_dir"]
    annotations_dir = config.get("annotations_dir", "../dataset/annotations")

    # Učitaj metadata
    all_metadata = {}
    for mf in ["metadata_train.json", "metadata_val.json", "metadata_test.json"]:
        p = os.path.join(metadata_dir, mf)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                all_metadata.update(json.load(f))

    # Splitovi
    def _read_split(path):
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    all_train = _read_split(os.path.join(metadata_dir, "train_split.txt"))
    test_vids = _read_split(os.path.join(metadata_dir, "test_split.txt"))

    # Ako su eksplicitni splitovi proslijeđeni (k-fold CV), koristi ih
    if train_vids is not None and val_vids is not None:
        print(f"\nSplit (CV override): Train={len(train_vids)}  Val={len(val_vids)}  Test={len(test_vids)}")
    else:
        # 80/20 train/val split
        random.seed(config.get("seed", 42))
        random.shuffle(all_train)
        split = int(len(all_train) * 0.8)
        train_vids = all_train[:split]
        val_vids = all_train[split:]
        print(f"\nSplit: Train={len(train_vids)}  Val={len(val_vids)}  Test={len(test_vids)}")

    yolo_dir = config.get("yolo_detections_dir", None)

    common = dict(
        seq_len=config.get("seq_len", 24),
        img_size=tuple(config.get("img_size", [128, 128])),
        stride=config.get("stride", 12),
        stride_safe=config.get("stride_safe", None),
        stride_danger=config.get("stride_danger", None),
        danger_horizon=config.get("danger_horizon", 10),
        max_objects=config.get("max_objects", 10),
        orig_img_size=tuple(config.get("orig_img_size", [1280, 720])),
        yolo_detections_dir=yolo_dir,
        cnn_cache_dir=config.get("cnn_cache_dir", None),
    )

    if yolo_dir:
        print(f"  YOLO detekcije: {yolo_dir}")

    flow_dir = config.get("optical_flow_dir", None) if config.get("use_optical_flow", False) else None
    use_per_obj_flow = config.get("use_per_object_flow", False)
    obj_flow_ps = config.get("obj_flow_pool_size", 5)
    if flow_dir:
        common["optical_flow_dir"] = flow_dir
        common["use_per_object_flow"] = use_per_obj_flow
        common["obj_flow_pool_size"] = obj_flow_ps
        print(f"  Optical flow: {flow_dir}")
        if use_per_obj_flow:
            print(f"  Per-object flow: pool_size={obj_flow_ps}")

    temporal_mask = config.get("temporal_mask_prob", 0.0)
    no_ramp = config.get("no_ramp", False)

    train_ds = DoTANextFrameDataset(
        frames_root, annotations_dir, all_metadata, train_vids,
        phase="train", augment=True, temporal_mask_prob=temporal_mask,
        no_ramp=no_ramp, **common,
    )
    # Val/test: stride=1 (gusto uzorkovanje), bez hybrid stride
    val_common = dict(common)
    val_common["stride_safe"] = None
    val_common["stride_danger"] = None
    val_ds = DoTANextFrameDataset(
        frames_root, annotations_dir, all_metadata, val_vids,
        phase="val", augment=False, no_ramp=True, **val_common,
    )
    test_ds = DoTANextFrameDataset(
        frames_root, annotations_dir, all_metadata, test_vids,
        phase="test", augment=False, no_ramp=True, **val_common,
    )

    bs = config.get("batch_size", 8)
    nw = config.get("num_workers", 0)
    persist = config.get("persistent_workers", False) and nw > 0
    prefetch = config.get("prefetch_factor", 2) if nw > 0 else None
    # Windows: pin_memory s num_workers>0 koristi file-mapped shared memory koja nije
    # resizable → RuntimeError u collate_fn. Na Windowsu koristimo pin_memory=False.
    pin_mem = (sys.platform != "win32") and (nw > 0)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=pin_mem, drop_last=True,
        persistent_workers=persist, prefetch_factor=prefetch,
    )
    # Val/test: veći batch (nema backward → manje VRAM-a) + pin_memory + paralelno učitavanje
    val_bs = config.get("val_batch_size", bs * 2)
    val_nw = config.get("val_num_workers", nw)
    val_persist = config.get("persistent_workers", False) and val_nw > 0
    val_prefetch = config.get("prefetch_factor", 2) if val_nw > 0 else None
    val_loader = DataLoader(
        val_ds, batch_size=val_bs, shuffle=False,
        num_workers=val_nw, pin_memory=False,
        persistent_workers=val_persist, prefetch_factor=val_prefetch,
    )
    test_loader = DataLoader(
        test_ds, batch_size=val_bs, shuffle=False,
        num_workers=val_nw, pin_memory=False,
        persistent_workers=val_persist, prefetch_factor=val_prefetch,
    )

    return train_loader, val_loader, test_loader


def create_test_dataloader(config, split="test"):
    """Gradi SAMO test (ili val) dataloader — brže za evaluaciju."""
    frames_root = config["frames_root"]
    metadata_dir = config.get("metadata_dir", "../dataset")
    annotations_dir = config.get("annotations_dir", "../dataset/annotations")

    all_metadata = {}
    for fname in ["metadata_train.json", "metadata_val.json", "metadata_test.json"]:
        p = os.path.join(metadata_dir, fname)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                all_metadata.update(json.load(f))

    def _read_split(path):
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    if split == "test":
        # Podrška za override split fajla (npr. dota_test_split.txt za DoTA-only evaluaciju)
        split_file = config.get("test_split_file", None) or os.path.join(metadata_dir, "test_split.txt")
        vids = _read_split(split_file)
    else:
        all_train = _read_split(os.path.join(metadata_dir, "train_split.txt"))
        random.seed(config.get("seed", 42))
        random.shuffle(all_train)
        s = int(len(all_train) * 0.8)
        vids = all_train[s:]  # val split

    print(f"\n{split.upper()} set: {len(vids)} videa")

    yolo_dir = config.get("yolo_detections_dir", None)

    common = dict(
        seq_len=config.get("seq_len", 24),
        img_size=tuple(config.get("img_size", [128, 128])),
        stride=config.get("stride", 12),
        danger_horizon=config.get("danger_horizon", 10),
        max_objects=config.get("max_objects", 10),
        orig_img_size=tuple(config.get("orig_img_size", [1280, 720])),
        yolo_detections_dir=yolo_dir,
        cnn_cache_dir=config.get("cnn_cache_dir", None),
    )

    flow_dir = config.get("optical_flow_dir", None) if config.get("use_optical_flow", False) else None
    if flow_dir:
        common["optical_flow_dir"] = flow_dir
        common["use_per_object_flow"] = config.get("use_per_object_flow", False)
        common["obj_flow_pool_size"] = config.get("obj_flow_pool_size", 5)

    ds = DoTANextFrameDataset(
        frames_root, annotations_dir, all_metadata, vids,
        phase=split, augment=False, no_ramp=True, **common,
    )

    bs = config.get("batch_size", 8)
    nw = config.get("num_workers", 0)
    persist = config.get("persistent_workers", False) and nw > 0
    prefetch = config.get("prefetch_factor", 2) if nw > 0 else None
    pin_mem = (sys.platform != "win32") and (nw > 0)
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=nw, pin_memory=pin_mem,
                        persistent_workers=persist, prefetch_factor=prefetch)
    return loader
