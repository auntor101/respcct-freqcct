"""Feature extraction for RespCCT and FreqCCT.

Every recording is resampled to 4 kHz and cut into 5 s frames with 50% overlap. In each frame the DFT bins
from 0 to 1 Hz are set to zero and the frame is divided by its largest absolute sample. A frame becomes a
3 x 64 x 40 image stack (mel spectrogram, constant-Q transform, MFCC), stored as float16. Non-COPD classes are
augmented with copies of real frames until they match the number of COPD frames.

Input: a CSV table of recordings with the columns
    original_filepath, source, unified_label, pkey, fold
where `pkey` identifies the patient and `fold` is the fold assigned to that patient.

Output files, written to `out_dir` (the layout read by the training notebook):
    feat_orig.npy, meta_orig.csv              real frames and their metadata
    wave_minority.npy, wave_meta.csv          waveforms of the real non-COPD frames, used for augmentation
    feat_aug_<class>.npy, meta_aug_<class>.csv    augmented frames per class

Usage:
    python code/features.py recordings.csv features/
"""
import json
import os
import sys
import time

import librosa
import numpy as np
import pandas as pd
from skimage.transform import resize

SR = 4000                      # sampling rate in Hz
FRAME = 20000                  # 5 s
HOP = 10000                    # 2.5 s, 50% overlap
N_FFT = 1024
MEL_HOP = 524
CQT_HOP = 512
MFCC_HOP = 512
IMAGE_SHAPE = (64, 40)
CLASSES = ["COPD", "Healthy", "Asthma", "Pneumonia", "URTI", "Bronchiectasis", "Bronchiolitis"]
META_COLUMNS = ["rec_idx", "label", "source", "pkey", "fold"]


def frames_from_recording(path):
    """Resample to 4 kHz, cut into frames, remove 0 to 1 Hz and normalise each frame to a peak of one."""
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if len(y) < FRAME:
        y = np.pad(y, (0, FRAME - len(y)))
    n_frames = 1 + (len(y) - FRAME) // HOP
    cutoff_bin = int(np.ceil(1.0 * FRAME / SR))          # DFT bin of 1 Hz, which is 5 for a 5 s frame
    frames = []
    for p in range(n_frames):
        x = y[p * HOP: p * HOP + FRAME].astype(np.float64)
        spectrum = np.fft.rfft(x)
        spectrum[:cutoff_bin + 1] = 0
        x = np.fft.irfft(spectrum, n=FRAME)
        peak = np.max(np.abs(x))
        frames.append((x / peak if peak > 0 else x).astype(np.float32))
    return np.stack(frames)


def _scale_and_resize(a):
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    if a.shape != IMAGE_SHAPE:
        a = resize(a, IMAGE_SHAPE, order=1, anti_aliasing=False)
    return a.astype(np.float32)


def spectral_images(x):
    """Mel, CQT and MFCC images of one frame before resizing: shapes (64, 39), (64, 40) and (30, 40)."""
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, hop_length=MEL_HOP, win_length=N_FFT,
                                       window="hamming", n_mels=64), ref=np.max)
    cqt = librosa.power_to_db(
        np.abs(librosa.cqt(x, sr=SR, hop_length=CQT_HOP, fmin=32.7, n_bins=64, bins_per_octave=12,
                           window="hamming")) ** 2, ref=np.max)
    mfcc = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=30, n_mels=30, n_fft=N_FFT, hop_length=MFCC_HOP,
                                win_length=N_FFT, window="hamming")
    return mel, cqt, mfcc


def to_image_stack(x):
    """One frame to a float16 array of shape (3, 64, 40)."""
    return np.stack([_scale_and_resize(a) for a in spectral_images(x)]).astype(np.float16)


def augment(x, rng):
    """One of time stretching, pitch shifting, additive noise, or all three in sequence."""
    x = x.copy()
    choice = rng.choice(["stretch", "pitch", "noise", "combo"])
    if choice in ("stretch", "combo"):
        x = librosa.effects.time_stretch(x, rate=rng.uniform(0.85, 1.15))
    if choice in ("pitch", "combo"):
        x = librosa.effects.pitch_shift(x, sr=SR, n_steps=rng.uniform(-3, 3))
    if choice in ("noise", "combo"):
        x = x + rng.uniform(0.001, 0.01) * rng.standard_normal(len(x))
    x = x[:FRAME] if len(x) >= FRAME else np.pad(x, (0, FRAME - len(x)))
    peak = np.max(np.abs(x))
    return (x / peak if peak > 0 else x).astype(np.float32)


def build_real_frames(recordings, out_dir, log=print):
    """Stage A: frames of all recordings. Waveforms of non-COPD frames are kept for augmentation."""
    feats, meta, waves, wave_meta, failed = [], [], [], [], 0
    start = time.time()
    for i in range(len(recordings)):
        r = recordings.iloc[i]
        try:
            frames = frames_from_recording(r["original_filepath"])
        except Exception as err:
            failed += 1
            log(f"could not read {r['original_filepath']}: {type(err).__name__}")
            continue
        row = (i, r["unified_label"], r["source"], r["pkey"], int(r["fold"]))
        for x in frames:
            feats.append(to_image_stack(x))
            meta.append(row)
            if r["unified_label"] != "COPD":
                waves.append(x)
                wave_meta.append(row)
        if (i + 1) % 100 == 0:
            log(f"{i + 1}/{len(recordings)} recordings, {len(feats)} frames, {(time.time() - start) / 60:.1f} min")
    np.save(os.path.join(out_dir, "wave_minority.npy"), np.stack(waves))
    pd.DataFrame(wave_meta, columns=META_COLUMNS).to_csv(os.path.join(out_dir, "wave_meta.csv"), index=False)
    pd.DataFrame(meta, columns=META_COLUMNS).to_csv(os.path.join(out_dir, "meta_orig.csv"), index=False)
    np.save(os.path.join(out_dir, "feat_orig.npy"), np.stack(feats))   # written last: its presence marks Stage A as complete
    log(f"Stage A: {len(feats)} frames, {failed} recordings failed")


def build_augmented_frames(out_dir, seed=42, log=print):
    """Stage B: augment each non-COPD class up to the COPD frame count. Finished classes are skipped."""
    rng = np.random.default_rng(seed)
    meta = pd.read_csv(os.path.join(out_dir, "meta_orig.csv"))
    counts = meta["label"].value_counts()
    target = int(counts.max())
    waves = np.load(os.path.join(out_dir, "wave_minority.npy"))
    wave_meta = pd.read_csv(os.path.join(out_dir, "wave_meta.csv"))
    for cls in CLASSES[1:]:
        path = os.path.join(out_dir, f"feat_aug_{cls}.npy")
        state_path = os.path.join(out_dir, f"rng_state_after_{cls}.json")   # lets an interrupted run resume with the same random sequence
        if os.path.exists(path):
            if os.path.exists(state_path):
                rng.bit_generator.state = json.load(open(state_path))
            log(f"{cls}: already built")
            continue
        need = target - int(counts.get(cls, 0))
        picks = rng.choice(np.where(wave_meta["label"].values == cls)[0], size=need, replace=True)
        out = np.zeros((need, 3, *IMAGE_SHAPE), dtype=np.float16)
        for n, j in enumerate(picks):
            out[n] = to_image_stack(augment(waves[j], rng))
        wave_meta.iloc[picks].reset_index(drop=True).to_csv(os.path.join(out_dir, f"meta_aug_{cls}.csv"), index=False)
        np.save(path, out)
        json.dump(rng.bit_generator.state, open(state_path, "w"))
        log(f"{cls}: {need} augmented frames")


def build_features(recordings_csv, out_dir):
    recordings = pd.read_csv(recordings_csv)
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(os.path.join(out_dir, "feat_orig.npy")):
        build_real_frames(recordings, out_dir)
    build_augmented_frames(out_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python code/features.py recordings.csv output_dir")
    build_features(sys.argv[1], sys.argv[2])
