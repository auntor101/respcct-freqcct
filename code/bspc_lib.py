# bspc_lib.py : models, metrics, training engine (checkpoint + resume), explanation helpers.
# Written by notebook cell 3. Everything the notebook trains or explains goes through this file.
import os, json, time, copy, math, hashlib, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score, confusion_matrix

NC = 7
CLASSES = ["COPD", "Healthy", "Asthma", "Pneumonia", "URTI", "Bronchiectasis", "Bronchiolitis"]
LIB_HASH = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:10] if "__file__" in globals() else "nofile"
DATA_FP = "unset"          # set by the notebook: fingerprint of the feature files

# ======================= models =======================
class DropPath(nn.Module):
    def __init__(self, p=0.0): super().__init__(); self.p = float(p)
    def forward(self, x):
        if not self.training or self.p == 0.0: return x
        keep = 1.0 - self.p
        return x * (torch.rand(x.shape[0], 1, 1, device=x.device) < keep).to(x.dtype) / keep

class ConvTokenizer(nn.Module):
    """Two conv blocks (3x3 conv, ReLU, max-pool 3/2/1). dw=True: second conv = depthwise 3x3 + pointwise 1x1."""
    def __init__(self, c=3, e=64, dw=False):
        super().__init__()
        conv2 = nn.Sequential(nn.Conv2d(e, e, 3, 1, 1, groups=e), nn.Conv2d(e, e, 1)) if dw else nn.Conv2d(e, e, 3, 1, 1)
        self.net = nn.Sequential(nn.Conv2d(c, e, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
                                 conv2, nn.ReLU(), nn.MaxPool2d(3, 2, 1))
    def forward(self, x): return self.net(x).flatten(2).transpose(1, 2)       # (B, nf*nt, e), frequency-major

class Block(nn.Module):
    def __init__(self, e, h, r=2, mlp_drop=0.0, attn_drop=0.0, dp=0.0, use_attn=True):
        super().__init__()
        self.use_attn = use_attn; hid = int(round(e * r))
        self.n1, self.n2 = nn.LayerNorm(e), nn.LayerNorm(e)
        if use_attn: self.attn = nn.MultiheadAttention(e, h, dropout=attn_drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(e, hid), nn.GELU(), nn.Dropout(mlp_drop), nn.Linear(hid, e), nn.Dropout(mlp_drop))
        self.dp = DropPath(dp); self.keep_attn, self.last_attn = False, None
    def forward(self, x):
        if not self.use_attn: return x + self.dp(self.mlp(self.n2(x)))          # capacity-matched control: no token mixing
        h = self.n1(x)
        if self.keep_attn:
            a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True); self.last_attn = w
        else:
            a = self.attn(h, h, h, need_weights=False)[0]
        x = x + self.dp(a)
        return x + self.dp(self.mlp(self.n2(x)))

def _blocks(e, layers, heads, ratio, mlp_drop, attn_drop, sd, use_attn=True):
    return nn.ModuleList([Block(e, heads, ratio, mlp_drop, attn_drop, sd * i / max(layers - 1, 1), use_attn) for i in range(layers)])

class ViewFusion(nn.Module):
    """Optional input stage over the 3 views (mel, CQT, MFCC). mode: norm = per-view standardisation per frame | stem = grouped conv
       (one small filter bank per view, residual) | se = squeeze-and-excitation gate over the views (weights kept in last_gate).
       view_drop: probability of zeroing one whole view per frame during training (never all of them)."""
    def __init__(self, c=3, mode=None, view_drop=0.0):
        super().__init__(); self.mode, self.vd, self.last_gate = mode, float(view_drop), None
        if mode == "stem": self.g1, self.g2 = nn.Conv2d(c, 4 * c, 3, padding=1, groups=c), nn.Conv2d(4 * c, c, 1)
        if mode == "se": self.fc = nn.Sequential(nn.Linear(c, 8), nn.ReLU(), nn.Linear(8, c))
    def gate(self, x): return torch.sigmoid(self.fc(x.mean((2, 3))))
    def forward(self, x):
        if self.training and self.vd > 0 and x.shape[1] > 1:
            keep = (torch.rand(x.shape[0], x.shape[1], 1, 1, device=x.device) >= self.vd).to(x.dtype)
            keep[keep.sum(1).view(-1) == 0] = 1.0; x = x * keep
        if self.mode == "norm": x = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-4)
        elif self.mode == "stem": x = x + self.g2(F.relu(self.g1(x)))
        elif self.mode == "se": g = self.gate(x); self.last_gate = g.detach(); x = x * (2.0 * g)[:, :, None, None]
        return x

def make_fuse(c, mode, vd):
    return ViewFusion(c, mode, vd) if (mode or vd > 0) else None

class CCT(nn.Module):
    """Compact convolutional transformer: tokenizer, encoder, SeqPool. pe=True adds a learned positional embedding."""
    def __init__(self, nc=NC, e=64, layers=4, heads=4, ratio=2, mlp_drop=0., attn_drop=0., sd=0., pe=False, dw=False, nf=16, nt=10, c=3,
                 attn=True, pool="seq", fusion=None, view_drop=0.0):
        super().__init__()
        self.nf, self.nt, self.kind, self.pooling = nf, nt, "cct", pool
        self.fuse = make_fuse(c, fusion, view_drop)
        self.tok = ConvTokenizer(c, e, dw)
        self.pos = None
        if pe: self.pos = nn.Parameter(torch.zeros(1, nf * nt, e)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, e)) if pool == "cls" else None
        self.blocks = _blocks(e, layers, heads, ratio, mlp_drop, attn_drop, sd, attn)
        self.pool = nn.Linear(e, 1) if pool == "seq" else None
        self.fc = nn.Linear(e, nc)
    def encode(self, x, perm=None):
        if self.fuse is not None: x = self.fuse(x)
        z = self.tok(x)
        if perm is not None: z = z[:, perm]
        if self.pos is not None: z = z + self.pos
        if self.cls is not None: z = torch.cat([self.cls.expand(z.shape[0], -1, -1), z], 1)
        for b in self.blocks: z = b(z)
        return z
    def forward(self, x, perm=None, ret_pool=False, ret_emb=False):
        z = self.encode(x, perm)
        if self.pooling == "seq":
            w = torch.softmax(self.pool(z), dim=1); emb = (w.transpose(1, 2) @ z).squeeze(1)
        elif self.pooling == "mean": w = torch.full(z.shape[:2] + (1,), 1.0 / z.shape[1], device=z.device); emb = z.mean(1)
        else: w = torch.zeros(z.shape[0], z.shape[1], 1, device=z.device); emb = z[:, 0]
        out = self.fc(emb)
        if ret_emb: return out, emb
        return (out, w.squeeze(-1)) if ret_pool else out
    @torch.no_grad()
    def contributions(self, x, target):
        """Exact additive contribution of every final token to the target logit (bias excluded): w_t * (W_c . z_t)."""
        assert self.pooling in ("seq", "mean"), "token contributions are defined for SeqPool and mean pooling only"
        z = self.encode(x)
        w = torch.softmax(self.pool(z), dim=1).squeeze(-1) if self.pooling == "seq" else torch.full(z.shape[:2], 1.0 / z.shape[1], device=z.device)
        Wc = self.fc.weight[target]                                     # (B, e)
        return (w * torch.einsum("bte,be->bt", z, Wc)).view(-1, self.nf, self.nt)

class FreqCCT(nn.Module):
    """dw: depthwise-separable 2nd tokenizer conv. pe: learned positional embedding. fp: frequency-preserving head
       (LayerNorm, pool over time, keep nf rows, linear nf*e -> classes) else SeqPool. tpool: 'mean' or 'attn' (learned weights over time)."""
    def __init__(self, nc=NC, e=64, layers=4, heads=4, ratio=2, mlp_drop=0., attn_drop=0., sd=0., dw=True, pe=True, fp=True,
                 tpool="mean", nf=16, nt=10, c=3, attn=True, fusion=None, view_drop=0.0):
        super().__init__()
        self.nf, self.nt, self.fp, self.tpool, self.kind = nf, nt, fp, tpool, "freqcct"
        self.fuse = make_fuse(c, fusion, view_drop)
        self.tok = ConvTokenizer(c, e, dw)
        self.pos = None
        if pe: self.pos = nn.Parameter(torch.zeros(1, nf * nt, e)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = _blocks(e, layers, heads, ratio, mlp_drop, attn_drop, sd, attn)
        if fp:
            self.norm = nn.LayerNorm(e); self.fc = nn.Linear(nf * e, nc)
            if tpool == "attn": self.tp = nn.Linear(e, 1)
        else:
            self.pool = nn.Linear(e, 1); self.fc = nn.Linear(e, nc)
    def encode(self, x, perm=None):
        if self.fuse is not None: x = self.fuse(x)
        z = self.tok(x)
        if perm is not None: z = z[:, perm]                             # shuffle content BEFORE position is added
        if self.pos is not None: z = z + self.pos
        for b in self.blocks: z = b(z)
        return z
    def _rows(self, z):
        h = self.norm(z).view(z.shape[0], self.nf, self.nt, -1)
        if self.tpool == "attn":
            a = torch.softmax(self.tp(h).squeeze(-1), dim=2)             # (B, nf, nt)
            return (h * a.unsqueeze(-1)).sum(2), a
        return h.mean(2), None
    def forward(self, x, perm=None, ret_pool=False, ret_emb=False):
        z = self.encode(x, perm)
        if self.fp:
            r, a = self._rows(z); emb = r.flatten(1); out = self.fc(emb); w = None if a is None else a.flatten(1)
        else:
            w = torch.softmax(self.pool(z), dim=1); emb = (w.transpose(1, 2) @ z).squeeze(1); out = self.fc(emb); w = w.squeeze(-1)
        if ret_emb: return out, emb
        return (out, w) if ret_pool else out
    @torch.no_grad()
    def contributions(self, x, target):
        z = self.encode(x)
        if not self.fp:
            w = torch.softmax(self.pool(z), dim=1).squeeze(-1)
            return (w * torch.einsum("bte,be->bt", z, self.fc.weight[target])).view(-1, self.nf, self.nt)
        h = self.norm(z).view(z.shape[0], self.nf, self.nt, -1)
        Wc = self.fc.weight[target].view(-1, self.nf, 1, h.shape[-1])    # (B, nf, 1, e)
        if self.tpool == "attn": w = torch.softmax(self.tp(h).squeeze(-1), dim=2)
        else: w = torch.full(h.shape[:3], 1.0 / self.nt, device=x.device)
        return w * (h * Wc).sum(-1)

BASELINES = {"shufflenet05": "ShuffleNetV2 x0.5", "mnasnet05": "MNASNet 0.5", "mobilenetv3s": "MobileNetV3-Small", "mobilevitxxs": "MobileViT-XXS",
             "effnetb0": "EfficientNet-B0", "resnet18": "ResNet-18"}

class Pretrained(nn.Module):
    """ImageNet-pretrained torchvision backbone, 7-class linear head. Input (B,3,64,40) in [0,1]: ImageNet normalisation,
       bilinear upscaling by `scale`, then the backbone."""
    def __init__(self, name, scale=2, pretrained=True, nc=NC):
        super().__init__()
        import torchvision.models as tv
        self.scale, self.kind = scale, name
        W = (lambda enum: enum.DEFAULT if pretrained else None)
        if name == "shufflenet05":
            m = tv.shufflenet_v2_x0_5(weights=W(tv.ShuffleNet_V2_X0_5_Weights)); m.fc = nn.Linear(m.fc.in_features, nc)
        elif name == "mnasnet05":
            m = tv.mnasnet0_5(weights=W(tv.MNASNet0_5_Weights)); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, nc)
        elif name == "mobilenetv3s":
            m = tv.mobilenet_v3_small(weights=W(tv.MobileNet_V3_Small_Weights)); m.classifier = nn.Linear(576, nc)
        elif name == "mobilevitxxs":                                  # timm, ImageNet-1k pretrained, 7-class linear head
            import timm
            m = timm.create_model("mobilevit_xxs", pretrained=pretrained, num_classes=nc)
        elif name == "effnetb0":
            m = tv.efficientnet_b0(weights=W(tv.EfficientNet_B0_Weights)); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, nc)
        elif name == "resnet18":
            m = tv.resnet18(weights=W(tv.ResNet18_Weights)); m.fc = nn.Linear(m.fc.in_features, nc)
        else: raise ValueError(name)
        self.m = m
        mu, sd = ([0., 0., 0.], [1., 1., 1.]) if name == "mobilevitxxs" else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])   # MobileViT is trained on [0,1] input
        self.register_buffer("mu", torch.tensor(mu).view(1, 3, 1, 1)); self.register_buffer("sd", torch.tensor(sd).view(1, 3, 1, 1))
    def forward(self, x, **kw):
        x = (x - self.mu) / self.sd
        if self.scale != 1: x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return self.m(x)

def n_rows_of(spec): return 16 if spec.get("rows") is None else len(spec["rows"])
def n_channels_of(spec): return 3 if spec.get("channels") is None else len(spec["channels"])

def build_model(spec):
    a = spec["arch"]; g = lambda k, d: spec.get(k, d)
    kw = dict(e=g("e", 64), layers=g("layers", 4), heads=g("heads", 4), ratio=g("ratio", 2), mlp_drop=g("mlp_drop", 0.),
              attn_drop=g("attn_drop", 0.), sd=g("sd", 0.), nf=n_rows_of(spec), c=n_channels_of(spec))
    kw.update(attn=g("attn", True), fusion=g("fusion", None), view_drop=g("view_drop", 0.0))
    if a == "cct": return CCT(pe=g("pe", False), dw=g("dw", False), pool=g("pool", "seq"), **kw)
    if a == "freqcct": return FreqCCT(dw=g("dw", True), pe=g("pe", True), fp=g("fp", True), tpool=g("tpool", "mean"), **kw)
    if a in BASELINES: return Pretrained(a, scale=g("scale", 2), pretrained=g("pretrained", True))
    raise ValueError(a)

def n_params(m): return int(sum(p.numel() for p in m.parameters()))

def match_ratio(spec, target_params, grid=None):
    """FFN ratio of the attention-free control (spec with attn=False) whose parameter count is closest to target_params."""
    grid = np.arange(1.0, 12.0, 0.05) if grid is None else grid
    best = min(grid, key=lambda r: abs(n_params(build_model({**spec, "attn": False, "ratio": float(r)})) - target_params))
    return float(round(best, 2))

def analytic_mflops(arch="cct", e=64, layers=4, ratio=2, nf=16, nt=10, c=3, nc=NC, dw=None, fp=None, tpool="mean", **_):
    """2 x multiply-accumulates per frame (attention included), CCT family. Reproduces 124.2 (CCT) and 83.0 (FreqCCT)."""
    dw = (arch == "freqcct") if dw is None else dw; fp = (arch == "freqcct") if fp is None else fp
    H, W = 4 * nf, 4 * nt; n = nf * nt
    conv1 = H * W * c * e * 9
    conv2 = (H // 2) * (W // 2) * ((e * 9 + e * e) if dw else e * e * 9)
    attn = layers * (n * e * 3 * e + 2 * n * n * e + n * e * e)
    mlp = layers * (2 * n * e * e * ratio)
    head = nf * e * nc + (n * e if tpool == "attn" else 0) if fp else n * e + e * nc
    return 2.0 * (conv1 + conv2 + attn + mlp + head) / 1e6

def spec_mflops(spec):
    if spec["arch"] not in ("cct", "freqcct"): return float("nan")
    return analytic_mflops(spec["arch"], e=spec.get("e", 64), layers=spec.get("layers", 4), ratio=spec.get("ratio", 2),
                           nf=n_rows_of(spec), c=n_channels_of(spec), dw=spec.get("dw"), fp=spec.get("fp"), tpool=spec.get("tpool", "mean"))

def counted_mflops(model, shape):
    """torch FlopCounterMode on a CPU copy in train mode with keep_attn=True (the fused eval path is invisible to the counter)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        m = copy.deepcopy(model).cpu().train()
        for b in getattr(m, "blocks", []):
            b.dp.p = 0.0
            if getattr(b, "use_attn", True): b.keep_attn = True; b.attn.dropout = 0.0
        with torch.no_grad(), FlopCounterMode(display=False) as fc: m(torch.zeros(1, *shape))
        return fc.get_total_flops() / 1e6
    except Exception as ex:
        return float("nan")

@torch.no_grad()
def cpu_latency_ms(model, shape, warm=30, reps=200):
    m = copy.deepcopy(model).cpu().eval(); x = torch.zeros(1, *shape)
    for _ in range(warm): m(x)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); m(x); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 90))

def peak_activation_kb(model, shape):
    m = copy.deepcopy(model).cpu().eval(); sizes = []
    hs = [mod.register_forward_hook(lambda mod_, i, o: sizes.append(o.numel() if torch.is_tensor(o) else 0))
          for mod in m.modules() if len(list(mod.children())) == 0]
    with torch.no_grad(): m(torch.zeros(1, *shape))
    for h in hs: h.remove()
    big = max(sizes) if sizes else 0
    if hasattr(m, "blocks") and len(m.blocks) and getattr(m.blocks[0], "use_attn", True): n = m.nf * m.nt; big = max(big, m.blocks[0].attn.num_heads * n * n)
    return 4.0 * big / 1e3

# ======================= metrics =======================
def macro_f1_present(y, p):
    labs = np.unique(y)
    return float(f1_score(y, p, labels=labs, average="macro", zero_division=0))

def scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return dict(accuracy=float((y == p).mean()), macro_f1=macro_f1_present(y, p),
                macro_f1_all7=float(f1_score(y, p, labels=list(range(NC)), average="macro", zero_division=0)),
                bal_acc=float(balanced_accuracy_score(y, p)), mcc=float(matthews_corrcoef(y, p)),
                majority_acc=float(np.bincount(y, minlength=NC).max() / len(y)), n_classes=int(len(np.unique(y))), n=int(len(y)))

def per_class(y, p):
    cm = confusion_matrix(y, p, labels=list(range(NC))); N = cm.sum(); rows = []
    for c, name in enumerate(CLASSES):
        TP = cm[c, c]; FN = cm[c].sum() - TP; FP = cm[:, c].sum() - TP; TN = N - TP - FN - FP
        rc, sp, pr = TP / max(TP + FN, 1), TN / max(TN + FP, 1), TP / max(TP + FP, 1)
        rows.append(dict(cls=name, n=int(cm[c].sum()), precision=float(pr), recall=float(rc), specificity=float(sp),
                         f1=float(2 * pr * rc / max(pr + rc, 1e-9))))
    return cm, rows

def rec_vote(P, y, rec):
    df = pd.DataFrame(np.asarray(P)); cols = list(df.columns); df["rec"] = np.asarray(rec); df["y"] = np.asarray(y)
    g = df.groupby("rec", sort=True)
    return g[cols].mean().values, g["y"].first().values.astype(int), np.array(list(g.groups.keys()))

def bootstrap_diff(y, pa, pb, groups, B=2000, seed=0, metric="macro_f1"):
    """Paired percentile CI of metric(pa) - metric(pb), resampling whole groups (recordings)."""
    y, pa, pb, groups = map(np.asarray, (y, pa, pb, groups))
    ug, inv = np.unique(groups, return_inverse=True); members = [np.where(inv == k)[0] for k in range(len(ug))]
    rng = np.random.default_rng(seed); d = []
    for _ in range(B):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(ug), len(ug))])
        d.append(scores(y[idx], pa[idx])[metric] - scores(y[idx], pb[idx])[metric])
    d0 = scores(y, pa)[metric] - scores(y, pb)[metric]
    return float(d0), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def mcnemar(ca, cb):
    from scipy.stats import chi2
    a, b = np.asarray(ca, bool), np.asarray(cb, bool); n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0: return 0.0, 1.0, n01, n10
    s = (abs(n01 - n10) - 1) ** 2 / (n01 + n10); return float(s), float(chi2.sf(s, 1)), n01, n10

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj

def logit_adjust(P, prior_train, prior_real, tau):
    """Post-hoc prior correction: p * (pi_real / pi_train)^tau, renormalised. tau chosen on validation only."""
    A = np.asarray(P) * (np.asarray(prior_real) / np.asarray(prior_train)) ** tau
    return A / A.sum(1, keepdims=True)

# ======================= splits =======================
def make_splits(M, y, seed=42):
    """Returns dict with 'paper' (list of 5 (train,val,test)) and 'patient' (list of 5) plus 'tune'.
       paper  : stratified 5-fold over pooled frames (reference protocol); val = 10% of train, stratified; test = real frames of the fold.
       patient: StratifiedGroupKFold by patient key over real frames; augmented copies follow their patient;
                val = real frames of 10% of the training patients; test = real frames of the test patients.
       tune   : inside the TRAINING patients of patient fold 0 only: inner validation = real frames of ~20% of those patients."""
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit, train_test_split
    aug = M["aug"].values; pk = M["pkey"].values; out = {"paper": [], "patient": []}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        tri, vai = train_test_split(tr, test_size=0.10, stratify=y[tr], random_state=seed)
        out["paper"].append((tri, vai, te[aug[te] == 0]))
    real = np.where(aug == 0)[0]
    fold_of_pk = {}
    for f, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(real, y[real], groups=pk[real])):
        for k in np.unique(pk[real[b]]): fold_of_pk[k] = f
    fold = np.array([fold_of_pk.get(k, -1) for k in pk])
    for f in range(5):
        tr = np.where(fold != f)[0]; te = np.where((fold == f) & (aug == 0))[0]
        a, b = next(GroupShuffleSplit(1, test_size=0.10, random_state=seed).split(tr, groups=pk[tr]))
        tri, vai = tr[a], tr[b]; vai = vai[aug[vai] == 0]
        out["patient"].append((tri, vai, te))
    tr0 = np.where(fold != 0)[0]; r0 = tr0[aug[tr0] == 0]
    sg = next(StratifiedGroupKFold(5, shuffle=True, random_state=7).split(r0, y[r0], groups=pk[r0]))
    val_pk = np.unique(pk[r0[sg[1]]])
    in_val = np.isin(pk[tr0], val_pk)
    out["tune"] = (tr0[~in_val], tr0[in_val & (aug[tr0] == 0)], None)
    out["fold_of_frame"] = fold
    # tune_frame: frame-level protocol (no patient grouping). Inside the TRAINING frames of paper fold 0: 85 % inner train, 15 % inner validation (real frames only).
    tr0, va0, _ = out["paper"][0]; pool_ = np.concatenate([tr0, va0])
    a_, b_ = train_test_split(pool_, test_size=0.15, stratify=y[pool_], random_state=7)
    out["tune_frame"] = (a_, b_[aug[b_] == 0], None)
    return out

# ======================= augmentation on the GPU =======================
def spec_augment(x, fmax, tmax, n=2):
    B, C, H, W = x.shape
    for dim, mx, size in ((2, fmax, H), (3, tmax, W)):
        if mx <= 0: continue
        ar = torch.arange(size, device=x.device)[None]
        for _ in range(n):
            w = torch.randint(0, mx + 1, (B, 1), device=x.device); s = (torch.rand(B, 1, device=x.device) * (size - w)).long()
            m = (ar >= s) & (ar < s + w)
            x = x.masked_fill(m[:, None, :, None] if dim == 2 else m[:, None, None, :], 0.0)
    return x

def prep(x, spec):
    if spec.get("channels") is not None: x = x[:, spec["channels"]]
    if spec.get("rows") is not None:
        r = torch.as_tensor(np.concatenate([np.arange(4 * k, 4 * k + 4) for k in spec["rows"]]), device=x.device); x = x[:, :, r]
    return x

# ======================= engine =======================
class Pruned(Exception): pass

def stable_hash(o): return hashlib.sha1(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:12]
def idx_hash(a): return hashlib.md5(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes()).hexdigest()[:10] if a is not None else "none"
def _json(o, path):
    tmp = path + ".tmp"; json.dump(o, open(tmp, "w"), indent=2, default=str); os.replace(tmp, path)
def _save(o, path):
    tmp = path + ".tmp"; torch.save(o, tmp); os.replace(tmp, path)

DEFAULT_TRAIN = dict(lr=2e-3, wd=0.0, batch_size=128, smooth=0.1, sched="plateau", fmask=0, tmask=0, mixup=0.0, ema=0.0,
                     max_epochs=100, patience=12, clip=1.0, seed=0, mlp_drop=0.0, attn_drop=0.0, sd=0.0, balance=False)

@torch.no_grad()
def predict_probs(model, X, idx, spec, device, bs=2048, use_amp=False):
    model.eval(); out = []
    for i in range(0, len(idx), bs):
        xb = prep(X[torch.as_tensor(idx[i:i + bs], device=device)].float(), spec)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out.append(F.softmax(model(xb).float(), 1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, NC), np.float32)

def _eval(model, X, y, idx, spec, device, use_amp):
    P = predict_probs(model, X, idx, spec, device, use_amp=use_amp); yt = y[idx]
    return P, float(-np.log(P[np.arange(len(yt)), yt] + 1e-9).mean()), float((P.argmax(1) == yt).mean()), macro_f1_present(yt, P.argmax(1))

def train_run(spec, X, y, idx, out_dir, device, evaluate_test=False, meta=None, report=None, log=print, hist_every=5):
    """Train one model. Files in out_dir: config.json status.json history.csv last.pt best.pt preds.npz metrics.json time.json.
       Resumes from last.pt when config.json matches (same settings, same data, same library); refuses to mix otherwise."""
    S = {**DEFAULT_TRAIN, **spec}; os.makedirs(out_dir, exist_ok=True)
    cfg = dict(spec=S, train=idx_hash(idx["train"]), val=idx_hash(idx["val"]), test=idx_hash(idx.get("test")), lib=LIB_HASH, data=DATA_FP)
    cfg["hash"] = stable_hash(cfg)
    cpath, spath = os.path.join(out_dir, "config.json"), os.path.join(out_dir, "status.json")
    if os.path.exists(cpath):
        old = json.load(open(cpath))
        if old.get("hash") != cfg["hash"]:
            raise RuntimeError(f"{out_dir} holds a run with different settings/data/code ({old.get('hash')} vs {cfg['hash']}). "
                               f"Set FRESH=True in cell 0 for a new run folder; nothing is overwritten silently.")
    else: _json(cfg, cpath)
    st = json.load(open(spath)) if os.path.exists(spath) else {}
    if st.get("state") == "done" and os.path.exists(os.path.join(out_dir, "metrics.json")):
        return json.load(open(os.path.join(out_dir, "metrics.json")))
    use_amp = device.type == "cuda"
    torch.manual_seed(42 + S["seed"]); np.random.seed(42 + S["seed"])
    model = build_model(S).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=S["lr"], weight_decay=S["wd"])
    if S["sched"] == "plateau": sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    else:
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda ep: (ep + 1) / 3 if ep < 3 else max(0.01, 0.5 * (1 + math.cos(math.pi * min(1.0, (ep - 3) / 57)))))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = None
    if S["ema"] > 0: ema = copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in ema.parameters()]
    hist, best, best_ep, bad, ep0, elapsed, steps = [], float("inf"), 0, 0, 0, 0.0, 0
    lp = os.path.join(out_dir, "last.pt")
    if os.path.exists(lp):
        ck = torch.load(lp, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sch.load_state_dict(ck["sch"]); scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema") is not None: ema.load_state_dict(ck["ema"])
        hist, best, best_ep, bad, ep0, elapsed, steps = ck["hist"], ck["best"], ck["best_ep"], ck["bad"], ck["epoch"], ck["elapsed"], ck["steps"]
        torch.set_rng_state(ck["rng"].cpu()); log(f"    resumed from epoch {ep0}")
    tr = torch.as_tensor(idx["train"], device=device); va = idx["val"]; yg = torch.as_tensor(y, device=device).long()
    cw = (1.0 / torch.bincount(yg[tr], minlength=NC).float().clamp(min=1)) if S["balance"] else None
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    _json({**st, "state": "running", "epoch": ep0}, spath)
    stopped = False
    if bad >= S["patience"] or ep0 >= S["max_epochs"]: ep0 = S["max_epochs"]       # nothing left to train
    for ep in range(ep0, S["max_epochs"]):
        t0 = time.time(); model.train(); perm = tr[torch.randperm(len(tr), device=device)] if not S["balance"] else tr[torch.multinomial(cw[yg[tr]], len(tr), replacement=True)]; tot, corr = 0.0, 0
        for i in range(0, len(perm), S["batch_size"]):
            b = perm[i:i + S["batch_size"]]; xb = prep(X[b].float(), S); yb = yg[b]
            if S["fmask"] or S["tmask"]: xb = spec_augment(xb, S["fmask"], S["tmask"])
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if S["mixup"] > 0:
                    lam = float(np.random.beta(S["mixup"], S["mixup"])); pi = torch.randperm(len(b), device=device)
                    out = model(lam * xb + (1 - lam) * xb[pi])
                    loss = lam * F.cross_entropy(out.float(), yb, label_smoothing=S["smooth"]) + (1 - lam) * F.cross_entropy(out.float(), yb[pi], label_smoothing=S["smooth"])
                else:
                    out = model(xb); loss = F.cross_entropy(out.float(), yb, label_smoothing=S["smooth"])
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            if S["clip"]: nn.utils.clip_grad_norm_(model.parameters(), S["clip"])
            scaler.step(opt); scaler.update(); steps += 1
            if ema is not None:
                d = min(S["ema"], (1 + steps) / (10 + steps))
                with torch.no_grad():
                    for pe_, pm in zip(ema.parameters(), model.parameters()): pe_.mul_(d).add_(pm.detach(), alpha=1 - d)
                    for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
            tot += loss.item() * len(b); corr += (out.argmax(1) == yb).sum().item()
        em = ema if ema is not None else model
        _, vl, vacc, vf1 = _eval(em, X, y, va, S, device, use_amp)
        lr_now = opt.param_groups[0]["lr"]
        if S["sched"] == "plateau": sch.step(vl)
        else: sch.step()
        dt = time.time() - t0; elapsed += dt
        hist.append(dict(epoch=ep + 1, train_loss=tot / len(perm), train_acc=corr / len(perm), val_loss=vl, val_acc=vacc, val_macro_f1=vf1, lr=lr_now, seconds=dt))
        if vl < best - 1e-4:
            best, best_ep, bad = vl, ep + 1, 0; _save(copy.deepcopy(em.state_dict()), os.path.join(out_dir, "best.pt"))
        else: bad += 1
        _save(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), scaler=scaler.state_dict(),
                   ema=None if ema is None else ema.state_dict(), hist=hist, best=best, best_ep=best_ep, bad=bad, epoch=ep + 1,
                   elapsed=elapsed, steps=steps, rng=torch.get_rng_state()), lp)
        pd.DataFrame(hist).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        _json({"state": "running", "epoch": ep + 1}, spath)
        if ep == 0 or (ep + 1) % hist_every == 0:
            log(f"    epoch {ep+1:3d} ({dt:.0f}s) lr {lr_now:.5f} | train loss {tot/len(perm):.3f} acc {corr/len(perm):.3f} | val loss {vl:.3f} acc {vacc:.3f} F1 {vf1:.3f}")
        if report is not None:
            try: report(ep + 1, vf1, vl)
            except Pruned:
                _json({"state": "pruned", "epoch": ep + 1}, spath); raise
        if bad >= S["patience"]: stopped = True; log(f"    early stopping at epoch {ep+1} (best epoch {best_ep})"); break
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device)) if ema is None else ema.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    fm = ema if ema is not None else model
    Pv, vl, vacc, vf1 = _eval(fm, X, y, va, S, device, use_amp)
    shape = (n_channels_of(S), 4 * n_rows_of(S), 40)
    h = hist[best_ep - 1]
    m = dict(trial=os.path.basename(out_dir), epochs_run=len(hist), best_epoch=best_ep, early_stopped=bool(stopped),
             val_loss=vl, val_acc=vacc, val_macro_f1=vf1, params=n_params(model), mflops=counted_mflops(model, shape),
             mflops_analytic=spec_mflops(S), seconds_total=elapsed, seconds_per_epoch=float(np.mean([r["seconds"] for r in hist])),
             seconds_to_best=float(sum(r["seconds"] for r in hist[:best_ep])), minutes=elapsed / 60.0, gpu=gpu, hash=cfg["hash"])
    arrays = dict(val_idx=np.asarray(va), val_probs=Pv.astype(np.float32))
    if evaluate_test and idx.get("test") is not None:
        te = np.asarray(idx["test"]); Pt = predict_probs(fm, X, te, S, device, use_amp=use_amp); yt = y[te]; pr = Pt.argmax(1)
        sc = scores(yt, pr); cm, pcr = per_class(yt, pr)
        m.update({f"test_{k}": v for k, v in sc.items()}); m["per_class"] = pcr; m["confusion"] = cm.tolist()
        if meta is not None:
            Pr, yr, _ = rec_vote(Pt, yt, meta["rec_idx"].values[te]); sr = scores(yr, Pr.argmax(1))
            m.update({f"rec_{k}": v for k, v in sr.items()})
        arrays.update(test_idx=te, test_probs=Pt.astype(np.float32))
    np.savez_compressed(os.path.join(out_dir, "preds.npz"), **arrays)
    _json(m, os.path.join(out_dir, "metrics.json"))
    _json(dict(seconds_total=elapsed, minutes=elapsed / 60, seconds_per_epoch=m["seconds_per_epoch"], epochs=len(hist), best_epoch=best_ep,
               seconds_to_best=m["seconds_to_best"], gpu=gpu), os.path.join(out_dir, "time.json"))
    _json({"state": "done", "epoch": len(hist)}, spath)
    try: os.remove(lp)                                   # finished: the per-epoch checkpoint is no longer needed
    except OSError: pass
    return m

def load_model(spec, out_dir, device):
    S = {**DEFAULT_TRAIN, **spec}; m = build_model(S).to(device)
    m.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device)); return m.eval()

# ======================= explanation helpers =======================
def integrated_gradients(model, x, target, steps=24, baseline=0.0, chunk=8):
    """IG with a constant baseline. x (B,C,H,W), target (B,). Returns (B,C,H,W)."""
    model.eval(); out = []
    for i in range(0, len(x), chunk):
        xb, tb = x[i:i + chunk].float(), target[i:i + chunk]; base = torch.full_like(xb, baseline); acc = torch.zeros_like(xb)
        for a in (torch.arange(steps, device=xb.device) + 0.5) / steps:
            z = (base + a * (xb - base)).requires_grad_(True)
            g, = torch.autograd.grad(model(z).gather(1, tb[:, None]).sum(), z); acc += g
        out.append(((xb - base) * acc / steps).detach())
    return torch.cat(out)

@torch.no_grad()
def attention_stack(model, x):
    for b in model.blocks: b.keep_attn = True
    model(x)
    A = [b.last_attn.clone() for b in model.blocks]
    for b in model.blocks: b.keep_attn = False
    return A                                                          # list of (B, n, n), heads averaged

@torch.no_grad()
def rollout_matrix(model, x):
    """R[b, t, j]: how much input token j reaches final token t (0.5 W + 0.5 I per layer, multiplied over layers)."""
    A = attention_stack(model, x); n = A[0].shape[-1]; I = torch.eye(n, device=x.device)
    R = I.expand(x.shape[0], n, n).clone()
    for a in A:
        a = 0.5 * a + 0.5 * I; a = a / a.sum(-1, keepdim=True); R = a @ R
    return R

@torch.no_grad()
def rollout_map(model, x, token_weights=None):
    """Attention rollout read out with token weights (default uniform). Contribution-weighted rollout passes relu(contribution)."""
    R = rollout_matrix(model, x); B, n, _ = R.shape
    w = torch.full((B, n), 1.0 / n, device=x.device) if token_weights is None else token_weights.flatten(1).clamp(min=0)
    w = w / (w.sum(1, keepdim=True) + 1e-9)
    return (w[:, None, :] @ R)[:, 0].view(B, model.nf, model.nt)

def token_to_input(m):                                                # (B,nf,nt) -> (B,1,4nf,4nt)
    return m.repeat_interleave(4, 1).repeat_interleave(4, 2)[:, None]

def input_to_token(a):                                                # (B,C,H,W) attribution -> (B,nf,nt) by |.| mean over channels, 4x4 sum
    a = a.abs().sum(1); B, H, W = a.shape
    return a.view(B, H // 4, 4, W // 4, 4).sum((2, 4))

def mel_row_centres(fmax=2000.0, n=64):
    import librosa; return librosa.mel_frequencies(n_mels=n + 2, fmin=0.0, fmax=fmax)[1:-1]
def cqt_row_centres(n=64, fmin=32.7, bpo=12): return fmin * 2.0 ** (np.arange(n) / bpo)
def token_row_hz(nf=16):
    try: mel = mel_row_centres()
    except Exception: mel = np.linspace(0, 2000, 64)
    cqt = cqt_row_centres()
    return pd.DataFrame([dict(token_row=r, mel_lo_hz=round(float(mel[4 * r])), mel_hi_hz=round(float(mel[4 * r + 3])),
                              cqt_lo_hz=round(float(cqt[4 * r]), 1), cqt_hi_hz=round(float(cqt[4 * r + 3]), 1)) for r in range(nf)])
