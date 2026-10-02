"""Panorama stitcher (order independent, full resolution, lossless output).

Pipeline:
  1. SIFT features, matched between EVERY pair of images (filename order is irrelevant)
  2. A maximum spanning tree of the match graph gives the stitching order
  3. Several alignment models are built, scored on the same matches, best one wins
  4. Winner is warped with Lanczos at >= native pixel density, seams are found
     automatically and images are joined with multi-band blending
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from scipy.optimize import least_squares

VALID_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
RATIO = 0.75           # Lowe ratio test
RANSAC_PX = 4.0        # pair-matching tolerance (pixels)
MIN_INLIERS = 20
MAX_CANVAS_PX = 150e6  # safety cap on panorama size


# ======================= folder selection / loading =======================

def pick_folders():
    """Pop-up dialogs: input folder, then output folder (or use command-line args)."""
    if len(sys.argv) >= 3:
        return sys.argv[1], sys.argv[2]
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    src = filedialog.askdirectory(title="Select folder containing the images")
    if not src:
        sys.exit("No input folder selected.")
    dst = filedialog.askdirectory(title="Select folder to save the panorama")
    if not dst:
        sys.exit("No output folder selected.")
    return src, dst


def load_images(folder):
    """Read every supported image at full resolution."""
    images = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(VALID_EXT):
            img = cv2.imread(os.path.join(folder, name), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Skipping unreadable file: {name}")
            else:
                images.append(img)
    return images


# ======================= matching and image graph =======================

def match_pair(fa, fb):
    """Inlier point pairs (A pts, B pts, H A->B) or None if the images do not overlap."""
    (kp_a, des_a), (kp_b, des_b) = fa, fb
    if des_a is None or des_b is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    pa = np.float32([kp_a[m.queryIdx].pt for m in good])
    pb = np.float32([kp_b[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, RANSAC_PX, maxIters=10000, confidence=0.999)
    if H is None or mask.sum() < max(MIN_INLIERS, 8 + 0.3 * len(good)):
        return None  # too few consistent matches to trust this pair
    keep = mask.ravel() == 1
    return pa[keep], pb[keep], H


def match_all(images):
    """Match every image against every other one -> {(i, j): (pts_i, pts_j, H_i->j)}."""
    sift = cv2.SIFT_create(nfeatures=12000)
    feats = [sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None) for im in images]
    pairs = {}
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            res = match_pair(feats[i], feats[j])
            if res:
                pairs[(i, j)] = res
                print(f"  match {i + 1}-{j + 1}: {len(res[0])} inliers")
    return pairs


def stitch_tree(n, pairs):
    """Max-spanning tree on inlier counts. Returns (root, [(child, parent), ...])."""
    weight = {}
    for (i, j), (pa, _, _) in pairs.items():
        weight.setdefault(i, {})[j] = len(pa)
        weight.setdefault(j, {})[i] = len(pa)
    if not weight:
        return None, []
    start = max(weight, key=lambda k: sum(weight[k].values()))
    seen, edges = {start}, []
    while True:  # Prim: always add the strongest edge leaving the tree
        best = max(((weight[a][b], a, b) for a in seen for b in weight[a] if b not in seen), default=None)
        if best is None:
            break
        _, a, b = best
        seen.add(b)
        edges.append((a, b))
    adj = {k: [] for k in seen}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    def depth(r):  # BFS: order + tree height from r
        order, dist, queue = [], {r: 0}, [r]
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    order.append((nb, cur))
                    queue.append(nb)
        return max(dist.values()), order

    root = min(adj, key=lambda r: depth(r)[0])  # tree centre => least error build-up
    return root, depth(root)[1]


def pair_points(pairs, i, j):
    """Matched points of (i, j) in that order, whichever way they were stored."""
    if (i, j) in pairs:
        return pairs[(i, j)][0], pairs[(i, j)][1]
    return pairs[(j, i)][1], pairs[(j, i)][0]


def pair_homography(pairs, i, j):
    """Homography taking pixels of i to pixels of j."""
    return pairs[(i, j)][2] if (i, j) in pairs else np.linalg.inv(pairs[(j, i)][2])


# ======================= alignment models =======================
# Every model exposes fwd(i, pts) image->panorama plane and inv(i, pts) the reverse.

def apply_h(H, p):
    """Apply a 3x3 homography to Nx2 points."""
    w = p @ H[2, :2] + H[2, 2]
    return (p @ H[:2, :2].T + H[:2, 2]) / w[:, None]


def normalizer(img):
    """Maps pixels to ~[-0.5, 0.5] coordinates (keeps the optimiser well conditioned)."""
    h, w = img.shape[:2]
    s = max(w, h)
    return np.array([[1 / s, 0, -w / 2 / s], [0, 1 / s, -h / 2 / s], [0, 0, 1]])


class HomographyModel:
    """Planar panorama: one homography per image into a common frame."""

    def __init__(self, Hs):
        self.Hs = Hs
        self.Hinv = [None if H is None else np.linalg.inv(H) for H in Hs]
        self.used = [i for i, H in enumerate(Hs) if H is not None]

    def fwd(self, i, p):
        return apply_h(self.Hs[i], p)

    def inv(self, i, q):
        return apply_h(self.Hinv[i], q)


class RotationModel:
    """Camera-rotation panorama projected on a cylinder or a sphere (no edge stretching)."""

    def __init__(self, images, Rs, fn, kind, G=None):
        self.kind = kind
        self.R = Rs
        self.used = [i for i, R in enumerate(Rs) if R is not None]
        self.c = [np.array([im.shape[1] / 2, im.shape[0] / 2]) for im in images]
        self.f = [fn[i] * max(im.shape[:2]) for i, im in enumerate(images)]
        self.fref = float(np.mean([self.f[i] for i in self.used]))
        self.G = G if G is not None else [np.eye(3)] * len(images)
        self.Ginv = [np.linalg.inv(g) for g in self.G]

    def fwd(self, i, p):
        x = (apply_h(self.G[i], p) - self.c[i]) / self.f[i]
        d = np.c_[x, np.ones(len(x))] @ self.R[i]            # = R^T x
        theta = np.arctan2(d[:, 0], d[:, 2])
        rad = np.hypot(d[:, 0], d[:, 2])
        v = d[:, 1] / rad if self.kind == "cylinder" else np.arctan2(d[:, 1], rad)
        return self.fref * np.c_[theta, v]

    def inv(self, i, q):
        th, v = q[:, 0] / self.fref, q[:, 1] / self.fref
        if self.kind == "cylinder":
            ray = np.c_[np.sin(th), v, np.cos(th)]
        else:
            ray = np.c_[np.cos(v) * np.sin(th), np.sin(v), np.cos(v) * np.cos(th)]
        cam = ray @ self.R[i].T
        ok = cam[:, 2] > 1e-6
        z = np.where(ok, cam[:, 2], 1.0)
        p = apply_h(self.Ginv[i], cam[:, :2] / z[:, None] * self.f[i] + self.c[i])
        p[~ok] = -1e5                                         # behind the camera
        return p


def transfer_residuals(model, pairs):
    """Where do matched points of one image land in the other (pixels), both directions."""
    res = []
    for (i, j), (pa, pb, _) in pairs.items():
        if i in model.used and j in model.used:
            res.append((model.inv(j, model.fwd(i, pa)) - pb).ravel())
            res.append((model.inv(i, model.fwd(j, pb)) - pa).ravel())
    return np.clip(np.concatenate(res), -500, 500)


# ----- candidate 1: homographies chained along the tree -----

def chain_homographies(n, root, order, pairs):
    Hs = [None] * n
    Hs[root] = np.eye(3)
    for child, parent in order:
        Hs[child] = Hs[parent] @ pair_homography(pairs, child, parent)
    return Hs


# ----- candidate 2: all homographies refined together -----

def refine_homographies(images, Hs, pairs, root):
    """Jointly minimise the transfer error of ALL matches (removes chain drift)."""
    free = [i for i, H in enumerate(Hs) if H is not None and i != root]
    N = [normalizer(im) for im in images]
    to_norm = lambda i: N[root] @ Hs[i] @ np.linalg.inv(N[i])
    x0 = np.concatenate([(to_norm(i) / to_norm(i)[2, 2]).ravel()[:8] for i in free])

    def build(x):
        out = list(Hs)
        for k, i in enumerate(free):
            Hn = np.append(x[8 * k:8 * k + 8], 1.0).reshape(3, 3)
            out[i] = np.linalg.inv(N[root]) @ Hn @ N[i]
        return out

    sol = least_squares(lambda x: transfer_residuals(HomographyModel(build(x)), pairs),
                        x0, loss="soft_l1", f_scale=2.0, max_nfev=60)
    return build(sol.x)


# ----- candidates 3-5: rotation model (cylinder / sphere / sphere + local fixes) -----

def orthogonal(M):
    """Nearest rotation matrix."""
    U, _, Vt = np.linalg.svd(M / np.cbrt(np.linalg.det(M)))
    return U @ Vt


def rotations_from_tree(images, fn, root, order, pairs):
    """Initial camera rotations: R_child = M^T R_parent with M = orth(K_p^-1 H K_c)."""
    K = []
    for im in images:
        f = fn * max(im.shape[:2])
        K.append(np.array([[f, 0, im.shape[1] / 2], [0, f, im.shape[0] / 2], [0, 0, 1]]))
    Rs = [None] * len(images)
    Rs[root] = np.eye(3)
    for child, parent in order:
        H = pair_homography(pairs, child, parent)
        M = orthogonal(np.linalg.inv(K[parent]) @ H @ K[child])
        Rs[child] = M.T @ Rs[parent]
    return Rs


def pick_focal(images, root, order, pairs):
    """Scan focal lengths (relative to image size); keep the one with lowest error."""
    best = (np.inf, 1.0)
    for fn in np.geomspace(0.4, 3.5, 40):
        Rs = rotations_from_tree(images, fn, root, order, pairs)
        r = transfer_residuals(RotationModel(images, Rs, [fn] * len(images), "cylinder"), pairs)
        best = min(best, (np.mean(np.minimum(np.abs(r), 30)), fn))
    return best[1]


def bundle_adjust(images, Rs, fn0, pairs, root):
    """Refine all rotations + focal lengths on every match (focals tied loosely together)."""
    n = len(images)
    free = [i for i in range(n) if Rs[i] is not None and i != root]
    used = [i for i in range(n) if Rs[i] is not None]
    x0 = np.concatenate([cv2.Rodrigues(Rs[i])[0].ravel() for i in free] + [[fn0] * len(used)])

    def unpack(x):
        R = list(Rs)
        for k, i in enumerate(free):
            R[i] = cv2.Rodrigues(x[3 * k:3 * k + 3])[0]
        fn = [fn0] * n
        for k, i in enumerate(used):
            fn[i] = x[3 * len(free) + k]
        return R, fn

    def residual(x):
        R, fn = unpack(x)
        r = transfer_residuals(RotationModel(images, R, fn, "cylinder"), pairs)
        f = x[3 * len(free):]
        return np.append(r, 200.0 * (f / f.mean() - 1))      # soft "same lens" prior

    sol = least_squares(residual, x0, loss="soft_l1", f_scale=2.0, max_nfev=80)
    return unpack(sol.x)


def straighten(Rs):
    """Level the horizon: rotate the world so the cameras pan around the vertical axis."""
    used = [i for i, R in enumerate(Rs) if R is not None]
    xs = np.array([Rs[i].T[:, 0] for i in used])              # camera right vectors in world
    ys = np.array([Rs[i].T[:, 1] for i in used])              # camera down vectors in world
    up = np.linalg.svd(xs.T @ xs)[0][:, 2]                    # normal of the panning plane
    up = up if up @ ys.mean(0) > 0 else -up
    x = xs.mean(0) - (xs.mean(0) @ up) * up
    x /= np.linalg.norm(x)
    Rg = np.stack([x, up, np.cross(x, up)])                   # world -> straightened world
    return [None if R is None else R @ Rg.T for R in Rs]


def local_fixes(images, model, pairs, root):
    """Per-image homography correction on top of the rotation model (absorbs parallax/lens error)."""
    free = [i for i in model.used if i != root]
    N = [normalizer(im) for im in images]
    corners = lambda im: np.array([[0, 0], [im.shape[1], 0], [im.shape[1], im.shape[0]], [0, im.shape[0]]], float)

    def build(x):
        G = [np.eye(3)] * len(images)
        for k, i in enumerate(free):
            Gn = np.append(x[8 * k:8 * k + 8], 1.0).reshape(3, 3)
            G[i] = np.linalg.inv(N[i]) @ Gn @ N[i]
        return G

    ident = np.array([1, 0, 0, 0, 1, 0, 0, 0], float)
    x0 = np.tile(ident, len(free))

    def residual(x):
        G = build(x)
        m = RotationModel(images, model.R, [model.f[i] / max(images[i].shape[:2]) for i in range(len(images))],
                          model.kind, G)
        drift = [0.5 * (apply_h(G[i], corners(images[i])) - corners(images[i])).ravel() for i in free]
        return np.concatenate([transfer_residuals(m, pairs)] + drift)

    sol = least_squares(residual, x0, loss="soft_l1", f_scale=2.0, max_nfev=40)
    return build(sol.x)


# ======================= candidate comparison =======================

def sample_scales(images, model):
    """Panorama-pixels per source-pixel at a grid of points in every used image."""
    out = []
    for i in model.used:
        h, w = images[i].shape[:2]
        gx, gy = np.meshgrid(np.linspace(0, w, 15), np.linspace(0, h, 15))
        P = np.c_[gx.ravel(), gy.ravel()]
        q = model.fwd(i, P)
        a, b = model.fwd(i, P + [1, 0]) - q, model.fwd(i, P + [0, 1]) - q
        out.append(np.sqrt(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])))
    return np.concatenate(out)


def score_model(images, model, pairs):
    """(alignment error px [robust], stretch) - lower is better for both."""
    r = np.abs(transfer_residuals(model, pairs))
    err = float(np.sqrt(np.mean(np.minimum(r, 10.0) ** 2)))
    sc = sample_scales(images, model)
    stretch = float(np.percentile(sc, 98) / np.percentile(sc, 2))   # 1.0 = uniform pixel density
    return err, stretch


def choose_model(images, candidates, pairs):
    """Print a comparison table and return the best candidate."""
    rows = []
    for name, model in candidates:
        err, stretch = score_model(images, model, pairs)
        rows.append((err * (1 + 0.25 * (stretch - 1)), name, model, err, stretch))
        print(f"  {name:<18} align error {err:5.2f}px   pixel-density spread x{stretch:4.2f}")
    best = min(rows, key=lambda r: r[0])
    print(f"  -> best model: {best[1]}")
    return best[2]


def build_candidates(images, pairs, root, order):
    n = len(images)
    cands = []
    Hc = chain_homographies(n, root, order, pairs)
    cands.append(("homography-chain", HomographyModel(Hc)))
    cands.append(("homography-global", HomographyModel(refine_homographies(images, Hc, pairs, root))))
    fn = pick_focal(images, root, order, pairs)
    Rs, fns = bundle_adjust(images, rotations_from_tree(images, fn, root, order, pairs), fn, pairs, root)
    Rs = straighten(Rs)
    cyl = RotationModel(images, Rs, fns, "cylinder")
    cands.append(("cylinder", cyl))
    cands.append(("sphere", RotationModel(images, Rs, fns, "sphere")))
    G = local_fixes(images, cyl, pairs, root)
    cands.append(("cylinder+local", RotationModel(images, Rs, fns, "cylinder", G)))
    return cands


# ======================= rendering =======================

def make_canvas(images, model):
    """Scale + offset + size so no region is downsampled (pixel density >= source)."""
    scales = sample_scales(images, model)
    S = 1.0 / np.percentile(scales, 2)
    pts = []
    for i in model.used:
        h, w = images[i].shape[:2]
        t = np.linspace(0, 1, 300)
        edge = np.vstack([np.c_[t * w, 0 * t], np.c_[t * w, 0 * t + h], np.c_[0 * t, t * h], np.c_[0 * t + w, t * h]])
        pts.append(model.fwd(i, edge))
    pts = np.vstack(pts)
    lo, hi = pts.min(0), pts.max(0)
    if np.prod((hi - lo) * S) > MAX_CANVAS_PX:
        S *= np.sqrt(MAX_CANVAS_PX / np.prod((hi - lo) * S))
        print("  (panorama capped at the maximum size)")
    return S, lo


def warp_image(img, model, i, S, lo):
    """Warp one image with Lanczos into its own bounding box. Returns (patch, mask, corner)."""
    h, w = img.shape[:2]
    t = np.linspace(0, 1, 300)
    edge = np.vstack([np.c_[t * w, 0 * t], np.c_[t * w, 0 * t + h], np.c_[0 * t, t * h], np.c_[0 * t + w, t * h]])
    box = (model.fwd(i, edge) - lo) * S
    x0, y0 = np.floor(box.min(0)).astype(int)
    x1, y1 = np.ceil(box.max(0)).astype(int) + 1
    gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    q = np.c_[gx.ravel() + 0.0, gy.ravel() + 0.0] / S + lo
    p = model.inv(i, q)
    mx = p[:, 0].reshape(gx.shape).astype(np.float32)
    my = p[:, 1].reshape(gx.shape).astype(np.float32)
    patch = cv2.remap(img, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    inside = ((mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)).astype(np.uint8) * 255
    inside = cv2.erode(inside, np.ones((5, 5), np.uint8))     # keep Lanczos edge ringing out
    return patch, inside, (int(x0), int(y0))


def find_seams(patches, masks, corners):
    """Cut each overlap along a low-contrast path (done on a reduced copy for speed)."""
    sc = min(1.0, np.sqrt(1.5e6 / max(p.shape[0] * p.shape[1] for p in patches)))
    small = [cv2.resize(p, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA).astype(np.float32) for p in patches]
    small_masks = [cv2.UMat(cv2.resize(m, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)) for m in masks]
    small_corners = [(int(x * sc), int(y * sc)) for x, y in corners]
    cv2.detail_DpSeamFinder("COLOR_GRAD").find(small, small_corners, small_masks)
    out = []
    for m, sm in zip(masks, small_masks):
        grown = cv2.dilate(sm.get(), None)
        out.append(cv2.bitwise_and(cv2.resize(grown, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR), m))
    return out


def equalize_exposure(patches, masks, corners):
    """Per-image, per-channel gain so overlapping areas have equal mean brightness."""
    n = len(patches)
    rows, rhs = [], []
    for a in range(n):
        for b in range(a + 1, n):
            (xa, ya), (xb, yb) = corners[a], corners[b]
            x0, y0 = max(xa, xb), max(ya, yb)
            x1 = min(xa + patches[a].shape[1], xb + patches[b].shape[1])
            y1 = min(ya + patches[a].shape[0], yb + patches[b].shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            sa = (slice(y0 - ya, y1 - ya), slice(x0 - xa, x1 - xa))
            sb = (slice(y0 - yb, y1 - yb), slice(x0 - xb, x1 - xb))
            both = (masks[a][sa] > 0) & (masks[b][sb] > 0)
            if both.sum() < 2000:
                continue
            ma = patches[a][sa][both].mean(0) + 1.0
            mb = patches[b][sb][both].mean(0) + 1.0
            row = np.zeros(n)
            row[a], row[b] = 1, -1
            rows.append(row)
            rhs.append(np.log(mb / ma))                      # log g_a - log g_b = log(mb/ma)
    if not rows:
        return patches
    A = np.vstack(rows + [0.05 * np.eye(n)])                  # weak pull towards gain 1
    B = np.vstack(rhs + [np.zeros((n, 3))])
    gains = np.exp(np.linalg.lstsq(A, B, rcond=None)[0])      # n x 3
    return [np.clip(p * g, 0, 255).astype(np.uint8) for p, g in zip(patches, gains)]


def render(images, model):
    """Warp everything, equalise exposure, find seams, multi-band blend."""
    S, lo = make_canvas(images, model)
    warped = [warp_image(images[i], model, i, S, lo) for i in model.used]
    patches, masks, corners = (list(z) for z in zip(*warped))
    sizes = [(p.shape[1], p.shape[0]) for p in patches]
    print(f"  panorama: {int(max(c[0] + s[0] for c, s in zip(corners, sizes)))} x "
          f"{int(max(c[1] + s[1] for c, s in zip(corners, sizes)))} px")

    patches = equalize_exposure(patches, masks, corners)

    masks = find_seams(patches, masks, corners)
    roi = cv2.detail.resultRoi(corners, sizes)
    bands = max(1, int(np.ceil(np.log(np.sqrt(roi[2] * roi[3]) * 0.05) / np.log(2)) - 1))
    blender = cv2.detail_MultiBandBlender(0, bands)
    blender.prepare(roi)
    for p, m, c in zip(patches, masks, corners):
        blender.feed(p.astype(np.int16), m, c)
    result, _ = blender.blend(None, None)
    return np.clip(result, 0, 255).astype(np.uint8)


# ======================= main =======================

def build_panorama(images):
    print("Matching images...")
    pairs = match_all(images)
    root, order = stitch_tree(len(images), pairs)
    if root is None or not order:
        raise RuntimeError("Not enough overlap between images to stitch.")
    skipped = set(range(len(images))) - {root} - {c for c, _ in order}
    if skipped:
        print(f"  not connected to the others, skipped: {sorted(k + 1 for k in skipped)}")
    print("Comparing alignment models...")
    model = choose_model(images, build_candidates(images, pairs, root, order), pairs)
    print("Rendering...")
    return render(images, model)


def main():
    src, dst = pick_folders()
    images = load_images(src)
    if len(images) < 2:
        sys.exit("Need at least 2 readable images in the folder.")
    print(f"Stitching {len(images)} images...")
    try:
        pano = build_panorama(images)
    except RuntimeError as err:
        sys.exit(str(err))
    out = os.path.join(dst, "panorama.png")
    cv2.imwrite(out, pano, [cv2.IMWRITE_PNG_COMPRESSION, 3])  # PNG = lossless
    print(f"Saved: {out}")
    if len(sys.argv) < 3:
        messagebox.showinfo("Panorama", f"Saved to:\n{out}")


if __name__ == "__main__":
    main()
