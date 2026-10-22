"""
find_vp.py  --  find the "vanishing point" of the road in dashcam frames.

WHAT IT DOES
  Road lines (lane dashes, kerbs, edge lines) are parallel in real life, but in
  a photo they appear to converge to ONE point on the horizon. That point is the
  vanishing point (VP). Knowing where it is tells us how the camera is tilted,
  which is what we need to build the bird's-eye (top-down) view.

HOW TO RUN (Windows, one-time setup:  pip install opencv-python numpy)
  python find_vp.py results_folder frame1.jpg frame2.jpg frame3.jpg

OUTPUT
  * Printed numbers per frame:
      vp        = [x, y] pixel position of the vanishing point
      segments  = how many straight line pieces were found
      inliers   = how many of those pieces agree on the vp
                  (rule of thumb: under ~30 inliers = don't trust that frame)
  * results_folder/<name>_vp.jpg : the frame with found lines in orange and
      the vanishing point as a red cross, so you can SEE if it is sensible.
"""
import sys, json, random, os
import cv2
import numpy as np

random.seed(0)
out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)   # create the results folder if missing
paths = sys.argv[2:]

def detect_segments(img):
    h, w = img.shape[:2]
    y0 = int(h * 0.66)                      # road lives in the bottom third
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # local-contrast "paint" mask: brighter than neighbourhood
    ad = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 61, -14)
    # yellow paint via hue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yel = cv2.inRange(hsv, (15, 70, 110), (35, 255, 255))
    mask = cv2.bitwise_or(ad, yel)
    mask[:y0, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(mask, 50, 150)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                           minLineLength=60, maxLineGap=25)
    keep = []
    if segs is None:
        return keep, mask
    for x1, y1, x2, y2 in segs.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        ang = abs(np.degrees(np.arctan2(dy, dx)))
        ang = min(ang, 180 - ang)           # 0 = horizontal
        if 12 <= ang <= 75:                 # converging lines, not junction bars / poles
            keep.append((x1, y1, x2, y2, float(np.hypot(dx, dy))))
    return keep, mask

def line_params(s):
    x1, y1, x2, y2, _ = s
    a, b = y2 - y1, x1 - x2
    c = -(a * x1 + b * y1)
    n = np.hypot(a, b)
    return a / n, b / n, c / n

def ransac_vp(segs, w, h, iters=4000, tol=12.0):
    best, best_score = None, 0
    L = [line_params(s) for s in segs]
    W = [s[4] for s in segs]
    n = len(segs)
    if n < 4:
        return None, 0
    for _ in range(iters):
        i, j = random.sample(range(n), 2)
        a1, b1, c1 = L[i]; a2, b2, c2 = L[j]
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            continue
        x = (b1 * c2 - b2 * c1) / det
        y = (c1 * a2 - c2 * a1) / det
        if not (0.25 * w < x < 0.75 * w and 0.35 * h < y < 0.75 * h):
            continue
        score = sum(W[k] for k in range(n) if abs(L[k][0] * x + L[k][1] * y + L[k][2]) < tol)
        if score > best_score:
            best, best_score = (x, y), score
    if best is None:
        return None, 0
    # refine with all inliers (weighted least squares)
    x, y = best
    inl = [k for k in range(n) if abs(L[k][0] * x + L[k][1] * y + L[k][2]) < tol]
    A = np.array([[L[k][0], L[k][1]] for k in inl]) * np.sqrt(np.array([W[k] for k in inl]))[:, None]
    B = -np.array([L[k][2] for k in inl]) * np.sqrt(np.array([W[k] for k in inl]))
    sol, *_ = np.linalg.lstsq(A, B, rcond=None)
    return (float(sol[0]), float(sol[1])), len(inl)

results = {}
for p in paths:
    img = cv2.imread(p)
    h, w = img.shape[:2]
    segs, mask = detect_segments(img)
    vp, ninl = ransac_vp(segs, w, h)
    name = os.path.splitext(os.path.basename(p))[0]
    results[name] = {"vp": vp, "segments": len(segs), "inliers": ninl}
    vis = img.copy()
    for x1, y1, x2, y2, _ in segs:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
    if vp:
        cv2.drawMarker(vis, (int(vp[0]), int(vp[1])), (0, 0, 255), cv2.MARKER_CROSS, 60, 4)
        cv2.line(vis, (0, int(vp[1])), (w, int(vp[1])), (0, 0, 255), 1)
    cv2.imwrite(f"{out_dir}/{name}_vp.jpg", cv2.resize(vis, (w // 2, h // 2)))
print(json.dumps(results, indent=1))
