"""
video_score.py  -  Stage 1 scoring for the in-car video frames.

WHAT THIS DOES (read top to bottom, it follows this order):
  1. CLEAN your hand labels (merge CSVs, drop mouse-slip boxes, split long diagonal lines).
  2. DETECT markings with classical computer vision in a bird's-eye view.
  3. SCORE: match detections to your labels (IoU) -> precision / recall / F1.

Run (PowerShell, from the folder that holds this script):
  python video_score.py --frames ..\frames --labels labels_3.csv labels_seg_3.csv labels_seg_45.csv ^
         --calib calibration.json --segs seg1 seg2 --out results_dev
  python video_score.py ... --segs seg3 seg4 seg5 --out results_test      (ONLY when finished tuning)
"""
import argparse, csv, json, math, os, glob, collections
import cv2, numpy as np

# ---------------- settings you can tune (on seg1/seg2 ONLY) ----------------
IOU_THRESHOLD   = 0.3     # how much a prediction must overlap a label to count as a hit
MIN_W, MIN_H    = 30, 20  # labels smaller than this are treated as mouse slips
ZONE_Y          = 992     # markings whose centre is above this are not scored (too far away)
LONG_W, LONG_H  = 600, 200  # a label wider AND taller than this is a "long diagonal line"
PIECE_H         = 110     # each split piece of a long line is about this tall (pixels)
RING_DARK_MAX   = 0.12    # kerb stripes sit next to BLACK paint; real markings sit on mid-grey road. Max share of dark pixels around a blob
DARK_LEVEL      = 55      # grey level below which a pixel counts as 'black'
KERB_MAX_AREA   = 700     # the kerb test only applies to small blobs, so long edge lines beside a dark kerb survive
TOPHAT_K        = 31      # bird's-eye: size of the "local background" window (bigger = ignores wider things)
BRIGHT_MIN      = 40      # how much brighter than local background paint must be (0-255)
MIN_AREA        = 40      # bird's-eye component minimum area (pixels)
MIN_ELONG       = 1.3     # big components must be longer than wide by this ratio
SMALL_AREA      = 500     # components smaller than this (dashes) skip the elongation test: far dashes look square
# ----------------------------------------------------------------------------

# ===================== 1. LABEL CLEANING =====================
def load_labels(paths):
    """Reads several label CSVs IN THE ORDER GIVEN. If a later file contains a frame that an earlier file
    also had, the later file REPLACES that frame's labels completely (boxes and no_marking flag).
    So put your corrections file last."""
    boxes = {}
    nomark = set()
    for p in paths:
        fresh = collections.defaultdict(list); fresh_nomark = set()
        for r in csv.DictReader(open(p, newline='')):
            f = r['filename']
            if r.get('no_marking', '').upper() == 'TRUE':
                fresh_nomark.add(f); fresh.setdefault(f, []); continue
            if r['x'] == '': continue
            x, y, w, h = (int(float(r[k])) for k in ('x', 'y', 'width', 'height'))
            x0, y0 = max(0, x), max(0, y)                 # clip boxes that poke 1-2 px outside the frame
            w, h = w - (x0 - x), h - (y0 - y)
            if w > 0 and h > 0: fresh[f].append((x0, y0, w, h))
        for f, bs in fresh.items():
            boxes[f] = bs                                  # replace, do not append
            nomark.discard(f)
        nomark |= fresh_nomark
    return boxes, nomark

def split_long(box, vpx):
    """A long diagonal line fills its box corner-to-corner. Because every lane line points at the
    vanishing point, we know WHICH diagonal: right of the VP it runs top-left -> bottom-right,
    left of the VP it runs top-right -> bottom-left. Cut it into small pieces along that line."""
    x, y, w, h = box
    if x < vpx < x + w:            # straddles the centre: ambiguous, skip (returned as None)
        return None
    k = max(2, round(h / PIECE_H))
    pieces = []
    for i in range(k):
        y0, y1 = y + h * i / k, y + h * (i + 1) / k
        t0, t1 = i / k, (i + 1) / k
        if x >= vpx:   xa, xb = x + w * t0, x + w * t1           # right side
        else:          xa, xb = x + w * (1 - t0), x + w * (1 - t1)  # left side
        px0, px1 = min(xa, xb), max(xa, xb)
        pieces.append((int(px0), int(y0), max(1, int(px1 - px0)), max(1, int(y1 - y0))))
    return pieces

def clean_labels(boxes, vpx, stats):
    out = {}
    for f, bs in boxes.items():
        keep = []
        for b in bs:
            x, y, w, h = b
            if w < MIN_W or h < MIN_H:
                stats['tiny_dropped'] += 1; continue
            if w > LONG_W and h > LONG_H:
                parts = split_long(b, vpx)
                if parts is None:
                    stats['straddle_dropped'] += 1; continue
                stats['long_split'] += 1; keep += parts; continue
            keep.append(b)
        zone = [b for b in keep if b[1] + b[3] / 2 >= ZONE_Y]
        stats['above_zone_ignored'] += len(keep) - len(zone)
        out[f] = zone
    return out

# ===================== 2a. VEHICLE MASK =====================
VEHICLE_CLASSES = (0, 2, 3, 5, 7)   # COCO: person, car, motorcycle, bus, truck
VEHICLE_CONF    = 0.25
VEHICLE_MARGIN  = 10                # grow each vehicle box a little (bumpers, tail lights, shadow edges)

class VehicleFinder:
    """Cars have bright bumpers, plates and tail lights that look like paint. A pretrained YOLO model
    (trained on the COCO dataset, which knows 'car', 'bus', 'truck'...) finds vehicles; we then ignore
    any marking detection whose centre lies inside a vehicle box. It is NOT trained on our road markings."""
    def __init__(self, weights):
        from ultralytics import YOLO
        self.model = YOLO(weights)
    def find(self, img):
        r = self.model.predict(img, conf=VEHICLE_CONF, classes=list(VEHICLE_CLASSES), verbose=False, imgsz=1280)[0]
        out = []
        for x0, y0, x1, y1 in r.boxes.xyxy.cpu().numpy():
            out.append((int(x0) - VEHICLE_MARGIN, int(y0) - VEHICLE_MARGIN, int(x1) + VEHICLE_MARGIN, int(y1) + VEHICLE_MARGIN))
        return out

def drop_in_vehicles(pred, vehicles):
    keep = []
    for (x, y, w, h) in pred:
        cx, cy = x + w / 2, y + h / 2
        if any(a <= cx <= c and b <= cy <= d for a, b, c, d in vehicles): continue
        keep.append((x, y, w, h))
    return keep

# ===================== 2. DETECTOR =====================
class Detector:
    def __init__(self, calib_path):
        c = json.load(open(calib_path))
        W, Hh = c['dst_size']
        src = np.float32(c['src'])
        dst = np.float32([[0, 0], [W - 1, 0], [W - 1, Hh - 1], [0, Hh - 1]])
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.Hinv = np.linalg.inv(self.H)
        self.size = (W, Hh)
        self.vpx = c['vanishing_point'][0]

    def detect(self, img):
        bev = cv2.warpPerspective(img, self.H, self.size, flags=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        # White-hat: "pixel minus its local background". Keeps small bright things (paint),
        # removes smooth brightness changes (shade, sky reflection, road colour).
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_K, TOPHAT_K))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
        white = tophat > BRIGHT_MIN
        # Yellow paint: hue band in HSV, with decent saturation
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (15, 90, 110), (35, 255, 255)) > 0
        mask = (white | yellow).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < MIN_AREA: continue
            if a > SMALL_AREA and max(w, h) / max(1, min(w, h)) < MIN_ELONG: continue   # big round blobs are not paint
            # kerb test: look at a thin ring of pixels around the blob
            x0r, y0r = max(0, x - 8), max(0, y - 8); x1r, y1r = min(self.size[0], x + w + 8), min(self.size[1], y + h + 8)
            sub = lab[y0r:y1r, x0r:x1r] == i
            ring = cv2.dilate(sub.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            ring &= ~sub
            if a < KERB_MAX_AREA and ring.sum() > 0 and (gray[y0r:y1r, x0r:x1r][ring] < DARK_LEVEL).mean() > RING_DARK_MAX: continue
            corners = np.float32([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
            o = cv2.perspectiveTransform(corners, self.Hinv).reshape(-1, 2)
            x0, y0 = o.min(0); x1, y1 = o.max(0)
            bw, bh = x1 - x0, y1 - y0
            if bw < 20 or bh < 10: continue   # smaller than any real marking
            if y0 + bh / 2 < ZONE_Y: continue
            box = (int(x0), int(y0), int(bw), int(bh))
            if bw > LONG_W and bh > LONG_H:          # long diagonal line: cut it like we cut the labels
                parts = split_long(box, self.vpx)
                if parts: boxes += [p for p in parts if p[1] + p[3] / 2 >= ZONE_Y]
                continue
            boxes.append(box)
        return boxes

# ===================== 3. SCORING =====================
def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy; u = a[2] * a[3] + b[2] * b[3] - i
    return i / u if u else 0.0

def match(pred, gt):
    """Greedy one-to-one matching, best IoU first. Returns (TP, FP, FN)."""
    pairs = sorted(((iou(p, g), pi, gi) for pi, p in enumerate(pred) for gi, g in enumerate(gt)), reverse=True)
    up, ug = set(), set()
    for s, pi, gi in pairs:
        if s < IOU_THRESHOLD: break
        if pi in up or gi in ug: continue
        up.add(pi); ug.add(gi)
    return len(up), len(pred) - len(up), len(gt) - len(ug), up, ug

def match_loose(pred, gt):
    """Forgiving rule for hand-drawn boxes: a detection is correct if its CENTRE lies inside one of your
    boxes (or IoU>=0.3). A label is found if any detection hits it. Several fragments inside one label
    are not punished as false alarms."""
    hit_p, hit_g = set(), set()
    for pi, p in enumerate(pred):
        cx, cy = p[0] + p[2] / 2, p[1] + p[3] / 2
        for gi, g in enumerate(gt):
            m = 6
            inside = g[0] - m <= cx <= g[0] + g[2] + m and g[1] - m <= cy <= g[1] + g[3] + m
            if inside or iou(p, g) >= IOU_THRESHOLD:
                hit_p.add(pi); hit_g.add(gi)
    return len(hit_g), len(pred) - len(hit_p), len(gt) - len(hit_g), hit_p, hit_g

def draw(img, pred, gt, up, ug):
    v = img.copy()
    for gi, (x, y, w, h) in enumerate(gt):      # green = found label, yellow = MISSED label
        cv2.rectangle(v, (x, y), (x + w, y + h), (0, 255, 0) if gi in ug else (0, 220, 255), 3)
    for pi, (x, y, w, h) in enumerate(pred):    # blue = correct detection, red = FALSE alarm
        cv2.rectangle(v, (x, y), (x + w, y + h), (255, 120, 0) if pi in up else (0, 0, 255), 2)
    cv2.line(v, (0, ZONE_Y), (v.shape[1], ZONE_Y), (0, 200, 255), 1)
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True); ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--calib', default='calibration.json'); ap.add_argument('--segs', nargs='+', required=True)
    ap.add_argument('--out', default='results'); ap.add_argument('--save', type=int, default=12)
    ap.add_argument('--vehicles', default='', help='path to yolov8n.pt; if given, detections inside cars/buses/people are ignored')
    ap.add_argument('--show', nargs='*', default=[], help='frame names to always save a picture for, e.g. seg2_0033.jpg')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    det = Detector(a.calib)
    vf = VehicleFinder(a.vehicles) if a.vehicles else None
    raw, nomark = load_labels(a.labels)
    stats = collections.Counter()
    gt_all = clean_labels(raw, det.vpx, stats)
    print("Label cleaning:", dict(stats))
    print("Vehicle filter:", "ON (" + a.vehicles + ")" if vf else "OFF  (add --vehicles yolov8n.pt to turn it on)")
    removed = 0

    tot = collections.defaultdict(lambda: [0, 0, 0]); tot_l = collections.defaultdict(lambda: [0, 0, 0]); rows = []; saved = 0
    for f in sorted(gt_all):
        seg = f.split('_')[0]
        if seg not in a.segs: continue
        path = os.path.join(a.frames, f)
        img = cv2.imread(path)
        if img is None: print("missing frame:", path); continue
        pred = det.detect(img); gt = gt_all[f]
        if vf:
            n0 = len(pred); pred = drop_in_vehicles(pred, vf.find(img)); removed += n0 - len(pred)
        tp, fp, fn, up, ug = match(pred, gt)                 # strict (IoU)
        ltp, lfp, lfn, lup, lug = match_loose(pred, gt)      # loose (centre-in-box)
        for key in (seg, 'ALL'):
            tot[key][0] += tp; tot[key][1] += fp; tot[key][2] += fn
            tot_l[key][0] += ltp; tot_l[key][1] += lfp; tot_l[key][2] += lfn
        up, ug = lup, lug   # pictures use the loose rule
        rows.append((f, len(gt), len(pred), ltp, lfp, lfn))   # per_frame.csv uses the LOOSE rule
        if f in a.show or (saved < a.save and (lfp + lfn) > 0 and f not in nomark):
            cv2.imwrite(os.path.join(a.out, 'viz_' + f), cv2.resize(draw(img, pred, gt, up, ug), None, fx=0.5, fy=0.5)); saved += 1
    if vf: print(f"Vehicle filter removed {removed} detections")
    with open(os.path.join(a.out, 'per_frame.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['filename', 'gt', 'pred', 'tp', 'fp', 'fn']); w.writerows(rows)
    for title, T in (("STRICT (IoU >= 0.3)", tot), ("LOOSE (centre inside your box)  <- main number", tot_l)):
        print(f"\n{title}\nsegment   TP    FP    FN   precision  recall    F1")
        for k, (tp, fp, fn) in sorted(T.items()):
            p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
            f1 = 2 * p * r / (p + r) if p + r else 0
            print(f"{k:8} {tp:5} {fp:5} {fn:5}   {p:7.1%}  {r:7.1%}  {f1:6.1%}")
    print(f"\nPictures saved in {a.out}\\  (yellow=missed label, red=false alarm, green/blue=hit)")

if __name__ == '__main__':
    main()
