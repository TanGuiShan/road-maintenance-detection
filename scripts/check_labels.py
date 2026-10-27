import csv, sys, collections, statistics
F = sys.argv[1]
allrows = list(csv.DictReader(open(F, newline='')))
rows = [r for r in allrows if r['x']]
B = [(r['filename'], int(r['x']), int(r['y']), int(r['width']), int(r['height'])) for r in rows]
VPX = 1263
def seg_of(f): return f.split('_')[0]
for seg in sorted({seg_of(b[0]) for b in B}):
    nums = sorted({int(b[0].split('_')[1].split('.')[0]) for b in B if seg_of(b[0]) == seg})
    consec = sum(1 for a, b in zip(nums, nums[1:]) if b - a == 1)
    print(f"{seg}: {len(nums)} frames, numbers {nums[0]}-{nums[-1]}, adjacent pairs (consecutive frames): {consec}")
far = [b for b in B if b[2] + b[4] / 2 < 992]
print("boxes above detector zone (centre y<992):", len(far), "of", len(B))
diag = [b for b in B if b[3] > 600 and b[4] > 200]
straddle = [b for b in diag if b[1] < VPX < b[1] + b[3]]
print("wide AND tall (>600 x >200):", len(diag), "| straddling the centre:", len(straddle))
flat = [b for b in B if b[3] > 600 and b[4] <= 200]
print("wide but flat (<=200 tall; horizontal lines, fine):", len(flat))
dash = [b for b in B if b[3] <= 250 and b[4] <= 130 and b[2] + b[4]/2 >= 992]
print("dash-sized in zone:", len(dash), "| median", statistics.median(b[3] for b in dash), "x", statistics.median(b[4] for b in dash))
tiny = [b for b in B if b[3] < 30 or b[4] < 20]
print("tiny (<30 wide or <20 tall):", len(tiny))
per = collections.Counter(b[0] for b in B)
print("boxes per frame: mean", round(sum(per.values())/len(per), 1), "| >=20:", sorted(f for f, c in per.items() if c >= 20))
def iou(a, b):
    ax, ay, aw, ah = a[1:]; bx, by, bw, bh = b[1:]
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx)); iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    i = ix*iy; u = aw*ah + bw*bh - i
    return i/u if u else 0
byf = collections.defaultdict(list)
for b in B: byf[b[0]].append(b)
dups = [(f, bs[i][1:], bs[j][1:], round(iou(bs[i], bs[j]), 2)) for f, bs in byf.items() for i in range(len(bs)) for j in range(i+1, len(bs)) if iou(bs[i], bs[j]) > 0.5]
print("overlapping boxes (IoU>0.5):", len(dups)); [print("   ", d) for d in dups[:10]]
exact = collections.Counter(b for b in B)
print("exact duplicate rows:", sum(1 for c in exact.values() if c > 1))
nomark = sorted({r['filename'] for r in allrows if r['no_marking'] == 'TRUE'})
print("no_marking frames:", nomark)
# frames with suspiciously few boxes
few = sorted((c, f) for f, c in per.items() if c <= 2)
print("frames with <=2 boxes:", few)
