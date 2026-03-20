import json
import statistics
import math
from collections import defaultdict

DATA_PATH = r"c:\Users\Georgiy\Documents\coursework\hateful content localisation\data\hateclipseg\dataset\hateclipseg_merged.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

db = data["database"]

# ── collect per-video data ──────────────────────────────────────────────────
split_counts = defaultdict(int)
video_durations = []          # declared duration field
video_max_end = []            # max segment end time
seg_durations = []            # every segment (end - start)
segs_per_video = []           # how many segments each video has
has_hate = []                 # bool: ≥1 hateful segment?
hateful_time_per_video = []   # total annotated hateful seconds
gaps = []                     # gaps between consecutive segs in same video

for vid_id, info in db.items():
    subset = info.get("subset", "unknown")
    split_counts[subset] += 1

    duration = info.get("duration", None)
    annots = info.get("annotations", [])

    # filter hate segments
    hate_segs = [a["segment"] for a in annots if a.get("label") == "hate"]
    hate_segs.sort(key=lambda s: s[0])   # sort by start time

    # video duration: prefer declared field, fall back to max end
    if duration is not None:
        video_durations.append(duration)
    if hate_segs:
        video_max_end.append(hate_segs[-1][1])

    # segment durations
    seg_dur_this = []
    for seg in hate_segs:
        d = seg[1] - seg[0]
        seg_durations.append(d)
        seg_dur_this.append(d)

    segs_per_video.append(len(hate_segs))
    has_hate.append(len(hate_segs) > 0)
    hateful_time_per_video.append(sum(seg_dur_this))

    # gaps between consecutive hate segments
    for i in range(1, len(hate_segs)):
        gap = hate_segs[i][0] - hate_segs[i-1][1]
        gaps.append(gap)

# ── helper functions ────────────────────────────────────────────────────────
def percentile(data_sorted, p):
    """Linear-interpolation percentile on pre-sorted list."""
    n = len(data_sorted)
    if n == 0:
        return float("nan")
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return data_sorted[-1]
    frac = idx - lo
    return data_sorted[lo] + frac * (data_sorted[hi] - data_sorted[lo])

def stats_block(label, values):
    if not values:
        print(f"{label}: NO DATA")
        return
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    std = math.sqrt(variance)
    print(f"\n{'-'*60}")
    print(f"  {label}  (n={n})")
    print(f"{'-'*60}")
    print(f"  mean   : {mean:.2f}")
    print(f"  median : {percentile(s, 50):.2f}")
    print(f"  std    : {std:.2f}")
    print(f"  min    : {s[0]:.2f}")
    print(f"  max    : {s[-1]:.2f}")
    print(f"  p10    : {percentile(s, 10):.2f}")
    print(f"  p25    : {percentile(s, 25):.2f}")
    print(f"  p75    : {percentile(s, 75):.2f}")
    print(f"  p90    : {percentile(s, 90):.2f}")
    print(f"  p95    : {percentile(s, 95):.2f}")
    print(f"  p99    : {percentile(s, 99):.2f}")

def hist(label, values, bins):
    """bins: list of (lo, hi) or (lo, None) for open-ended last bin."""
    print(f"\n  {label} histogram:")
    for i, (lo, hi) in enumerate(bins):
        if hi is None:
            count = sum(1 for v in values if v >= lo)
            print(f"    [{lo:>4}+     ) : {count:>5}  ({100*count/len(values):.1f}%)")
        else:
            count = sum(1 for v in values if lo <= v < hi)
            print(f"    [{lo:>4}, {hi:>4}) : {count:>5}  ({100*count/len(values):.1f}%)")

# ── 1. split counts ──────────────────────────────────────────────────────────
total = sum(split_counts.values())
print("=" * 60)
print("  DATASET SPLIT COUNTS")
print("=" * 60)
print(f"  Total videos : {total}")
for subset in ["train", "val", "test"]:
    c = split_counts.get(subset, 0)
    print(f"  {subset:<6}       : {c}  ({100*c/total:.1f}%)")

# ── 2. segment duration stats ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SEGMENT DURATION STATISTICS  (end - start, seconds)")
print("=" * 60)
stats_block("Segment durations (all segments)", seg_durations)

dur_bins = [(0,5),(5,10),(10,20),(20,40),(40,60),(60,None)]
if seg_durations:
    hist("Segment duration", seg_durations, dur_bins)

# ── 3. segments per video ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SEGMENTS PER VIDEO")
print("=" * 60)
s_spv = sorted(segs_per_video)
n = len(s_spv)
mean_spv = sum(s_spv) / n
print(f"  mean   : {mean_spv:.2f}")
print(f"  median : {percentile(s_spv, 50):.2f}")
print(f"  max    : {s_spv[-1]}")
print(f"  p75    : {percentile(s_spv, 75):.2f}")
print(f"  p90    : {percentile(s_spv, 90):.2f}")
print(f"  p95    : {percentile(s_spv, 95):.2f}")
# distribution
for k in range(0, max(s_spv)+1):
    c = s_spv.count(k)
    if c:
        print(f"    segs={k}: {c} videos ({100*c/n:.1f}%)")

# ── 4. video duration stats ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  VIDEO DURATION STATISTICS  (declared 'duration' field, seconds)")
print("=" * 60)
stats_block("Video durations", video_durations)

# ── 5. D/2 distribution ──────────────────────────────────────────────────────
half_dur = [d / 2 for d in seg_durations]
half_sorted = sorted(half_dur)
print("\n" + "=" * 60)
print("  D/2 DISTRIBUTION  (half-duration = (end-start)/2)")
print("=" * 60)
if half_sorted:
    print(f"  p25 : {percentile(half_sorted, 25):.2f}")
    print(f"  p50 : {percentile(half_sorted, 50):.2f}")
    print(f"  p75 : {percentile(half_sorted, 75):.2f}")
    print(f"  p90 : {percentile(half_sorted, 90):.2f}")

# ── 6. segment density ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SEGMENT DENSITY  (hateful time / video duration)")
print("=" * 60)
densities = []
for vid_id, info in db.items():
    dur = info.get("duration")
    annots = info.get("annotations", [])
    hate_segs = [a["segment"] for a in annots if a.get("label") == "hate"]
    if dur and dur > 0:
        total_hate = sum(s[1] - s[0] for s in hate_segs)
        densities.append(total_hate / dur)

d_sorted = sorted(densities)
n_d = len(d_sorted)
if n_d:
    mean_d = sum(d_sorted) / n_d
    var_d = sum((x - mean_d)**2 for x in d_sorted) / n_d
    print(f"  mean   : {mean_d:.4f}  ({mean_d*100:.2f}%)")
    print(f"  median : {percentile(d_sorted, 50):.4f}  ({percentile(d_sorted,50)*100:.2f}%)")
    print(f"  std    : {math.sqrt(var_d):.4f}")
    print(f"  min    : {d_sorted[0]:.4f}")
    print(f"  max    : {d_sorted[-1]:.4f}")
    print(f"  p25    : {percentile(d_sorted, 25):.4f}  ({percentile(d_sorted,25)*100:.2f}%)")
    print(f"  p75    : {percentile(d_sorted, 75):.4f}  ({percentile(d_sorted,75)*100:.2f}%)")
    print(f"  p90    : {percentile(d_sorted, 90):.4f}  ({percentile(d_sorted,90)*100:.2f}%)")

# ── 7. class balance ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CLASS BALANCE")
print("=" * 60)
with_hate = sum(has_hate)
without_hate = len(has_hate) - with_hate
print(f"  Videos WITH >=1 hateful segment : {with_hate}  ({100*with_hate/len(has_hate):.1f}%)")
print(f"  Videos with NO hateful segments : {without_hate}  ({100*without_hate/len(has_hate):.1f}%)")

# also per split
for subset in ["train", "val", "test"]:
    with_h = sum(
        1 for vid_id, info in db.items()
        if info.get("subset") == subset and
           any(a.get("label") == "hate" for a in info.get("annotations", []))
    )
    total_s = split_counts.get(subset, 0)
    print(f"    {subset}: {with_h}/{total_s} have hate ({100*with_h/total_s if total_s else 0:.1f}%)")

# ── 8. gap between segments ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  GAP BETWEEN CONSECUTIVE HATEFUL SEGMENTS  (seconds)")
print("=" * 60)
if gaps:
    g_sorted = sorted(gaps)
    n_g = len(g_sorted)
    mean_g = sum(g_sorted) / n_g
    var_g = sum((x - mean_g)**2 for x in g_sorted) / n_g
    print(f"  count  : {n_g}")
    print(f"  mean   : {mean_g:.2f}")
    print(f"  median : {percentile(g_sorted, 50):.2f}")
    print(f"  std    : {math.sqrt(var_g):.2f}")
    print(f"  min    : {g_sorted[0]:.2f}")
    print(f"  max    : {g_sorted[-1]:.2f}")
    print(f"  p25    : {percentile(g_sorted, 25):.2f}")
    print(f"  p75    : {percentile(g_sorted, 75):.2f}")
    print(f"  p90    : {percentile(g_sorted, 90):.2f}")
    hist("Gap", g_sorted, [(0,5),(5,10),(10,20),(20,40),(40,60),(60,None)])
else:
    print("  No multi-segment videos found.")

print("\n" + "=" * 60)
print("  DONE")
print("=" * 60)
