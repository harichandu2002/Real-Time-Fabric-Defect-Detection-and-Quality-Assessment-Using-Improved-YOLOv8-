from collections import Counter
from dataclasses import dataclass, field
from typing import List
import datetime
import csv

SEVERITY = {
    'Hole':          10,
    'Broken_Thread':  7,
    'Misweave':       5,
    'Stain':          3,
}

GRADE_THRESHOLDS = [
    (90, 'A'),
    (75, 'B'),
    (55, 'C'),
    (35, 'D'),
    (0,  'F'),
]


@dataclass
class DefectEvent:
    timestamp:  float
    class_name: str
    confidence: float


@dataclass
class SessionResult:
    total_frames:    int
    defect_frames:   int
    clean_frames:    int
    defect_counts:   Counter
    defect_rate_pct: float
    quality_score:   float
    quality_grade:   str
    suggested_price: float
    max_market_rate: float
    duration_secs:   float
    events:          List[DefectEvent]


def analyse_session(events: List[DefectEvent],
                    total_frames: int,
                    duration_secs: float,
                    max_market_rate: float) -> SessionResult:

    defect_frames = len(events)
    clean_frames  = max(0, total_frames - defect_frames)
    defect_rate   = (defect_frames / max(total_frames, 1)) * 100.0
    defect_counts = Counter(e.class_name for e in events)

    # ── Quality score ──────────────────────────────────────────────
    # Weighted defect rate: weight each defect by its severity,
    # then normalise so 0 defects = 100, all frames defective = 0.
    # Max possible severity per frame = 10 (Hole).
    # weighted_defect_rate is in [0, 1].
    total_severity = sum(
        SEVERITY.get(cls, 5) * count
        for cls, count in defect_counts.items()
    )
    max_possible   = 10 * max(total_frames, 1)   # if every frame were a Hole
    weighted_rate  = total_severity / max_possible   # 0.0 … 1.0

    quality_score  = round(max(0.0, min(100.0, (1.0 - weighted_rate) * 100.0)), 2)

    # ── Grade ──────────────────────────────────────────────────────
    grade = 'F'
    for threshold, g in GRADE_THRESHOLDS:
        if quality_score >= threshold:
            grade = g
            break

    # ── Price: quadratic penalty ───────────────────────────────────
    # quality 100 → full price; quality 0 → ₹0
    price_ratio     = (quality_score / 100.0) ** 2
    suggested_price = round(max_market_rate * price_ratio, 2)

    return SessionResult(
        total_frames    = total_frames,
        defect_frames   = defect_frames,
        clean_frames    = clean_frames,
        defect_counts   = defect_counts,
        defect_rate_pct = round(defect_rate, 2),
        quality_score   = quality_score,
        quality_grade   = grade,
        suggested_price = suggested_price,
        max_market_rate = max_market_rate,
        duration_secs   = round(duration_secs, 1),
        events          = events,
    )


def export_csv(result: SessionResult, filepath: str):
    with open(filepath, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['FABRIC INSPECTION REPORT'])
        w.writerow(['Generated',
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        w.writerow([])
        w.writerow(['SUMMARY'])
        w.writerow(['Duration (s)',          result.duration_secs])
        w.writerow(['Total frames',          result.total_frames])
        w.writerow(['Clean frames',          result.clean_frames])
        w.writerow(['Defect frames',         result.defect_frames])
        w.writerow(['Defect rate (%)',        result.defect_rate_pct])
        w.writerow(['Quality score (0-100)', result.quality_score])
        w.writerow(['Quality grade',         result.quality_grade])
        w.writerow(['Max market rate',       result.max_market_rate])
        w.writerow(['Suggested price',       result.suggested_price])
        w.writerow([])
        w.writerow(['DEFECT BREAKDOWN'])
        w.writerow(['Class', 'Count', 'Severity', 'Contribution'])
        for cls, count in result.defect_counts.most_common():
            sev = SEVERITY.get(cls, 5)
            w.writerow([cls, count, sev, count * sev])
        w.writerow([])
        w.writerow(['EVENT LOG'])
        w.writerow(['Time (s)', 'Class', 'Confidence'])
        for e in result.events:
            w.writerow([round(e.timestamp, 2),
                        e.class_name,
                        round(e.confidence, 3)])