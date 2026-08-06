import os
import sqlite3
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from Database.db import get_connection

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_output_dir(output_dir):
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _resolve_image_path(path):
    if not path:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


# ── Colours ───────────────────────────────────────────────────────────────────
NAVY    = HexColor("#0D1B2A")
TEAL    = HexColor("#028090")
LIGHT   = HexColor("#EAF4F6")
WHITE   = HexColor("#FFFFFF")
RED     = HexColor("#C0392B")
GREEN   = HexColor("#27AE60")
WARN    = HexColor("#E05C00")
MUTED   = HexColor("#64748B")
FLAG_THRESHOLD = 15.0

# ── Styles ────────────────────────────────────────────────────────────────────
_styles = getSampleStyleSheet()

def _style(name, **kwargs):
    return ParagraphStyle(name, parent=_styles["Normal"], **kwargs)

H1    = _style("H1",    fontSize=18, textColor=NAVY,  bold=True, spaceAfter=4,  fontName="Helvetica-Bold",  alignment=TA_CENTER)
H2    = _style("H2",    fontSize=13, textColor=TEAL,  bold=True, spaceAfter=4,  fontName="Helvetica-Bold")
BODY  = _style("BODY",  fontSize=10, textColor=NAVY,  spaceAfter=3,  fontName="Helvetica")
SMALL = _style("SMALL", fontSize=9,  textColor=MUTED, spaceAfter=2,  fontName="Helvetica", alignment=TA_CENTER)
FLAG  = _style("FLAG",  fontSize=10, textColor=RED,   bold=True, spaceAfter=2,  fontName="Helvetica-Bold")

# ── Database helpers ──────────────────────────────────────────────────────────

def _fetch_session_info(conn, session_id: str) -> dict:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT s.session_id, s.patient_id, s.session_date, s.administered_by, "
        "s.composite_score, p.created_at "
        "FROM sessions s JOIN patients p ON s.patient_id = p.patient_id "
        "WHERE s.session_id = ?",
        (session_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Session '{session_id}' not found in database")
    return dict(row)


def _fetch_readings(conn, session_id: str) -> list[dict]:
    """Return one row per landmark (latest L+R pair) with AI (frequency and
    stiffness) and flagged."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT landmark, side, frequency, stiffness, "
        "asymmetry_idx_frequency, asymmetry_idx_stiffness, flagged "
        "FROM readings WHERE session_id = ? ORDER BY taken_at ASC",
        (session_id,)
    ).fetchall()

    # keep latest per (landmark, side)
    latest = {}
    for r in rows:
        latest[(r["landmark"], r["side"])] = dict(r)

    # collapse into one row per landmark
    landmarks = {}
    for (lm, side), data in latest.items():
        landmarks.setdefault(lm, {}).update({side: data})

    result = []
    for lm, sides in sorted(landmarks.items()):
        ai_freq  = None
        ai_stiff = None
        flg = 0
        if "L" in sides:
            ai_freq  = sides["L"]["asymmetry_idx_frequency"]
            ai_stiff = sides["L"]["asymmetry_idx_stiffness"]
            flg = sides["L"]["flagged"]
        elif "R" in sides:
            ai_freq  = sides["R"]["asymmetry_idx_frequency"]
            ai_stiff = sides["R"]["asymmetry_idx_stiffness"]
            flg = sides["R"]["flagged"]
        result.append({
            "landmark": lm,
            "ai_frequency": ai_freq,
            "ai_stiffness": ai_stiff,
            "flagged": flg,
            "sides": sides,
        })
    return result


def _fetch_previous_session(conn, patient_id: str, current_session_id: str):
    row = conn.execute(
        "SELECT session_id FROM sessions "
        "WHERE patient_id = ? AND session_id != ? "
        "ORDER BY session_date DESC LIMIT 1",
        (patient_id, current_session_id)
    ).fetchone()
    return row[0] if row else None

# ── PDF builder ───────────────────────────────────────────────────────────────

def generate_report(
    session_id: str,
    heatmap_path: str | None = None,
    delta_path: str | None   = None,
    output_dir: str          = "reports",
) -> str:
    
    output_dir_path = _resolve_output_dir(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    pdf_path = str(output_dir_path / f"{session_id}_report.pdf")

    conn = get_connection()
    try:
        info     = _fetch_session_info(conn, session_id)
        readings = _fetch_readings(conn, session_id)
    finally:
        conn.close()

    # ── Derived values ────────────────────────────────────────────────────────
    patient_id      = info["patient_id"]
    session_date    = info["session_date"] or "—"
    administered_by = info["administered_by"] or "—"
    composite       = info["composite_score"]
    composite_str   = f"{composite:.1f}%" if composite is not None else "—"

    flagged = [
        r for r in readings if r["flagged"]
    ]
    # One entry per flagged landmark, not per side -- flagging is a
    # pair-level comparison (L vs R), so both sides always share the same
    # flagged status. Listing "Left X" and "Right X" as if independently
    # flagged doubled the row count for no real information gain.
    flagged_labels = [r["landmark"].replace("_", " ").title() for r in flagged]

    heatmap_path = _resolve_image_path(heatmap_path)
    delta_path = _resolve_image_path(delta_path)

    # ── Build flowables ───────────────────────────────────────────────────────
    story = []
    W = A4[0] - 4 * cm   # usable width

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("Clinical Report", H1))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=6))

    # ── Section 1: Patient / Session / Date ───────────────────────────────────
    header_data = [[
        Paragraph(f"<b>Patient</b><br/>{patient_id}",   BODY),
        Paragraph(f"<b>Session</b><br/>{session_id}",   BODY),
        Paragraph(f"<b>Date</b><br/>{session_date}",    BODY),
        Paragraph(f"<b>Administered by</b><br/>{administered_by}", BODY),
    ]]
    header_table = Table(header_data, colWidths=[W/4]*4)
    header_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
        ("BOX",           (0, 0), (-1, -1), 0.5, TEAL),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.3 * cm))

    # ── Section 2: Composite Score ────────────────────────────────────────────
    story.append(Paragraph("Composite Asymmetry Score", H2))

    comp_color = GREEN if (composite is not None and composite < FLAG_THRESHOLD) else RED
    comp_data = [[Paragraph(
        f'<font color="{comp_color.hexval()}" size="22"><b>{composite_str}</b></font>',
        _style("CS", alignment=TA_CENTER, fontName="Helvetica-Bold", fontSize=22, textColor=comp_color)
    )]]
    comp_table = Table(comp_data, colWidths=[W])
    comp_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT),
        ("BOX",           (0, 0), (-1, -1), 0.5, TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(comp_table)
    story.append(Paragraph(
        f"Mean bilateral asymmetry index (frequency and stiffness pooled) across "
        f"all measured landmarks. Threshold for flagging: {FLAG_THRESHOLD:.0f}%.",
        SMALL
    ))
    story.append(Spacer(1, 0.3 * cm))

    # ── Section 3: Heat maps ──────────────────────────────────────────────────
    story.append(Paragraph("Tension Heat Maps", H2))

    MAX_HEATMAP_H = 9 * cm  # generous cap so a tall portrait photo still fits the page

    def _img_or_placeholder(path, label, w, max_h=MAX_HEATMAP_H):
        if path and Path(path).exists():
            # Scale height to the image's real aspect ratio instead of a
            # fixed value -- these are portrait back photos, so a fixed
            # short height squishes/crops them. Capped at max_h so a very
            # tall image still fits on the page.
            try:
                from reportlab.lib.utils import ImageReader
                iw, ih = ImageReader(path).getSize()
                h = min(max_h, w * ih / iw)
            except Exception:
                h = 6 * cm  # fallback if the image can't be read for some reason
            return RLImage(path, width=w, height=h)
        # grey placeholder box with label -- keep it a fixed, modest height
        # since there's no real image driving an aspect ratio here
        h = 6 * cm
        placeholder_data = [[Paragraph(
            f"<i>{label}<br/>(not generated)</i>",
            _style("PH", alignment=TA_CENTER, textColor=MUTED, fontSize=9)
        )]]
        t = Table(placeholder_data, colWidths=[w])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#F0F0F0")),
            ("BOX",           (0, 0), (-1, -1), 0.5, MUTED),
            ("TOPPADDING",    (0, 0), (-1, -1), int(h * 0.4)),
            ("BOTTOMPADDING", (0, 0), (-1, -1), int(h * 0.4)),
        ]))
        return t

    hmap_w = W / 2 - 0.3 * cm
    hm_row = [[
        _img_or_placeholder(heatmap_path, "Heat Map", hmap_w),
        _img_or_placeholder(delta_path,   "Delta Heat Map", hmap_w),
    ]]
    hm_labels = [[
        Paragraph("Current session", SMALL),
        Paragraph("Change vs previous session", SMALL),
    ]]
    hm_table = Table(hm_row + hm_labels, colWidths=[hmap_w + 0.3*cm, hmap_w + 0.3*cm])
    hm_table.setStyle(TableStyle([
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(hm_table)
    story.append(Spacer(1, 0.3 * cm))

    # ── Section 4: Landmark table ─────────────────────────────────────────────
    story.append(Paragraph("Asymmetry Index by Landmark (Top 4)", H2))

    def _worst_ai_for_sort(r):
        candidates = [v for v in (r["ai_frequency"], r["ai_stiffness"]) if v is not None]
        return max(candidates) if candidates else -1  # unmeasured landmarks sort last

    top_readings = sorted(readings, key=_worst_ai_for_sort, reverse=True)[:4]

    tbl_header = [
        Paragraph("<b>Landmark</b>",             BODY),
        Paragraph("<b>Frequency AI</b>",         BODY),
        Paragraph("<b>Stiffness AI</b>",         BODY),
        Paragraph("<b>Flag</b>",                 BODY),
    ]
    tbl_data = [tbl_header]
    for r in top_readings:
        freq_str  = f"{r['ai_frequency']:.1f}%" if r["ai_frequency"] is not None else "—"
        stiff_str = f"{r['ai_stiffness']:.1f}%" if r["ai_stiffness"] is not None else "—"
        flag_str = "⚠ FLAGGED" if r["flagged"] else ""

        freq_color  = RED if (r["ai_frequency"]  is not None and r["ai_frequency"]  > FLAG_THRESHOLD) else NAVY
        stiff_color = RED if (r["ai_stiffness"] is not None and r["ai_stiffness"] > FLAG_THRESHOLD) else NAVY

        tbl_data.append([
            Paragraph(r["landmark"].replace("_", " ").title(), BODY),
            Paragraph(f'<font color="{freq_color.hexval()}"><b>{freq_str}</b></font>', BODY),
            Paragraph(f'<font color="{stiff_color.hexval()}"><b>{stiff_str}</b></font>', BODY),
            Paragraph(f'<font color="{RED.hexval()}">{flag_str}</font>', BODY),
        ])

    col_w = [W * 0.40, W * 0.22, W * 0.22, W * 0.16]
    lm_table = Table(tbl_data, colWidths=col_w, repeatRows=1)
    row_styles = [
        ("BACKGROUND",    (0, 0), (-1,  0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1,  0), WHITE),
        ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("BOX",           (0, 0), (-1, -1), 0.5, TEAL),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, HexColor("#CBD5E1")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]
    # alternate row shading
    for i in range(1, len(tbl_data)):
        if i % 2 == 0:
            row_styles.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    lm_table.setStyle(TableStyle(row_styles))
    story.append(lm_table)
    story.append(Spacer(1, 0.3 * cm))

    # ── Section 5: Flagged muscles ────────────────────────────────────────────
    story.append(Paragraph("Flagged Muscles", H2))

    if flagged_labels:
        flag_data = [[Paragraph(f"⚠  {label}", FLAG)] for label in flagged_labels]
        flag_table = Table(flag_data, colWidths=[W])
        flag_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#FFF0F0")),
            ("BOX",           (0, 0), (-1, -1), 0.5, RED),
            ("INNERGRID",     (0, 0), (-1, -1), 0.3, HexColor("#FADBD8")),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ]))
        story.append(flag_table)
        story.append(Paragraph(
            f"Muscles where frequency or stiffness asymmetry index exceeds "
            f"{FLAG_THRESHOLD:.0f}%. Review treatment plan for these regions.",
            SMALL
        ))
    else:
        story.append(Paragraph(
            "No muscles flagged — all bilateral asymmetry indices are within the normal range.",
            _style("OK", fontSize=10, textColor=GREEN, fontName="Helvetica-Bold")
        ))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
        f"Back Documentation Sensing Monitor  ·  For clinical reference only — not a standalone diagnosis.",
        SMALL
    ))

    # ── Write PDF ─────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
        title=f"Back Documentation Sensing Monitor Report — {session_id}",
        author="Back Documentation Sensing Monitor",
    )
    doc.build(story)
    return pdf_path


# ── Convenience: also save the path back to the database ─────────────────────

def generate_and_save_report(
    session_id: str,
    heatmap_path: str | None = None,
    delta_path: str | None   = None,
    output_dir: str          = "reports",
) -> str:
    """
    Like generate_report() but also writes the PDF path back to the
    reports table in the database so the pipeline can retrieve it later.
    """
    pdf_path = generate_report(session_id, heatmap_path, delta_path, output_dir)

    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO reports (session_id, heatmap_path, delta_path, pdf_path)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                heatmap_path = excluded.heatmap_path,
                delta_path   = excluded.delta_path,
                pdf_path     = excluded.pdf_path,
                generated_at = datetime('now')
        """, (session_id, heatmap_path, delta_path, pdf_path))
        conn.commit()
    except Exception:
        # reports table may not exist yet — add it to schema if needed
        pass
    finally:
        conn.close()

    return pdf_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python report.py <session_id> [heatmap.png] [delta.png]")
        sys.exit(1)
    sid  = sys.argv[1]
    hm   = sys.argv[2] if len(sys.argv) > 2 else None
    dlta = sys.argv[3] if len(sys.argv) > 3 else None
    out  = generate_report(sid, hm, dlta)
    print(f"Report written to: {out}")