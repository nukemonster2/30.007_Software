"""
compute.py

Computes derived metrics for a session after readings have been saved.

MyotonPro outputs two parameters per landmark: frequency (Hz, muscle tone)
and stiffness (N/m). Asymmetry Index is computed separately for each:

    AI = |L - R| / ((L + R) / 2) * 100

A landmark pair is flagged (flagged = 1 on both its L and R rows) if
EITHER parameter's AI exceeds FLAG_THRESHOLD. Change the `or` to `and`
below if you want both parameters to need to exceed it instead.

Composite Score (per session) pools every AI value computed across all
landmarks and both parameters, then averages them:
    composite_score = sum(all AI values) / count

Note: readings also stores a raw `acceleration` value per reading, but no
asymmetry index is computed for it (by request) -- it's captured for
reference/future use only and doesn't factor into flagging or the
composite score.
"""

from Database.db import get_connection

FLAG_THRESHOLD = 15  # AI percentage above which a landmark pair is flagged


def compute_asymmetry_index(left_value, right_value):
    """Return the Asymmetry Index (%) between a left and right reading."""
    if left_value is None or right_value is None:
        return None
    denom = (left_value + right_value) / 2
    if denom == 0:
        return 0.0
    return abs(left_value - right_value) / denom * 100


def compute_session(session_id):
    """
    Compute AI (frequency + stiffness) per landmark, flag readings, and
    update the session's composite_score. Call after save_readings().
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        cur.execute("""
            SELECT reading_id, landmark, side, frequency, stiffness
            FROM readings
            WHERE session_id = ?
        """, (session_id,))
        rows = cur.fetchall()

        # Group readings by landmark
        by_landmark = {}
        for row_id, landmark, side, frequency, stiffness in rows:
            by_landmark.setdefault(landmark, {})[side] = (row_id, frequency, stiffness)

        all_ai_values = []

        for landmark, sides in by_landmark.items():
            if "L" not in sides or "R" not in sides:
                # Can't compute AI without both sides present
                continue

            l_id, l_freq, l_stiff = sides["L"]
            r_id, r_freq, r_stiff = sides["R"]

            ai_freq = compute_asymmetry_index(l_freq, r_freq)
            ai_stiff = compute_asymmetry_index(l_stiff, r_stiff)

            for ai in (ai_freq, ai_stiff):
                if ai is not None:
                    all_ai_values.append(ai)

            flagged = 1 if (
                (ai_freq is not None and ai_freq > FLAG_THRESHOLD) or
                (ai_stiff is not None and ai_stiff > FLAG_THRESHOLD)
            ) else 0

            cur.execute("""
                UPDATE readings
                SET asymmetry_idx_frequency = ?,
                    asymmetry_idx_stiffness = ?,
                    flagged = ?
                WHERE reading_id IN (?, ?)
            """, (ai_freq, ai_stiff, flagged, l_id, r_id))

        composite_score = (
            sum(all_ai_values) / len(all_ai_values) if all_ai_values else None
        )

        cur.execute("""
            UPDATE sessions
            SET composite_score = ?
            WHERE session_id = ?
        """, (composite_score, session_id))

        conn.commit()
        return composite_score

    finally:
        conn.close()