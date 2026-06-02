from __future__ import annotations

import numpy as np
import pandas as pd


FIT_COLUMNS = [
    "player",
    "tuchel_fit_score",
    "best_role",
    "runs",
    "vertical_threat",
    "lane_creation",
    "wide_threat",
    "half_space_threat",
    "final_third_threat",
    "repeatable_intensity",
    "avg_score",
    "top_score",
    "avg_speed",
    "run_in_behind",
    "lane_openers",
    "wide_runs",
    "half_space_runs",
    "final_third_runs",
]

ROLE_COLUMNS = [
    "Wing-back / wide runner",
    "Inside forward / half-space runner",
    "Central forward / depth runner",
    "Connector / decoy runner",
]


def _safe_score(series: pd.Series, high: float) -> pd.Series:
    if series.empty:
        return series
    return (series.fillna(0).clip(lower=0) / high * 100).clip(upper=100)


def _is_wide(y: pd.Series) -> pd.Series:
    return (y <= 14) | (y >= 54)


def _is_half_space(y: pd.Series) -> pd.Series:
    return ((y > 14) & (y <= 28)) | ((y >= 40) & (y < 54))


def _summarise_player(group: pd.DataFrame) -> dict[str, float | int | str]:
    runs = len(group)
    run_in_behind = group["run_type"].eq("Run In Behind").sum()
    lane_openers = group["lane_improvement"].gt(0).sum()
    wide_runs = (_is_wide(group["end_y"]) | group["run_type"].eq("Wide Stretch")).sum()
    half_space_runs = _is_half_space(group["end_y"]).sum()
    final_third_runs = group["end_x"].ge(70).sum()

    avg_score = float(group["space_created_score"].mean())
    top_score = float(group["space_created_score"].max())
    avg_speed = float(group["max_speed_mps"].mean())
    avg_progression = float(group["x_progression_m"].mean())
    avg_line_break = float(group["line_break_m"].mean())
    avg_lane_gain = float(group["lane_improvement"].clip(lower=0).mean())
    avg_defender_sep = float(group["nearest_defender_end_m"].mean())

    vertical_threat = np.mean(
        [
            min(avg_progression / 14 * 100, 100),
            min(avg_line_break / 6 * 100, 100),
            min(run_in_behind / max(runs, 1) * 160, 100),
            min(final_third_runs / max(runs, 1) * 140, 100),
        ]
    )
    lane_creation = np.mean(
        [
            min(lane_openers / max(runs, 1) * 160, 100),
            min(avg_lane_gain / 1.2 * 100, 100),
            min(avg_defender_sep / 8 * 100, 100),
        ]
    )
    wide_threat = np.mean(
        [
            min(wide_runs / max(runs, 1) * 170, 100),
            min(group.loc[_is_wide(group["end_y"]), "x_progression_m"].mean() / 12 * 100, 100)
            if wide_runs
            else 0,
        ]
    )
    half_space_threat = np.mean(
        [
            min(half_space_runs / max(runs, 1) * 170, 100),
            min(group.loc[_is_half_space(group["end_y"]), "space_created_score"].mean() / 70 * 100, 100)
            if half_space_runs
            else 0,
        ]
    )
    final_third_threat = np.mean(
        [
            min(final_third_runs / max(runs, 1) * 160, 100),
            min(top_score / 80 * 100, 100),
            min(avg_score / 55 * 100, 100),
        ]
    )
    repeatable_intensity = np.mean(
        [
            min(runs / 12 * 100, 100),
            min(avg_speed / 6.2 * 100, 100),
            min(avg_score / 50 * 100, 100),
        ]
    )

    role_scores = {
        "Wing-back / wide runner": 0.36 * wide_threat
        + 0.26 * final_third_threat
        + 0.22 * repeatable_intensity
        + 0.16 * lane_creation,
        "Inside forward / half-space runner": 0.35 * half_space_threat
        + 0.28 * vertical_threat
        + 0.22 * final_third_threat
        + 0.15 * lane_creation,
        "Central forward / depth runner": 0.44 * vertical_threat
        + 0.25 * final_third_threat
        + 0.21 * repeatable_intensity
        + 0.10 * lane_creation,
        "Connector / decoy runner": 0.44 * lane_creation
        + 0.22 * repeatable_intensity
        + 0.18 * half_space_threat
        + 0.16 * vertical_threat,
    }
    tuchel_fit_score = (
        0.28 * vertical_threat
        + 0.22 * lane_creation
        + 0.18 * repeatable_intensity
        + 0.14 * final_third_threat
        + 0.10 * half_space_threat
        + 0.08 * wide_threat
    )

    return {
        "player": str(group["player"].iloc[0]),
        "tuchel_fit_score": round(float(tuchel_fit_score), 1),
        "best_role": max(role_scores, key=role_scores.get),
        "runs": int(runs),
        "vertical_threat": round(float(vertical_threat), 1),
        "lane_creation": round(float(lane_creation), 1),
        "wide_threat": round(float(wide_threat), 1),
        "half_space_threat": round(float(half_space_threat), 1),
        "final_third_threat": round(float(final_third_threat), 1),
        "repeatable_intensity": round(float(repeatable_intensity), 1),
        "avg_score": round(avg_score, 1),
        "top_score": round(top_score, 1),
        "avg_speed": round(avg_speed, 2),
        "run_in_behind": int(run_in_behind),
        "lane_openers": int(lane_openers),
        "wide_runs": int(wide_runs),
        "half_space_runs": int(half_space_runs),
        "final_third_runs": int(final_third_runs),
        **{role: round(float(score), 1) for role, score in role_scores.items()},
    }


def tuchel_fit_table(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame(columns=FIT_COLUMNS + ROLE_COLUMNS)

    rows = [_summarise_player(group) for _, group in runs.groupby("player")]
    return (
        pd.DataFrame(rows)
        .sort_values(["tuchel_fit_score", "top_score"], ascending=False)
        .reset_index(drop=True)
    )


def tuchel_role_matrix(fit_table: pd.DataFrame) -> pd.DataFrame:
    if fit_table.empty:
        return pd.DataFrame(columns=["player", *ROLE_COLUMNS])
    return fit_table[["player", *ROLE_COLUMNS]].copy()


def tuchel_notes(fit_table: pd.DataFrame) -> list[str]:
    if fit_table.empty:
        return ["No player fit table is available for the current run sample."]

    top = fit_table.iloc[0]
    notes = [
        f"{top['player']} is the strongest Tuchel-fit profile in this sample, scoring {top['tuchel_fit_score']:.1f}/100.",
        f"Their best role fit is {top['best_role']}, driven by vertical threat {top['vertical_threat']:.1f}, lane creation {top['lane_creation']:.1f} and repeatable intensity {top['repeatable_intensity']:.1f}.",
    ]

    depth_runner = fit_table.sort_values("Central forward / depth runner", ascending=False).iloc[0]
    wide_runner = fit_table.sort_values("Wing-back / wide runner", ascending=False).iloc[0]
    connector = fit_table.sort_values("Connector / decoy runner", ascending=False).iloc[0]
    notes.append(
        f"For role balance: {depth_runner['player']} grades best as a depth runner, {wide_runner['player']} as a wide runner and {connector['player']} as a connector or decoy runner."
    )
    return notes
