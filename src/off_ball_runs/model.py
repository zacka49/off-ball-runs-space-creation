from __future__ import annotations

import math

import pandas as pd

from off_ball_runs.data import (
    Direction,
    TeamName,
    frame_column,
    frame_positions,
    nearest_distance,
    player_pairs,
    time_column,
)


def _lane_blockers(
    ball: tuple[float, float] | None,
    runner: tuple[float, float],
    defenders: pd.DataFrame,
    lane_width_m: float = 3.0,
) -> int:
    if ball is None or defenders.empty:
        return 0

    bx, by = ball
    rx, ry = runner
    vx = rx - bx
    vy = ry - by
    length_sq = vx * vx + vy * vy
    if length_sq == 0:
        return 0

    blockers = 0
    for _, defender in defenders.iterrows():
        wx = defender["x"] - bx
        wy = defender["y"] - by
        projection = (wx * vx + wy * vy) / length_sq
        if projection <= 0 or projection >= 1:
            continue
        closest_x = bx + projection * vx
        closest_y = by + projection * vy
        distance = math.hypot(defender["x"] - closest_x, defender["y"] - closest_y)
        if distance <= lane_width_m:
            blockers += 1
    return blockers


def _score_run(row: dict[str, float | int | str]) -> float:
    speed_component = min(float(row["max_speed_mps"]) / 8.0, 1.0)
    progression_component = min(max(float(row["x_progression_m"]), 0.0) / 25.0, 1.0)
    separation_component = min(max(float(row["nearest_defender_end_m"]), 0.0) / 12.0, 1.0)
    line_break_component = min(max(float(row["line_break_m"]), 0.0) / 12.0, 1.0)
    lane_component = min(max(float(row["lane_improvement"]), 0.0) / 3.0, 1.0)
    final_third_component = 1.0 if float(row["end_x"]) >= 70 else 0.0

    return round(
        100
        * (
            0.24 * speed_component
            + 0.22 * progression_component
            + 0.18 * separation_component
            + 0.18 * line_break_component
            + 0.10 * lane_component
            + 0.08 * final_third_component
        ),
        1,
    )


def _run_type(row: dict[str, float | int | str]) -> str:
    start_y = float(row["start_y"])
    end_y = float(row["end_y"])
    end_x = float(row["end_x"])
    line_break = float(row["line_break_m"])
    lane_improvement = float(row["lane_improvement"])

    starts_wide = start_y <= 14 or start_y >= 54
    ends_central = 22 <= end_y <= 46
    ends_wide = end_y <= 14 or end_y >= 54

    if line_break >= 3 and end_x >= 70:
        return "Run In Behind"
    if starts_wide and ends_central and end_x >= 60:
        return "Underlap"
    if ends_wide and float(row["x_progression_m"]) >= 8:
        return "Wide Stretch"
    if lane_improvement >= 1:
        return "Decoy / Lane Opener"
    return "Support Run"


def _deduplicate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs

    selected = []
    ranked = runs.sort_values("space_created_score", ascending=False)
    for _, candidate in ranked.iterrows():
        overlaps = [
            existing
            for existing in selected
            if existing["player"] == candidate["player"]
            and not (
                candidate["end_frame"] < existing["start_frame"]
                or candidate["start_frame"] > existing["end_frame"]
            )
        ]
        if not overlaps:
            selected.append(candidate.to_dict())
    return pd.DataFrame(selected).sort_values("start_time_s").reset_index(drop=True)


def detect_off_ball_runs(
    home: pd.DataFrame,
    away: pd.DataFrame,
    attacking_team: TeamName = "Home",
    direction: Direction = "left-to-right",
    start_minute: float = 0.0,
    end_minute: float = 15.0,
    frame_step: int = 12,
    window_seconds: float = 2.0,
    min_speed_mps: float = 4.8,
    min_x_progression_m: float = 6.0,
    min_off_ball_distance_m: float = 6.0,
) -> pd.DataFrame:
    """Detect high-value off-ball forward runs from tracking data.

    The method is deliberately explainable. It scores candidate runs using speed,
    forward progression, defender separation, line-breaking movement and passing
    lane improvement from the ball to the runner.
    """

    attack_frame = home if attacking_team == "Home" else away
    fcol = frame_column(attack_frame)
    tcol = time_column(attack_frame)
    start_second = start_minute * 60
    end_second = end_minute * 60

    sample = attack_frame[
        attack_frame[tcol].between(start_second, end_second, inclusive="both")
    ].iloc[::frame_step]
    players = player_pairs(attack_frame)
    if sample.empty or not players:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for _, start_row in sample.iterrows():
        start_time = float(start_row[tcol])
        end_time = start_time + window_seconds
        if end_time > end_second:
            continue

        end_row = attack_frame.iloc[(attack_frame[tcol] - end_time).abs().idxmin()]
        start_frame = int(start_row[fcol])
        end_frame = int(end_row[fcol])
        start_players, start_ball = frame_positions(home, away, start_frame, direction)
        end_players, _ = frame_positions(home, away, end_frame, direction)

        attackers_start = start_players[start_players["team"].eq(attacking_team)]
        defenders_start = start_players[start_players["team"].ne(attacking_team)]
        attackers_end = end_players[end_players["team"].eq(attacking_team)]
        defenders_end = end_players[end_players["team"].ne(attacking_team)]

        if defenders_start.empty or attackers_start.empty:
            continue

        defensive_line = float(defenders_start["x"].quantile(0.85))

        for player in players:
            start_player = attackers_start[attackers_start["player"].eq(player)]
            end_player = attackers_end[attackers_end["player"].eq(player)]
            if start_player.empty or end_player.empty:
                continue

            x0 = float(start_player.iloc[0]["x"])
            y0 = float(start_player.iloc[0]["y"])
            x1 = float(end_player.iloc[0]["x"])
            y1 = float(end_player.iloc[0]["y"])
            distance = math.hypot(x1 - x0, y1 - y0)
            speed = distance / max(window_seconds, 0.1)
            x_progression = x1 - x0

            if speed < min_speed_mps or x_progression < min_x_progression_m:
                continue

            if start_ball is not None:
                distance_to_ball = math.hypot(x0 - start_ball[0], y0 - start_ball[1])
                if distance_to_ball < min_off_ball_distance_m:
                    continue
            else:
                distance_to_ball = float("nan")

            nearest_start = nearest_distance((x0, y0), defenders_start)
            nearest_end = nearest_distance((x1, y1), defenders_end)
            blockers_start = _lane_blockers(start_ball, (x0, y0), defenders_start)
            blockers_end = _lane_blockers(start_ball, (x1, y1), defenders_end)

            row: dict[str, float | int | str] = {
                "player": player,
                "attacking_team": attacking_team,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": round(start_time, 2),
                "end_time_s": round(float(end_row[tcol]), 2),
                "start_x": round(x0, 2),
                "start_y": round(y0, 2),
                "end_x": round(x1, 2),
                "end_y": round(y1, 2),
                "distance_m": round(distance, 2),
                "max_speed_mps": round(speed, 2),
                "x_progression_m": round(x_progression, 2),
                "distance_to_ball_start_m": round(distance_to_ball, 2),
                "nearest_defender_start_m": round(nearest_start, 2),
                "nearest_defender_end_m": round(nearest_end, 2),
                "line_break_m": round(max(0.0, x1 - defensive_line), 2),
                "lane_blockers_start": int(blockers_start),
                "lane_blockers_end": int(blockers_end),
                "lane_improvement": int(blockers_start - blockers_end),
            }
            row["space_created_score"] = _score_run(row)
            row["run_type"] = _run_type(row)
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return _deduplicate_runs(pd.DataFrame(rows))


def coach_notes(runs: pd.DataFrame) -> list[str]:
    if runs.empty:
        return ["No off-ball runs were detected with the current thresholds."]

    top = runs.sort_values("space_created_score", ascending=False).iloc[0]
    notes = [
        f"The top run is {top['player']} at {top['start_time_s']:.1f}s, scored {top['space_created_score']:.1f}/100 as a {top['run_type']}.",
    ]

    type_counts = runs["run_type"].value_counts()
    notes.append(
        f"The most common run type is {type_counts.index[0]} ({int(type_counts.iloc[0])} detected runs)."
    )

    final_third_rate = runs["end_x"].ge(70).mean()
    notes.append(
        f"{final_third_rate:.0%} of detected runs finish in the final third, which is useful for reviewing timing and supporting pass options."
    )

    lane_openers = runs[runs["lane_improvement"].gt(0)]
    if not lane_openers.empty:
        notes.append(
            f"{len(lane_openers)} runs improved the ball-to-runner passing lane, highlighting off-ball value beyond touches received."
        )
    return notes
