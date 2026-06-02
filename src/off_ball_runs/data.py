from __future__ import annotations

from pathlib import Path
import re
from typing import Literal

import numpy as np
import pandas as pd
import requests


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
RAW_BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
TeamName = Literal["Home", "Away"]
Direction = Literal["left-to-right", "right-to-left"]


def _download(url: str, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)


def download_game(game_id: int = 1, cache_dir: str | Path = "data/raw/metrica") -> dict[str, Path]:
    if game_id not in {1, 2}:
        raise ValueError("This project supports Metrica sample games 1 and 2 in the standard CSV format.")

    cache = Path(cache_dir) / f"Sample_Game_{game_id}"
    names = {
        "events": f"Sample_Game_{game_id}_RawEventsData.csv",
        "home": f"Sample_Game_{game_id}_RawTrackingData_Home_Team.csv",
        "away": f"Sample_Game_{game_id}_RawTrackingData_Away_Team.csv",
    }
    paths = {key: cache / name for key, name in names.items()}
    for key, name in names.items():
        url = f"{RAW_BASE_URL}/Sample_Game_{game_id}/{name}"
        _download(url, paths[key])
    return paths


def load_events(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _normalise_tracking_columns(columns: list[str]) -> list[str]:
    """Rename Metrica's blank y-coordinate headers into explicit .1 pairs."""

    normalised = list(columns)
    for index, column in enumerate(columns[:-1]):
        is_entity_x = re.fullmatch(r"Player\d+", column) or column in {"Ball", "ball"}
        if not is_entity_x:
            continue

        next_column = columns[index + 1]
        if next_column.startswith("Unnamed:") or next_column == "":
            normalised[index + 1] = f"{column}.1"
    return normalised


def load_tracking(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=2)
    frame.columns = _normalise_tracking_columns([str(column).strip() for column in frame.columns])
    frame = frame.dropna(how="all")
    return frame


def load_game(game_id: int = 1, cache_dir: str | Path = "data/raw/metrica") -> dict[str, pd.DataFrame]:
    paths = download_game(game_id, cache_dir)
    return {
        "events": load_events(paths["events"]),
        "home": load_tracking(paths["home"]),
        "away": load_tracking(paths["away"]),
    }


def frame_column(frame: pd.DataFrame) -> str:
    for candidate in ["Frame", "frame"]:
        if candidate in frame.columns:
            return candidate
    raise KeyError("Could not find a Frame column in the tracking data.")


def time_column(frame: pd.DataFrame) -> str:
    for candidate in ["Time [s]", "Time", "time"]:
        if candidate in frame.columns:
            return candidate
    raise KeyError("Could not find a time column in the tracking data.")


def player_pairs(frame: pd.DataFrame) -> dict[str, tuple[str, str]]:
    pairs: dict[str, tuple[str, str]] = {}
    for column in frame.columns:
        base = re.sub(r"\.\d+$", "", str(column))
        if not re.fullmatch(r"Player\d+", base):
            continue
        if str(column) != base:
            continue
        y_column = f"{base}.1"
        if y_column in frame.columns:
            pairs[base] = (base, y_column)

    for column in frame.columns:
        if str(column).endswith("_x"):
            base = str(column)[:-2]
            y_column = f"{base}_y"
            if y_column in frame.columns:
                pairs[base] = (str(column), y_column)
    return pairs


def _normalise_x(x: float, direction: Direction) -> float:
    metres = x * PITCH_LENGTH_M
    return metres if direction == "left-to-right" else PITCH_LENGTH_M - metres


def _normalise_y(y: float) -> float:
    return y * PITCH_WIDTH_M


def ball_position(row: pd.Series, direction: Direction) -> tuple[float, float] | None:
    x_column = None
    y_column = None
    for candidate in ["Ball", "ball"]:
        if candidate in row.index and f"{candidate}.1" in row.index:
            x_column = candidate
            y_column = f"{candidate}.1"
            break
    if x_column is None:
        return None
    x = row.get(x_column)
    y = row.get(y_column)
    if pd.isna(x) or pd.isna(y):
        return None
    return _normalise_x(float(x), direction), _normalise_y(float(y))


def team_positions(frame_row: pd.Series, tracking: pd.DataFrame, team: TeamName, direction: Direction) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for player, (x_column, y_column) in player_pairs(tracking).items():
        x = frame_row.get(x_column)
        y = frame_row.get(y_column)
        if pd.isna(x) or pd.isna(y):
            continue
        rows.append(
            {
                "team": team,
                "player": player,
                "x": _normalise_x(float(x), direction),
                "y": _normalise_y(float(y)),
            }
        )
    return pd.DataFrame(rows)


def row_at_frame(frame: pd.DataFrame, frame_number: int) -> pd.Series:
    fcol = frame_column(frame)
    exact = frame[frame[fcol].eq(frame_number)]
    if not exact.empty:
        return exact.iloc[0]
    index = (frame[fcol] - frame_number).abs().idxmin()
    return frame.loc[index]


def matching_row_by_time(frame: pd.DataFrame, target_time: float) -> pd.Series:
    tcol = time_column(frame)
    index = (frame[tcol] - target_time).abs().idxmin()
    return frame.loc[index]


def frame_positions(
    home: pd.DataFrame,
    away: pd.DataFrame,
    frame_number: int,
    direction: Direction,
) -> tuple[pd.DataFrame, tuple[float, float] | None]:
    home_row = row_at_frame(home, frame_number)
    away_row = row_at_frame(away, frame_number)
    players = pd.concat(
        [
            team_positions(home_row, home, "Home", direction),
            team_positions(away_row, away, "Away", direction),
        ],
        ignore_index=True,
    )
    ball = ball_position(home_row, direction) or ball_position(away_row, direction)
    return players, ball


def nearest_distance(point: tuple[float, float], candidates: pd.DataFrame) -> float:
    if candidates.empty:
        return float("nan")
    distances = np.hypot(candidates["x"] - point[0], candidates["y"] - point[1])
    return float(distances.min())
