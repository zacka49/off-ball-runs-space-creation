from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from off_ball_runs.data import load_game
from off_ball_runs.model import coach_notes, detect_off_ball_runs
from off_ball_runs.plots import plot_run, plot_run_map


st.set_page_config(page_title="Off-Ball Runs & Space Creation", layout="wide")
st.title("Off-Ball Runs & Space Creation")
st.caption("Tracking-data project for detecting off-ball value that event data often misses.")


@st.cache_data(show_spinner=True)
def cached_game(game_id: int, cache_dir: str):
    return load_game(game_id=game_id, cache_dir=cache_dir)


@st.cache_data(show_spinner=True)
def cached_runs(
    game_id: int,
    cache_dir: str,
    attacking_team: str,
    direction: str,
    start_minute: float,
    end_minute: float,
    frame_step: int,
    window_seconds: float,
    min_speed_mps: float,
    min_x_progression_m: float,
):
    game = load_game(game_id=game_id, cache_dir=cache_dir)
    return detect_off_ball_runs(
        home=game["home"],
        away=game["away"],
        attacking_team=attacking_team,
        direction=direction,
        start_minute=start_minute,
        end_minute=end_minute,
        frame_step=frame_step,
        window_seconds=window_seconds,
        min_speed_mps=min_speed_mps,
        min_x_progression_m=min_x_progression_m,
    )


with st.sidebar:
    st.header("Data")
    game_id = st.selectbox("Metrica sample game", [1, 2])
    cache_dir = st.text_input("Cache directory", "data/raw/metrica")
    attacking_team = st.selectbox("Attacking team", ["Home", "Away"])
    direction = st.selectbox("Attacking direction", ["left-to-right", "right-to-left"])

    st.header("Detection")
    start_minute, end_minute = st.slider("Time window, minutes", 0.0, 90.0, (0.0, 15.0), step=1.0)
    frame_step = st.slider("Frame sampling step", 6, 30, 12, step=2)
    window_seconds = st.slider("Run window, seconds", 1.0, 4.0, 2.0, step=0.25)
    min_speed_mps = st.slider("Minimum speed, m/s", 3.5, 7.0, 4.8, step=0.1)
    min_x_progression_m = st.slider("Minimum forward movement, m", 3.0, 15.0, 6.0, step=0.5)

try:
    game = cached_game(game_id, cache_dir)
    runs = cached_runs(
        game_id,
        cache_dir,
        attacking_team,
        direction,
        start_minute,
        end_minute,
        frame_step,
        window_seconds,
        min_speed_mps,
        min_x_progression_m,
    )
except Exception as exc:
    st.error(
        "Could not load Metrica sample data. Check your internet connection, then rerun the app. "
        f"Details: {exc}"
    )
    st.stop()

if runs.empty:
    st.warning("No runs found. Try a wider time window or slightly lower speed/progression thresholds.")
    st.stop()

top_run = runs.sort_values("space_created_score", ascending=False).iloc[0]
metric_cols = st.columns(5)
metric_cols[0].metric("Detected runs", f"{len(runs):,}")
metric_cols[1].metric("Top score", f"{top_run['space_created_score']:.1f}")
metric_cols[2].metric("Top player", str(top_run["player"]))
metric_cols[3].metric("Avg speed", f"{runs['max_speed_mps'].mean():.1f} m/s")
metric_cols[4].metric("Final-third finish", f"{runs['end_x'].ge(70).mean():.0%}")

tab_map, tab_detail, tab_table, tab_notes = st.tabs(["Run Map", "Run Detail", "Run Table", "Coach Notes"])

with tab_map:
    st.subheader("Where off-ball value appears")
    st.pyplot(
        plot_run_map(runs, f"{attacking_team} off-ball runs | Metrica sample game {game_id}"),
        clear_figure=True,
    )

with tab_detail:
    st.subheader("Individual run audit")
    ordered = runs.sort_values("space_created_score", ascending=False).reset_index(drop=True)
    selected = st.selectbox(
        "Select run",
        options=list(range(len(ordered))),
        format_func=lambda i: (
            f"{ordered.loc[i, 'player']} | {ordered.loc[i, 'start_time_s']:.1f}s | "
            f"{ordered.loc[i, 'run_type']} | {ordered.loc[i, 'space_created_score']:.1f}"
        ),
    )
    run = ordered.loc[selected]
    st.pyplot(plot_run(game["home"], game["away"], run, direction), clear_figure=True)
    st.dataframe(run.to_frame("value"), use_container_width=True)

with tab_table:
    st.subheader("Detected runs")
    columns = [
        "player",
        "start_time_s",
        "end_time_s",
        "run_type",
        "space_created_score",
        "max_speed_mps",
        "x_progression_m",
        "nearest_defender_end_m",
        "line_break_m",
        "lane_improvement",
    ]
    st.dataframe(runs[columns].sort_values("space_created_score", ascending=False), use_container_width=True, hide_index=True)

with tab_notes:
    st.subheader("Coach-facing notes")
    for note in coach_notes(runs):
        st.write(f"- {note}")

    st.subheader("Method")
    st.write(
        "Each candidate run is scored from speed, forward progression, defender separation, line-breaking movement, "
        "and whether the ball-to-runner passing lane becomes less congested. Coordinates are normalised so the "
        "attacking team always moves left to right."
    )
