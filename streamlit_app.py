from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from off_ball_runs.data import load_game, player_pairs
from off_ball_runs.model import (
    coach_notes,
    detect_off_ball_runs,
    player_summary,
    run_type_summary,
)
from off_ball_runs.plots import plot_run, plot_run_map
from off_ball_runs.plots import (
    plot_player_leaderboard,
    plot_run_density,
    plot_run_type_breakdown,
    plot_score_timeline,
)


PRESETS = {
    "Balanced (recommended)": {
        "description": "Good first look: enough examples to discuss, without flooding the table.",
        "frame_step": 16,
        "window_seconds": 2.0,
        "min_speed_mps": 4.4,
        "min_x_progression_m": 5.0,
    },
    "More examples": {
        "description": "Looser detection for exploring movement patterns.",
        "frame_step": 18,
        "window_seconds": 2.0,
        "min_speed_mps": 3.8,
        "min_x_progression_m": 3.5,
    },
    "Only strongest runs": {
        "description": "Stricter detection for a cleaner shortlist of high-value runs.",
        "frame_step": 12,
        "window_seconds": 2.0,
        "min_speed_mps": 5.2,
        "min_x_progression_m": 8.0,
    },
}

DISPLAY_COLUMNS = [
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

FRIENDLY_COLUMNS = {
    "player": "Player",
    "runs": "Runs",
    "average_score": "Avg score",
    "top_score": "Top score",
    "main_run_type": "Main run type",
    "final_third_runs": "Final-third runs",
    "lane_opening_runs": "Lane-opening runs",
    "run_type": "Run type",
    "start_time_s": "Start, seconds",
    "end_time_s": "End, seconds",
    "space_created_score": "Score",
    "max_speed_mps": "Speed, m/s",
    "x_progression_m": "Forward movement, m",
    "nearest_defender_end_m": "Nearest defender, m",
    "line_break_m": "Line break, m",
    "lane_improvement": "Lane improvement",
}


st.set_page_config(page_title="Off-Ball Runs & Space Creation", layout="wide")


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


def sorted_runs(runs):
    return runs.sort_values("space_created_score", ascending=False).reset_index(drop=True)


def time_label(seconds: float) -> str:
    minute = int(seconds // 60)
    second = int(seconds % 60)
    return f"{minute}:{second:02d}"


def friendly(frame):
    return frame.rename(columns=FRIENDLY_COLUMNS)


with st.sidebar:
    st.title("Off-Ball Runs")
    page = st.radio(
        "Page",
        [
            "Start Here",
            "Visualisations",
            "Run Explorer",
            "Player Profiles",
            "Coach Report",
            "Method & Settings",
        ],
    )

    st.divider()
    st.header("Setup")
    game_id = st.selectbox("Metrica sample game", [1, 2])
    attacking_team = st.selectbox("Team to analyse", ["Home", "Away"])
    direction = st.selectbox("Attacking direction", ["left-to-right", "right-to-left"])
    preset_name = st.selectbox("Detection preset", list(PRESETS.keys()))

    start_minute, end_minute = st.slider(
        "Match period to analyse",
        0.0,
        90.0,
        (0.0, 15.0),
        step=1.0,
        help="Start with 15 minutes. Wider ranges take longer but give a bigger sample.",
    )
    cache_dir = st.text_input("Data cache", "data/raw/metrica")

    preset = PRESETS[preset_name].copy()
    st.caption(preset["description"])

    with st.expander("Advanced controls"):
        st.caption("You can ignore this section on a first run.")
        frame_step = st.slider("Frame sampling step", 6, 30, int(preset["frame_step"]), step=2)
        window_seconds = st.slider("Run window, seconds", 1.0, 4.0, float(preset["window_seconds"]), step=0.25)
        min_speed_mps = st.slider("Minimum speed, m/s", 3.0, 7.0, float(preset["min_speed_mps"]), step=0.1)
        min_x_progression_m = st.slider(
            "Minimum forward movement, m",
            2.0,
            15.0,
            float(preset["min_x_progression_m"]),
            step=0.5,
        )

st.title("Off-Ball Runs & Space Creation")
st.caption("A beginner-friendly tracking-data app for finding off-ball value that event data often misses.")

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
        "The app could not load the Metrica sample data. Check your internet connection, then run it again."
    )
    st.exception(exc)
    st.stop()

if runs.empty and preset_name != "More examples":
    st.info("No runs were found with the current settings, so the app tried a more sensitive fallback.")
    runs = cached_runs(
        game_id,
        cache_dir,
        attacking_team,
        direction,
        start_minute,
        end_minute,
        max(frame_step, 18),
        window_seconds,
        max(3.2, min_speed_mps - 0.8),
        max(2.5, min_x_progression_m - 2.0),
    )

home_players = len(player_pairs(game["home"]))
away_players = len(player_pairs(game["away"]))

if runs.empty:
    st.error("No runs were found. Try the 'More examples' preset or choose a wider match period.")
    st.stop()

ordered = sorted_runs(runs)
top_run = ordered.iloc[0]
players = player_summary(runs)
types = run_type_summary(runs)

metric_cols = st.columns(3)
metric_cols[0].metric("Runs", f"{len(runs):,}")
metric_cols[1].metric("Top score", f"{top_run['space_created_score']:.1f}")
metric_cols[2].metric("Top player", str(top_run["player"]))
metric_cols = st.columns(2)
metric_cols[0].metric("Avg speed", f"{runs['max_speed_mps'].mean():.1f} m/s")
metric_cols[1].metric("Final third", f"{runs['end_x'].ge(70).mean():.0%}")

if page == "Start Here":
    st.subheader("What this app does")
    st.write(
        "It looks through tracking data and finds moments where an attacker makes a fast forward run away from the ball. "
        "Each run gets a score based on speed, forward movement, defender separation, line-breaking movement and whether "
        "the passing lane becomes cleaner."
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Best example in this sample")
        st.write(
            f"{top_run['player']} makes a {top_run['run_type']} at {time_label(top_run['start_time_s'])}. "
            f"It scores {top_run['space_created_score']:.1f}/100."
        )
        st.pyplot(plot_run(game["home"], game["away"], top_run, direction), clear_figure=True)
    with right:
        st.subheader("Run types found")
        st.dataframe(friendly(types), use_container_width=True, hide_index=True)
        st.subheader("First coach notes")
        for note in coach_notes(runs):
            st.write(f"- {note}")

elif page == "Run Explorer":
    st.subheader("Explore individual runs")

    filter_cols = st.columns(3)
    player_options = ["All"] + sorted(runs["player"].unique().tolist())
    type_options = ["All"] + sorted(runs["run_type"].unique().tolist())
    selected_player = filter_cols[0].selectbox("Player", player_options)
    selected_type = filter_cols[1].selectbox("Run type", type_options)
    min_score = filter_cols[2].slider("Minimum score", 0, 100, 0)

    filtered = runs.copy()
    if selected_player != "All":
        filtered = filtered[filtered["player"].eq(selected_player)]
    if selected_type != "All":
        filtered = filtered[filtered["run_type"].eq(selected_type)]
    filtered = filtered[filtered["space_created_score"].ge(min_score)]

    if filtered.empty:
        st.warning("No runs match those filters.")
    else:
        filtered_ordered = sorted_runs(filtered)
        st.pyplot(plot_run_map(filtered_ordered, "Filtered run map"), clear_figure=True)

        selected = st.selectbox(
            "Pick a run to inspect",
            options=list(range(len(filtered_ordered))),
            format_func=lambda i: (
                f"{filtered_ordered.loc[i, 'player']} | {time_label(filtered_ordered.loc[i, 'start_time_s'])} | "
                f"{filtered_ordered.loc[i, 'run_type']} | {filtered_ordered.loc[i, 'space_created_score']:.1f}"
            ),
        )
        run = filtered_ordered.loc[selected]
        st.pyplot(plot_run(game["home"], game["away"], run, direction), clear_figure=True)
        st.dataframe(friendly(filtered_ordered[DISPLAY_COLUMNS]), use_container_width=True, hide_index=True)

elif page == "Visualisations":
    st.subheader("Visualise the off-ball runs")
    st.write(
        "Use this page to see the story at a glance: where runs finish, when the best runs happen, "
        "which players stand out and which run types appear most often."
    )

    visual_cols = st.columns(3)
    visual_player_options = ["All players"] + sorted(runs["player"].unique().tolist())
    visual_type_options = ["All run types"] + sorted(runs["run_type"].unique().tolist())
    visual_player = visual_cols[0].selectbox("Show player", visual_player_options)
    visual_type = visual_cols[1].selectbox("Show run type", visual_type_options)
    visual_min_score = visual_cols[2].slider("Visual minimum score", 0, 100, 20)

    visual_runs = runs.copy()
    if visual_player != "All players":
        visual_runs = visual_runs[visual_runs["player"].eq(visual_player)]
    if visual_type != "All run types":
        visual_runs = visual_runs[visual_runs["run_type"].eq(visual_type)]
    visual_runs = visual_runs[visual_runs["space_created_score"].ge(visual_min_score)]

    if visual_runs.empty:
        st.warning("No runs match those visual filters. Lower the minimum score or choose All players.")
    else:
        st.caption(f"Showing {len(visual_runs)} of {len(runs)} detected runs.")

        map_tab, heat_tab, timing_tab, profile_tab = st.tabs(
            ["Run Map", "Heatmaps", "Timing", "Players & Types"]
        )

        with map_tab:
            st.pyplot(plot_run_map(visual_runs, "Filtered run arrows"), clear_figure=True)

        with heat_tab:
            left, right = st.columns(2)
            with left:
                st.pyplot(plot_run_density(visual_runs, "Where runs start", point="start"), clear_figure=True)
            with right:
                st.pyplot(plot_run_density(visual_runs, "Where runs finish", point="end"), clear_figure=True)

        with timing_tab:
            st.pyplot(plot_score_timeline(visual_runs, "When high-value runs happen"), clear_figure=True)
            st.write(
                "Look for clusters: repeated high scores in a short spell can point to a tactical pattern, "
                "a matchup advantage or a moment where the defensive line was vulnerable."
            )

        with profile_tab:
            visual_players = player_summary(visual_runs)
            visual_types = run_type_summary(visual_runs)
            left, right = st.columns(2)
            with left:
                st.pyplot(plot_player_leaderboard(visual_players, "Player leaderboard"), clear_figure=True)
            with right:
                st.pyplot(plot_run_type_breakdown(visual_types, "Run-type breakdown"), clear_figure=True)

elif page == "Player Profiles":
    st.subheader("Which players create off-ball value?")
    st.write("Use this page to move from individual examples to player-level patterns.")
    st.dataframe(friendly(players), use_container_width=True, hide_index=True)

    selected_player = st.selectbox("Player detail", players["player"].tolist())
    player_runs = sorted_runs(runs[runs["player"].eq(selected_player)])
    st.pyplot(plot_run_map(player_runs, f"{selected_player} run map"), clear_figure=True)
    st.dataframe(friendly(player_runs[DISPLAY_COLUMNS]), use_container_width=True, hide_index=True)

elif page == "Coach Report":
    st.subheader("Coach-facing report")
    st.write(
        "This page is written as a short analyst note: what happened, why it matters, and which examples to review."
    )

    for note in coach_notes(runs):
        st.write(f"- {note}")

    st.subheader("Recommended clips to review")
    clip_list = ordered.head(10).copy()
    clip_list["time"] = clip_list["start_time_s"].apply(time_label)
    st.dataframe(
        friendly(clip_list[
            [
                "time",
                "player",
                "run_type",
                "space_created_score",
                "x_progression_m",
                "nearest_defender_end_m",
                "lane_improvement",
            ]
        ]),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download detected runs CSV",
        data=ordered.to_csv(index=False).encode("utf-8"),
        file_name="detected_off_ball_runs.csv",
        mime="text/csv",
    )

elif page == "Method & Settings":
    st.subheader("How to read the output")
    st.write(
        "The score is not a magic truth number. It is an explainable ranking tool that helps an analyst find useful clips quickly."
    )

    st.markdown(
        """
        **What the score rewards**

        - Fast movement over the selected run window.
        - Forward movement toward goal.
        - Separation from the nearest defender.
        - Movement beyond the opposition defensive line.
        - Runs that reduce blockers in the passing lane from ball to runner.
        - Runs finishing in the final third.
        """
    )

    st.subheader("Current data check")
    data_cols = st.columns(4)
    data_cols[0].metric("Home players parsed", home_players)
    data_cols[1].metric("Away players parsed", away_players)
    data_cols[2].metric("Preset", preset_name)
    data_cols[3].metric("Window", f"{start_minute:.0f}-{end_minute:.0f} min")

    st.subheader("Current run-type summary")
    st.dataframe(friendly(types), use_container_width=True, hide_index=True)
