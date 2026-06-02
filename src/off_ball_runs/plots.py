from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Rectangle

from off_ball_runs.data import Direction, frame_positions


def draw_pitch(ax: plt.Axes) -> None:
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect("equal")
    ax.axis("off")
    line_color = "#263238"
    ax.add_patch(Rectangle((0, 0), 105, 68, fill=False, color=line_color, linewidth=1.4))
    ax.axvline(52.5, color=line_color, linewidth=1)
    ax.add_patch(Rectangle((0, 13.84), 16.5, 40.32, fill=False, color=line_color, linewidth=1))
    ax.add_patch(Rectangle((88.5, 13.84), 16.5, 40.32, fill=False, color=line_color, linewidth=1))
    ax.add_patch(Rectangle((0, 24.84), 5.5, 18.32, fill=False, color=line_color, linewidth=1))
    ax.add_patch(Rectangle((99.5, 24.84), 5.5, 18.32, fill=False, color=line_color, linewidth=1))
    ax.add_patch(plt.Circle((52.5, 34), 9.15, fill=False, color=line_color, linewidth=1))
    ax.scatter([11, 94, 52.5], [34, 34, 34], color=line_color, s=10)
    ax.add_patch(Arc((11, 34), 18.3, 18.3, theta1=310, theta2=50, color=line_color, linewidth=1))
    ax.add_patch(Arc((94, 34), 18.3, 18.3, theta1=130, theta2=230, color=line_color, linewidth=1))


def plot_run(
    home: pd.DataFrame,
    away: pd.DataFrame,
    run: pd.Series,
    direction: Direction,
) -> plt.Figure:
    start_frame = int(run["start_frame"])
    end_frame = int(run["end_frame"])
    players, ball = frame_positions(home, away, start_frame, direction)

    fig, ax = plt.subplots(figsize=(11, 7))
    draw_pitch(ax)
    ax.set_title(
        f"{run['player']} | {run['run_type']} | score {run['space_created_score']:.1f}",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )

    attacking_team = run["attacking_team"]
    attackers = players[players["team"].eq(attacking_team)]
    defenders = players[players["team"].ne(attacking_team)]
    ax.scatter(attackers["x"], attackers["y"], c="#2563eb", s=70, label=f"{attacking_team} attackers", edgecolor="white")
    ax.scatter(defenders["x"], defenders["y"], c="#dc2626", s=70, label="Defenders", edgecolor="white")

    if ball is not None:
        ax.scatter([ball[0]], [ball[1]], c="#111827", s=80, marker="o", label="Ball")

    ax.annotate(
        "",
        xy=(float(run["end_x"]), float(run["end_y"])),
        xytext=(float(run["start_x"]), float(run["start_y"])),
        arrowprops={"arrowstyle": "->", "linewidth": 3, "color": "#16a34a"},
    )
    ax.scatter([run["start_x"]], [run["start_y"]], c="#f59e0b", s=120, edgecolor="white", zorder=5)
    ax.scatter([run["end_x"]], [run["end_y"]], c="#16a34a", s=140, edgecolor="white", zorder=5)

    ax.text(
        2,
        65,
        f"Frames {start_frame}-{end_frame} | Speed {run['max_speed_mps']:.1f} m/s | "
        f"x progression {run['x_progression_m']:.1f}m | line break {run['line_break_m']:.1f}m",
        fontsize=9,
        color="#374151",
    )
    ax.legend(loc="lower left", frameon=False)
    return fig


def plot_run_map(runs: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 7))
    draw_pitch(ax)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    if runs.empty:
        ax.text(52.5, 34, "No runs detected", ha="center", va="center", fontsize=12)
        return fig

    for _, run in runs.sort_values("space_created_score", ascending=False).head(30).iterrows():
        alpha = min(0.25 + float(run["space_created_score"]) / 120, 0.95)
        ax.annotate(
            "",
            xy=(run["end_x"], run["end_y"]),
            xytext=(run["start_x"], run["start_y"]),
            arrowprops={"arrowstyle": "->", "linewidth": 1.6, "color": "#2563eb", "alpha": alpha},
        )
    ax.text(2, 65, "Top 30 detected runs, normalised so the attacking team moves left to right.", fontsize=9)
    return fig


def plot_run_density(runs: pd.DataFrame, title: str, point: str = "end") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 7))
    draw_pitch(ax)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    if runs.empty:
        ax.text(52.5, 34, "No runs detected", ha="center", va="center", fontsize=12)
        return fig

    x_column = "end_x" if point == "end" else "start_x"
    y_column = "end_y" if point == "end" else "start_y"
    heat = ax.hist2d(
        runs[x_column],
        runs[y_column],
        bins=[np.linspace(0, 105, 22), np.linspace(0, 68, 18)],
        cmap="Blues",
        alpha=0.72,
    )
    fig.colorbar(heat[3], ax=ax, fraction=0.035, pad=0.02, label="Runs")
    ax.scatter(runs[x_column], runs[y_column], s=18, c="#0f172a", alpha=0.45, edgecolor="white", linewidth=0.25)
    ax.text(2, 65, f"Heatmap of run {point} locations.", fontsize=9, color="#374151")
    return fig


def plot_score_timeline(runs: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    if runs.empty:
        ax.text(0.5, 0.5, "No runs detected", ha="center", va="center", transform=ax.transAxes)
        return fig

    frame = runs.sort_values("start_time_s")
    colours = frame["run_type"].astype("category").cat.codes
    scatter = ax.scatter(
        frame["start_time_s"] / 60,
        frame["space_created_score"],
        c=colours,
        cmap="tab10",
        s=frame["max_speed_mps"].clip(lower=3) ** 2.2,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Match minute")
    ax.set_ylabel("Space creation score")
    ax.set_ylim(0, max(100, frame["space_created_score"].max() + 8))
    ax.grid(alpha=0.22)
    ax.text(
        0.01,
        0.96,
        "Bubble size = runner speed. Colour groups separate run types.",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#374151",
    )
    return fig


def plot_player_leaderboard(summary: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    if summary.empty:
        ax.text(0.5, 0.5, "No player summary available", ha="center", va="center", transform=ax.transAxes)
        return fig

    frame = summary.sort_values("top_score", ascending=True).tail(10)
    ax.barh(frame["player"], frame["top_score"], color="#2563eb", alpha=0.86, label="Top score")
    ax.scatter(frame["average_score"], frame["player"], color="#f59e0b", s=80, label="Average score", zorder=3)
    ax.set_xlabel("Space creation score")
    ax.set_xlim(0, max(100, frame["top_score"].max() + 8))
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, loc="lower right")
    return fig


def plot_run_type_breakdown(summary: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    if summary.empty:
        ax.text(0.5, 0.5, "No run-type summary available", ha="center", va="center", transform=ax.transAxes)
        return fig

    frame = summary.sort_values("runs", ascending=False)
    bars = ax.bar(frame["run_type"], frame["runs"], color="#16a34a", alpha=0.84)
    ax.set_ylabel("Detected runs")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.22)
    for bar, score in zip(bars, frame["average_score"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"avg {score:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return fig
