from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from off_ball_runs.data import load_game
from off_ball_runs.model import coach_notes, detect_off_ball_runs
from off_ball_runs.plots import plot_run_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect off-ball runs from Metrica sample tracking data.")
    parser.add_argument("--game-id", type=int, default=1, choices=[1, 2])
    parser.add_argument("--team", default="Home", choices=["Home", "Away"])
    parser.add_argument("--direction", default="left-to-right", choices=["left-to-right", "right-to-left"])
    parser.add_argument("--start-minute", type=float, default=0.0)
    parser.add_argument("--end-minute", type=float, default=15.0)
    parser.add_argument("--cache-dir", default="data/raw/metrica")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    game = load_game(game_id=args.game_id, cache_dir=args.cache_dir)
    runs = detect_off_ball_runs(
        home=game["home"],
        away=game["away"],
        attacking_team=args.team,
        direction=args.direction,
        start_minute=args.start_minute,
        end_minute=args.end_minute,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "detected_runs.csv", index=False)

    figure = plot_run_map(runs, f"{args.team} off-ball runs | Metrica sample game {args.game_id}")
    figure.savefig(output_dir / "run_map.png", dpi=180, bbox_inches="tight")

    print(f"Detected {len(runs)} runs.")
    print("Coach notes:")
    for note in coach_notes(runs):
        print(f"- {note}")
    print(f"Wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
