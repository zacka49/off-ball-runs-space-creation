# Off-Ball Runs & Space Creation

A tracking-data football analytics project that detects off-ball forward runs and estimates how much space they create. The aim is to show value that event data often misses: a player can improve an attack by moving defenders, opening a lane, stretching the back line, or creating a receiving option without touching the ball.

This is built as a portfolio project for football insight, performance analysis and coach-support roles.

## Why This Project Exists

The FA Insight Data Scientist role mentions player development, performance insight, advanced analytical models and communication with stakeholders. This project demonstrates those skills with tracking data:

- Parse real player and ball tracking data.
- Detect candidate off-ball runs from movement windows.
- Score each run using explainable spatial features.
- Visualise individual examples on a pitch.
- Produce coach-facing notes that translate data into tactical language.

## Data

The project uses [Metrica Sports sample data](https://github.com/metrica-sports/sample-data), which provides anonymised tracking and event data for sample matches. Data is downloaded on demand and cached locally under `data/raw/metrica`.

Metrica ask users to acknowledge the source when using the data publicly.

## Features

- On-demand Metrica sample-data downloader.
- Tracking parser for home, away and ball positions.
- Coordinate normalisation so the attacking team always moves left to right.
- Candidate off-ball run detection from sampled movement windows.
- Space creation score based on:
  - speed,
  - forward progression,
  - nearest defender separation,
  - line-breaking movement,
  - passing-lane congestion improvement,
  - final-third endpoint.
- Run labels: run in behind, underlap, wide stretch, decoy/lane opener and support run.
- Beginner-friendly Streamlit app with separate pages for getting started, visualisations, run exploration, player profiles, coach reports and method notes.
- Command-line script to export detected runs and a run map.

## Project Structure

```text
off-ball-runs-space-creation/
  README.md
  requirements.txt
  streamlit_app.py
  scripts/
    detect_runs.py
  src/
    off_ball_runs/
      __init__.py
      data.py
      model.py
      plots.py
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On macOS/Linux, use `source .venv/bin/activate` instead of the Windows activation command.

Do not run the app with `python streamlit_app.py`. Streamlit apps should be launched with `streamlit run`.

## VS Code Workflow

Open the project folder in VS Code:

```powershell
code "D:\AI projects\football-projects\off-ball-runs-space-creation"
```

Then open a new VS Code terminal and run:

```powershell
.\.venv\Scripts\activate
streamlit run streamlit_app.py
```

If VS Code asks for an interpreter, select:

```text
D:\AI projects\football-projects\off-ball-runs-space-creation\.venv\Scripts\python.exe
```

## App Pages

- `Start Here`: the default page for a first-time user, with the headline results, best example and coach notes.
- `Visualisations`: pitch run maps, start/end heatmaps, timing charts, player leaderboards and run-type breakdowns.
- `Run Explorer`: filter runs by player, type and score, then inspect individual examples on the pitch.
- `Player Profiles`: summarise which players repeatedly create off-ball value.
- `Coach Report`: turn the output into a short analyst-style report and download the run table.
- `Method & Settings`: explain the score and check that the tracking data has been parsed correctly.

## Command-Line Run

```bash
python scripts/detect_runs.py --game-id 1 --team Home --direction left-to-right --start-minute 0 --end-minute 15
```

Outputs are written to `outputs/`:

- `detected_runs.csv`
- `run_map.png`

## Portfolio Story

Use this as a short case study:

> I built a tracking-data model to detect off-ball runs and estimate space creation. The model is explainable: each run is scored using speed, forward progression, defender separation, line-breaking movement and lane congestion. The output is designed for a coach or analyst reviewing off-ball behaviour, not just for a data science notebook.

## Next Improvements

- Link detected runs to synced event data and identify whether the player received the next pass.
- Add possession phase labels such as settled attack, transition and counter-press recovery.
- Build player role profiles from repeated run types.
- Export video clip timestamps for analyst review.
