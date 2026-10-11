"""One frozen, paired CPU gameplay proxy for compact decision state."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from arc_agi import Arcade, OperationMode
from arcengine import GameAction

from agent_engagement_cpu import GAME_ID, ascii_of, board_of, differences

HERE = Path(__file__).resolve().parent
RULE = HERE / "RULE-compact-decision-cpu-20261011.md"
OUT = HERE / "compact-decision-cpu-20261011.json"
GAME_DIR = "/Users/morkeeth/CODE/arc-agi-3-probe-data/environment_files"
OPENING = ["ACTION3", "ACTION3", "ACTION1", "ACTION1", "ACTION4", "ACTION2"]
ALLOWED = {f"ACTION{i}" for i in range(1, 5)}
MAX_ACTIONS = 12


def new_game(seed):
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAME_DIR)
    game = arcade.make(GAME_ID, seed=seed)
    history = []
    for name in OPENING:
        before = board_of(game)
        previous_level = int(game.observation_space.levels_completed)
        result = game.step(getattr(GameAction, name))
        history.append({"action": name, "diff": differences(before, board_of(game)),
                        "level_change": int(game.observation_space.levels_completed) - previous_level,
                        "state": str(result.state)})
    return game, history


def compact(history):
    return [{"action": row["action"],
             "changed_cells": row["diff"]["changed_cells"],
             "interior_changed_cells": row["diff"]["interior_changed_cells"],
             "level_change": row["level_change"]} for row in history]


def propose(seed, arm, history, board):
    past = history if arm == "control" else compact(history)
    prompt = (
        "You are controlling a public offline grid game. Return one JSON object with key "
        '"actions" containing exactly 12 directions chosen from ACTION1, ACTION2, ACTION3, ACTION4. '
        "ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT. "
        "Use only the observed past and current frame, not an imagined future. "
        f"Game {GAME_ID}; seed {seed}; past observations: {json.dumps(past)}; "
        f"current 64-row letter grid: {json.dumps(ascii_of(board))}"
    )
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    start = time.monotonic()
    proc = subprocess.run(
        ["claude", "-p", "--safe-mode", "--restricted", "--strict-mcp-config",
         "--no-session-persistence", "--model", "sonnet", "--effort", "low", "--tools", "",
         "--output-format", "json", "--system-prompt",
         "Return only one JSON object with exactly twelve game actions; no Markdown.", prompt],
        env=env, cwd="/tmp", capture_output=True, text=True, timeout=90)
    row = {"prompt": prompt, "returncode": proc.returncode, "stderr": proc.stderr[-1000:],
           "elapsed_seconds": time.monotonic() - start}
    if proc.returncode:
        row["error"] = "CLI exit"
        return row, None
    try:
        row["model_result"] = json.loads(proc.stdout)
        if row["model_result"].get("is_error"):
            raise ValueError("model error")
        raw = row["model_result"].get("result", "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        actions = data["actions"]
        if not isinstance(actions, list) or len(actions) != MAX_ACTIONS or any(a not in ALLOWED for a in actions):
            raise ValueError("not exactly 12 allowed actions")
        row["actions"] = actions
        return row, actions
    except (ValueError, KeyError, TypeError) as exc:
        row["error"] = repr(exc)
        return row, None


def play(seed, actions):
    game, history = new_game(seed)
    start_level = int(game.observation_space.levels_completed)
    events = []
    for name in actions:
        before = board_of(game)
        outcome = game.step(getattr(GameAction, name))
        events.append({"action": name, "diff": differences(before, board_of(game)),
                       "level": int(game.observation_space.levels_completed),
                       "state": str(outcome.state)})
        if int(game.observation_space.levels_completed) != start_level or "GAME_OVER" in str(outcome.state):
            break
    return {"opening": history, "events": events, "actions": len(events),
            "start_level": start_level, "end_level": int(game.observation_space.levels_completed),
            "levels_gained": int(game.observation_space.levels_completed) - start_level}


def main():
    record = {"rule": RULE.read_text(), "game_id": GAME_ID,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (RULE, Path(__file__))}, "rows": []}
    for seed in (0, 1):
        for arm in ("control", "compact"):
            game, history = new_game(seed)
            model, actions = propose(seed, arm, history, board_of(game))
            row = {"seed": seed, "arm": arm, "model": model}
            if actions is not None:
                row["gameplay"] = play(seed, actions)
            record["rows"].append(row)
            OUT.write_text(json.dumps(record, indent=2, default=str))
    paired = all("gameplay" in row for row in record["rows"])
    by_seed = {seed: {row["arm"]: row for row in record["rows"] if row["seed"] == seed}
               for seed in (0, 1)}
    deltas = ([by_seed[seed]["compact"]["gameplay"]["levels_gained"] -
               by_seed[seed]["control"]["gameplay"]["levels_gained"] for seed in (0, 1)]
              if paired else [])
    record["valid"] = paired
    record["level_deltas"] = deltas
    record["gate_pass"] = bool(paired and max(deltas) >= 1 and min(deltas) >= 0 and all(
        by_seed[seed]["compact"]["gameplay"]["actions"] <=
        by_seed[seed]["control"]["gameplay"]["actions"] for seed in (0, 1)))
    OUT.write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"artifact": str(OUT), "valid": paired, "level_deltas": deltas,
                      "gate_pass": record["gate_pass"]}))


if __name__ == "__main__":
    main()
