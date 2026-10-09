"""Two-seed, paired public-offline ARC KAGGLE forced-step mechanics proxy."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from arc_agi import Arcade, OperationMode
from arcengine import GameAction

from agent_engagement_cpu import GAME_ID, ascii_of, board_of, differences
from mechanics_runtime import MechanicsHost

HERE = Path(__file__).resolve().parent
RULE = HERE / "RULE-forced-steps-cpu-20261010.md"
OUT = HERE / "forced-steps-cpu-20261010.json"
GAME_DIR = "/Users/morkeeth/CODE/arc-agi-3-probe-data/environment_files"
ACTIONS = {f"ACTION{i}" for i in range(1, 5)}
MAX_ACTIONS = 4


def new_game(seed):
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=GAME_DIR)
    game = arcade.make(GAME_ID, seed=seed)
    opening = []
    for _ in range(2):
        before = board_of(game)
        game.step(GameAction.ACTION3)
        opening.append({"action": "ACTION3", "diff": differences(before, board_of(game))})
    return game, opening


def proposal(seed, opening, grid):
    prompt = (
        "You are a grid-game agent in a bounded local experiment. Respond with one JSON object only: "
        "{\"actions\":[four of ACTION1,ACTION2,ACTION3,ACTION4],\"predictor\":\"def predict(grid, action): ...\"}. "
        "The predictor must be one standalone, pure Python function with exactly those two arguments, "
        "returning a complete predicted 64x64 numeric grid or None if unsupported. Do not invent certainty. "
        "ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT. "
        "Choose the next four game actions from this real past-only evidence. "
        f"Game {GAME_ID}, seed {seed}. Two LEFT transitions: {json.dumps(opening)}. "
        "Current frame in letter colors (64 rows): " + json.dumps(ascii_of(grid))
    )
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    process = subprocess.run(
        ["claude", "-p", "--safe-mode", "--restricted", "--strict-mcp-config",
         "--no-session-persistence", "--model", "sonnet", "--effort", "low",
         "--tools", "", "--output-format", "json", "--system-prompt",
         "Return one JSON object with actions and predictor. No Markdown or explanation.", prompt],
        env=env, cwd="/tmp", capture_output=True, text=True, timeout=90,
    )
    row = {"prompt": prompt, "returncode": process.returncode,
           "stderr": process.stderr[-1000:]}
    if process.returncode:
        return row, None
    row["cli_result"] = json.loads(process.stdout)
    if row["cli_result"].get("is_error"):
        return row, None
    raw = row["cli_result"].get("result", "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        content = json.loads(raw)
        actions = content["actions"]
        source = content["predictor"]
        if not isinstance(actions, list) or len(actions) != MAX_ACTIONS or any(a not in ACTIONS for a in actions):
            raise ValueError("Expected four allowed actions")
        if not isinstance(source, str):
            raise ValueError("Expected predictor source string")
        row["content"] = content
        return row, content
    except (ValueError, KeyError, TypeError) as exc:
        row["parse_error"] = str(exc)
        return row, None


def play(seed, actions, source, forced):
    game, opening = new_game(seed)
    start_level = int(game.observation_space.levels_completed)
    sdk_events = []
    state = {"current_frame": {"grid": board_of(game), "level": start_level}}

    def current():
        return {"grid": board_of(game), "level": int(game.observation_space.levels_completed)}

    def act(items, **_kwargs):
        if len(items) != 1 or len(sdk_events) >= MAX_ACTIONS:
            raise RuntimeError("SDK action cap")
        name = items[0]["action"]
        if name not in ACTIONS:
            raise ValueError("Invalid SDK action")
        before = board_of(game)
        outcome = game.step(getattr(GameAction, name))
        event = {"action": name, "diff": differences(before, board_of(game)),
                 "level": int(game.observation_space.levels_completed),
                 "level_completed": bool(outcome.levels_completed)}
        sdk_events.append(event)
        return {"action_result": {"executed": True, "executed_count": 1,
                                  "level_completed": event["level_completed"]},
                "state": {"current_frame": current()}}

    result = {"seed": seed, "arm": "forced" if forced else "plain",
              "opening": opening, "start_level": start_level, "sdk_events": sdk_events,
              "stages": [], "error": None}
    began = time.monotonic()
    if forced:
        host = MechanicsHost({"models": {}, "events": [], "level": None}, state, act, current)
        try:
            name = host.dispatch({"method": "register", "args": [source]})
            result["stages"].append({"register": name})
        except Exception as exc:
            result["error"] = "register: " + repr(exc)
            result["mechanics_state"] = host.mechanics.state
            result["elapsed_seconds"] = time.monotonic() - began
            return result
        for action in actions:
            if len(sdk_events) >= MAX_ACTIONS or current()["level"] != start_level:
                break
            stage = {"proposed_action": action}
            try:
                stage["test"] = host.dispatch({"method": "test", "args": [action]})
                if current()["level"] != start_level:
                    result["stages"].append(stage)
                    break
                predicted = host.mechanics._predict(name, current()["grid"], action)
                stage["target_supported"] = predicted is not None
                target = predicted if predicted is not None else current()["grid"]
                stage["plan"] = host.dispatch({"method": "plan", "args": [name, target, sorted(ACTIONS), 3, 64]})
                remaining = MAX_ACTIONS - len(sdk_events)
                planned = stage["plan"].get("actions", [])[:remaining]
                stage["execute"] = host.dispatch({"method": "execute", "args": [name, planned]})
            except Exception as exc:
                result["error"] = "forced stage: " + repr(exc)
                result["stages"].append(stage)
                break
            result["stages"].append(stage)
        result["mechanics_state"] = host.mechanics.state
    else:
        for action in actions:
            if len(sdk_events) >= MAX_ACTIONS or current()["level"] != start_level:
                break
            try:
                act([{"action": action}])
            except Exception as exc:
                result["error"] = "plain action: " + repr(exc)
                break
    result["end_level"] = current()["level"]
    result["levels_completed"] = result["end_level"] - start_level
    result["sdk_action_count"] = len(sdk_events)
    result["elapsed_seconds"] = time.monotonic() - began
    return result


def main():
    record = {"rule": RULE.read_text(), "game_id": GAME_ID,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [RULE, Path(__file__), HERE / "mechanics_runtime.py"]},
              "seeds": []}
    for seed in (0, 1):
        game, opening = new_game(seed)
        model, content = proposal(seed, opening, board_of(game))
        row = {"seed": seed, "model": model}
        if content is not None:
            row["plain"] = play(seed, content["actions"], content["predictor"], False)
            row["forced"] = play(seed, content["actions"], content["predictor"], True)
        record["seeds"].append(row)
        OUT.write_text(json.dumps(record, indent=2, default=str))
    valid = all("plain" in row and "forced" in row and
                not row["plain"]["error"] and not row["forced"]["error"]
                for row in record["seeds"])
    deltas = [row["forced"]["levels_completed"] - row["plain"]["levels_completed"]
              for row in record["seeds"] if "plain" in row and "forced" in row
              and "levels_completed" in row["plain"] and "levels_completed" in row["forced"]]
    record["bar_pass"] = bool(valid and len(deltas) == 2 and max(deltas) >= 1 and min(deltas) >= 0 and
                              all(row[arm]["sdk_action_count"] <= MAX_ACTIONS
                                  for row in record["seeds"] for arm in ("plain", "forced")))
    OUT.write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"artifact": str(OUT), "valid": valid, "level_deltas": deltas,
                      "bar_pass": record["bar_pass"]}))


if __name__ == "__main__":
    main()
