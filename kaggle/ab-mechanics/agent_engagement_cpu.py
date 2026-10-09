"""Bounded proxy-agent adoption probe on a real offline ARC-AGI-3 SDK game."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from arc_agi import Arcade, OperationMode
from arcengine import GameAction

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream/ARC3-Inference"))
from inference.agent.python_tool_sandbox import run_sandboxed_python
from inference.agent.tool_agent import ToolAgent
from inference.utils.grid_utils import ARC_COLOR_CHARS
from install_policy import PROMPT

GAME_ID = "ls20-9607627b"
SEED = 0
MAX_DECISIONS = 5
MAX_EVAL_ACTIONS = 4
OUT = HERE / "agent-choice-claude-login-cpu-20261009.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def board_of(game):
    return game.observation_space.frame[-1].tolist()


def ascii_of(board):
    return ["".join(ARC_COLOR_CHARS[x] if x < len(ARC_COLOR_CHARS) else "?" for x in row) for row in board]


def differences(before, after):
    changed = [(r, c) for r in range(64) for c in range(64) if before[r][c] != after[r][c]]
    if not changed:
        return {"changed_cells": 0, "interior_changed_cells": 0, "crop": []}
    r0, r1 = max(0, min(r for r, _ in changed) - 2), min(64, max(r for r, _ in changed) + 3)
    c0, c1 = max(0, min(c for _, c in changed) - 2), min(64, max(c for _, c in changed) + 3)
    # Never hand a full board to the model; the game agent did not get raw numeric pixels.
    r1, c1 = min(r1, r0 + 12), min(c1, c0 + 12)
    old, new = ascii_of(before), ascii_of(after)
    return {"changed_cells": len(changed),
            "interior_changed_cells": sum(4 <= r < 60 and 4 <= c < 60 for r, c in changed),
            "bbox": [r0, c0, r1, c1],
            "before_crop": [s[c0:c1] for s in old[r0:r1]],
            "after_crop": [s[c0:c1] for s in new[r0:r1]]}


def parse_code(response):
    # The model may decline or return prose. Never manufacture a call.
    match = re.search(r"```python\s*([\s\S]*?)```", response)
    return match.group(1).strip() if match else None


def main():
    arcade = Arcade(operation_mode=OperationMode.OFFLINE,
                    environments_dir="/Users/morkeeth/CODE/arc-agi-3-probe-data/environment_files")
    game = arcade.make(GAME_ID, seed=SEED)
    opening = []
    sdk_events = []
    action_count = 0

    def frame():
        observation = game.observation_space
        return {"grid": board_of(game), "level": int(observation.levels_completed),
                "step": len(opening) + len(sdk_events), "shape": [64, 64],
                "ascii": "\n".join(ascii_of(board_of(game)))}

    history_entries = [{"action": "", "frame": frame(), "result": {}}]
    for _ in range(2):
        before = board_of(game)
        game.step(GameAction.ACTION3)  # LEFT, exactly the first two retained actions.
        opening.append({"action": "LEFT", "diff": differences(before, board_of(game))})
        history_entries.append({"action": "ACTION3", "frame": frame(), "result": {"executed": True}})

    def live_frame():
        return {"grid": board_of(game), "level": int(game.observation_space.levels_completed)}

    def handler(actions, **kwargs):
        nonlocal action_count
        if len(actions) != 1 or action_count >= MAX_EVAL_ACTIONS:
            raise RuntimeError("Bounded CPU action limit")
        name = actions[0]["action"]
        if name not in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
            raise RuntimeError("Only four real SDK directions are allowed")
        before = frame()
        result = game.step(getattr(GameAction, name))
        action_count += 1
        sdk_events.append({"action": name, "before_sha256": hashlib.sha256(json.dumps(before["grid"]).encode()).hexdigest(),
                           "after_sha256": hashlib.sha256(json.dumps(board_of(game)).encode()).hexdigest(),
                           "diff": differences(before["grid"], board_of(game)),
                           "level": int(game.observation_space.levels_completed)})
        history_entries.append({"action": name, "frame": frame(), "result": {"executed": True,
                                                                                "level_completed": bool(result.levels_completed)}})
        return {"action_result": {"executed": True, "executed_count": 1,
                                  "level_completed": bool(result.levels_completed)},
                "state": {"current_frame": frame(), "history": history_entries,
                          "valid_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]}}

    class NoSession:
        def _ensure_session(self, state_path):
            return None

    tool_schema = ToolAgent._tools(NoSession(), HERE / 'proxy-state.json')[0]
    intro = (PROMPT + "\nYou are solving one public grid game with the normal Python action tool. "
             "The current frame is 64x64. `mechanics.view()` returns `{'level': int, 'grid': list[list[int]]}`; "
             "use `mechanics.view()['grid']` inside Python, never print the full grid. "
             "The installed Python tool schema is exactly: " + json.dumps(tool_schema) + " "
             "LEFT=ACTION3, RIGHT=ACTION4, UP=ACTION1, DOWN=ACTION2. "
             "Do not assume your previous calls ran unless a tool result says so. "
             "Reply with one executable Python code block for your next tool call. "
             "You may inspect, act, or use mechanics; choose based on evidence. "
             "Do not print the full board.\n")
    record = {"rule": (HERE / "RULE-agent-choice-claude-login-cpu-20261009.md").read_text(),
              "game_id": GAME_ID, "seed": SEED, "opening": opening, "turns": [],
              "installed_tool_schema": tool_schema,
              "source_sha256": {name: digest(HERE / name) for name in
                                 ("agent_engagement_cpu.py", "install_policy.py", "mechanics_runtime.py")},
              "model_kind": "Claude Sonnet existing-login isolated CLI proxy, not the Kaggle model", "sdk_events": sdk_events}
    mechanics_state = {"models": {}, "events": [], "level": None}
    previous_result = "No prior tool call in this probe."
    last_gameplay_changed = bool(opening[-1]['diff']['interior_changed_cells'])
    for turn in range(MAX_DECISIONS):
        checkpoint = ("\nMechanics decision checkpoint: the last real action changed gameplay. "
                      "Choose whether a full-grid predictor is viable now. mechanics.view() gives the current numeric frame inside Python; "
                      "if you have a supported rule, register it and use mechanics.test(action) for a prospective real action. "
                      "If not, continue the ordinary action policy. This is a choice, not a requirement to call mechanics."
                      if last_gameplay_changed else "")
        prompt = intro + "\nObserved opening (past only): " + json.dumps(opening) + \
                 "\nCurrent level: " + str(game.observation_space.levels_completed) + \
                 "\nMost recent tool result: " + previous_result + \
                 checkpoint + "\nDecide your next Python call now."
        row = {"prompt": prompt, "model": "sonnet", "cli_mode": "safe restricted nonpersistent"}
        try:
            model_env = os.environ.copy()
            model_env.pop('ANTHROPIC_API_KEY', None)
            model_process = subprocess.run(
                ['claude', '-p', '--safe-mode', '--restricted', '--strict-mcp-config',
                 '--no-session-persistence', '--model', 'sonnet', '--effort', 'low',
                 '--tools', '', '--output-format', 'json', '--system-prompt',
                 'You are the grid-game agent in a bounded local experiment. Follow only the game prompt. Return one executable Python code block and no other text.',
                 prompt],
                env=model_env, cwd='/tmp', capture_output=True, text=True, timeout=90,
            )
            if model_process.returncode:
                raise RuntimeError(model_process.stderr[-2000:])
            row['model_result'] = json.loads(model_process.stdout)
            if row['model_result'].get('is_error'):
                raise RuntimeError(str(row['model_result'].get('result')))
            response = row['model_result'].get('result', '')
        except Exception as exc:
            row["api_error"] = str(exc)
            record["turns"].append(row)
            break
        code = parse_code(response)
        row["emitted_code"] = code
        if not code:
            record["turns"].append(row)
            break
        previous_action_count = action_count
        sandbox = run_sandboxed_python(code=code, timeout_seconds=20,
                                       initial_state={"current_frame": frame(), "history": history_entries,
                                                      "valid_actions": ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]},
                                       action_handler=handler, mechanics_state=mechanics_state,
                                       mechanics_frame_handler=live_frame)
        row["sandbox_result"] = sandbox
        mechanics_state = sandbox.get("mechanics_state") or mechanics_state
        if action_count > previous_action_count:
            last_gameplay_changed = bool(sdk_events[-1]['diff']['interior_changed_cells'])
        previous_result = json.dumps({"stdout": sandbox.get("stdout"), "result": sandbox.get("result"),
                                      "error": sandbox.get("error"),
                                      "action_results": sandbox.get("action_results")})[:5000]
        record["turns"].append(row)
        if action_count >= MAX_EVAL_ACTIONS:
            break
    record["mechanics_state"] = mechanics_state
    record["sdk_action_count"] = action_count
    record["sdk_history"] = history_entries
    code_text = "\n".join(row.get("emitted_code") or "" for row in record["turns"])
    events = mechanics_state.get("events") or []
    record["observed_calls"] = {name: int(mechanics_state.get('counts', {}).get(name, 0))
                                for name in ("view", "register", "test", "plan", "execute", "status")}
    record["primary_pass"] = (bool(record["observed_calls"]["register"] and record["observed_calls"]["test"])
                              and any(e.get("status") == "observed" for e in events))
    record["full_chain_pass"] = bool(record["primary_pass"] and record["observed_calls"]["plan"]
                                     and record["observed_calls"]["execute"])
    OUT.write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps({"artifact": str(OUT), "game": GAME_ID, "turns": len(record["turns"]),
                      "sdk_actions": action_count, "calls": record["observed_calls"],
                      "event_count": len(events), "primary_pass": record["primary_pass"],
                      "full_chain_pass": record["full_chain_pass"]}))


if __name__ == "__main__":
    main()
