"""PR01: a fake ARC-AGI-3 game with known dead actions, for tests.

16x16 level, upscaled by the camera to 64x64 (4 px per cell).
- ACTION1 (UP): always a no-op for gameplay. Only the timer moves.
- ACTION2 (DOWN): no-op for gameplay. Only the timer moves.
- ACTION4 (RIGHT): moves the player one cell right. Blocked (no-op) at the wall.
- ACTION3 (LEFT): moves the player one cell left. Blocked at the wall.
- ACTION6 (click): clicking the target cell wins the level. Anywhere else: no-op.
Every action shrinks the timer bar in the top row by one cell. That is HUD.
"""

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

N = 16
PLAYER_Y = 8
TARGET = (12, 12)  # x, y

levels = [Level(sprites=[], grid_size=(N, N)), Level(sprites=[], grid_size=(N, N))]


class Pr01(ARCBaseGame):
    def __init__(self) -> None:
        self._x = 2
        self._t = N
        super().__init__(
            game_id="pr01",
            levels=levels,
            camera=Camera(background=0, letter_box=5),
            available_actions=[1, 2, 3, 4, 6],
        )

    def _draw(self) -> None:
        lvl = self.current_level
        lvl.remove_all_sprites()
        if self._t > 0:
            lvl.add_sprite(Sprite(pixels=[[8] * self._t], name="timer").set_position(0, 0))
        lvl.add_sprite(Sprite(pixels=[[9]], name="player").set_position(self._x, PLAYER_Y))
        lvl.add_sprite(Sprite(pixels=[[11]], name="target").set_position(*TARGET))

    def on_set_level(self, level: Level) -> None:
        self._x = 2
        self._t = N
        self._draw()

    def step(self) -> None:
        aid = self.action.id
        if aid == GameAction.ACTION4 and self._x < N - 1:
            self._x += 1
        elif aid == GameAction.ACTION3 and self._x > 0:
            self._x -= 1
        elif aid == GameAction.ACTION6:
            x = int(self.action.data.get("x", -1)) // 4
            y = int(self.action.data.get("y", -1)) // 4
            if (x, y) == TARGET:
                self._t = max(0, self._t - 1)
                self.next_level()
                self.complete_action()
                return
        self._t = max(0, self._t - 1)
        self._draw()
        self.complete_action()
