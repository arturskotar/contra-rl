"""Discrete controller action presets for Contra."""

RIGHT_ONLY = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
]

SIMPLE_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
]

COMPLEX_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["A"],
    ["left"],
    ["left", "A"],
    ["up", "left", "A"],
    ["down", "left", "A"],
    ["down", "left", "B"],
    ["up", "right", "A"],
    ["up", "right", "B"],
    ["right", "B"],
    ["down", "right", "A"],
    ["down", "right", "B"],
    ["left", "B"],
    ["up", "left", "B"],
    ["left", "A", "B"],
    ["down"],
    ["down", "A"],
    ["down", "B"],
    ["up"],
    ["up", "A"],
    ["up", "B"],
    ["right", "A", "B"],
    ["B"],
]

ACTION_SETS = {
    "RIGHT_ONLY": RIGHT_ONLY,
    "SIMPLE_MOVEMENT": SIMPLE_MOVEMENT,
    "COMPLEX_MOVEMENT": COMPLEX_MOVEMENT,
}
