from base import Action


def get_action(action):
    if action == 0:  # Up
        return Action.UP
    elif action == 1:  # Down
        return Action.DOWN
    elif action == 2:  # Left
        return Action.LEFT
    elif action == 3:  # Right
        return Action.RIGHT
    elif action == 4:  # Up Right
        return Action.UP_RIGHT
    elif action == 5:  # Up Left
        return Action.UP_LEFT
    elif action == 6:  # Down Right
        return Action.DOWN_RIGHT
    elif action == 7:  # Down Left
        return Action.DOWN_LEFT
    elif action == 8:  # NOOP
        return Action.NOOP