from base import Action
import json
import numpy as np


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


def get_map(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)
        map_ = config['map']
        map_path = '../server/maps/'
        map_path += map_
    return map_path


def fetch_grid(height, width):
    map_path = get_map("../server/config.json")
    map_ = open(map_path, 'r').read().replace('\n', '')
    return np.array(list(map_)).reshape(height, width)