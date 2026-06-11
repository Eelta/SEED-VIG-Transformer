import json
import os
from types import SimpleNamespace

_this_dir = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_this_dir, '..', 'Configs', 'config.json')
_BEST_PATH = os.path.join(_this_dir, '..', 'Configs', 'best_params.json')


def load_config():
    # 读取默认设置
    with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = SimpleNamespace(**json.load(f))
    # 最优设置覆盖
    if os.path.exists(_BEST_PATH):
        with open(_BEST_PATH, 'r', encoding='utf-8') as f:
            best = json.load(f)
        for key, value in best.items():
            setattr(config, key, value)

    return config
