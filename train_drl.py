"""Train DRL sizer: SB3 PPO if available else rule-based fallback policy.
Keeps medium scope runnable without GPU/torch.
Run: python train_drl.py
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

from config import DATA_OUT, TRAINED_DIR


def load_prices() -> pd.DataFrame:
    p = DATA_OUT / "train_data.csv"
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    # use first ticker's close as env proxy (keeps medium scope)
    t0 = df["ticker"].iloc[0]
    px = df[df["ticker"] == t0][["Close"]].copy()
    px["ret"] = px["Close"].pct_change().fillna(0)
    return px


def train_sb3(rets: np.ndarray, out_dir) -> dict:
    import gymnasium as gym
    from stable_baselines3 import PPO

    class TrendEnv(gym.Env):
        def __init__(self, rets):
            super().__init__()
            self.rets = rets
            import numpy as _np
            self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(5,), dtype=_np.float32)
            self.action_space = gym.spaces.Discrete(3)  # 0=flat 1=long 2=short
            self.i = 5
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.i, self.pos = 5, 0
            return self._obs(), {}
        def _obs(self):
            import numpy as _np
            return _np.array(self.rets[self.i-5:self.i], dtype=_np.float32)
        def step(self, a):
            self.pos = {0: 0, 1: 1, 2: -1}[int(a)]
            r = float(self.pos * self.rets[self.i])
            self.i += 1
            done = self.i >= len(self.rets) - 1
            import numpy as _np
            return _np.array(self.rets[self.i-5:self.i], dtype=_np.float32), r, done, False, {}
    env = TrendEnv(rets.astype(float))
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)
    model.save(str(out_dir / "ppo_policy"))
    return {"algo": "PPO", "timesteps": 5000}


def train_fallback(rets: np.ndarray, out_dir) -> dict:
    # momentum rule as policy: pos = sign(mean(last5))
    policy = {"type": "momentum-5", "note": "fallback without torch/SB3"}
    (out_dir / "ppo_policy.json").write_text(json.dumps(policy, indent=2))
    return {"algo": "momentum-fallback"}


def main() -> dict:
    TRAINED_DIR.mkdir(exist_ok=True)
    px = load_prices()
    rets = px["ret"].to_numpy()
    try:
        metrics = train_sb3(rets, TRAINED_DIR)
    except Exception as e:
        print(f"[train_drl] SB3 unavailable ({e}), using fallback")
        metrics = train_fallback(rets, TRAINED_DIR)
    (TRAINED_DIR / "drl_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train_drl] {metrics}")
    return metrics


if __name__ == "__main__":
    main()
