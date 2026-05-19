from pathlib import Path

import casbin


def create_enforcer() -> casbin.Enforcer:
    base = Path(__file__).parent
    return casbin.Enforcer(str(base / "model.conf"), str(base / "policy.csv"))
