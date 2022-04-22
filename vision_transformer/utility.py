from typing import Any


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, default_val: Any) -> Any:
    return val if exists(val) else default_val
