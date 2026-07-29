import importlib
from pathlib import Path

from backtester.strategy import Strategy


def list_strategies() -> list[dict]:
    """Discover all strategies by globbing *_strategy.py files.

    Returns a list of dicts: {"name": str, "class": type, "file": str}
    """
    strategies_dir = Path(__file__).parent
    results = []

    for path in sorted(strategies_dir.glob("*_strategy.py")):
        # Derive strategy name: rsi_ema_strategy.py -> rsi_ema
        name = path.stem.removesuffix("_strategy")
        module_name = f"strategies.{path.stem}"

        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Strategy)
                and attr is not Strategy
            ):
                results.append({
                    "name": name,
                    "class": attr,
                    "file": str(path),
                })
                break

    return results
