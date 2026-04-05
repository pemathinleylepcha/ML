from __future__ import annotations

import time


def stamp(name: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} import={name}", flush=True)


def main() -> int:
    stamp("start")
    import torch

    stamp("torch")
    import numpy  # noqa: F401

    stamp("numpy")
    import pandas  # noqa: F401

    stamp("pandas")
    print(torch.__version__, flush=True)
    print(torch.cuda.is_available(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
