from __future__ import annotations

import site
import sys
import time


def main() -> int:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} basic=start", flush=True)
    print(sys.executable, flush=True)
    print(sys.prefix, flush=True)
    print(site.getsitepackages(), flush=True)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} basic=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
