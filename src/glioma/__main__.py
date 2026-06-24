"""Allow running ``python -m src.glioma``."""

from src.glioma.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
