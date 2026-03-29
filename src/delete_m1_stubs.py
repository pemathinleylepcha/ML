"""Delete M1 candle stub files (< 500 bytes) from 2024 and 2025 quarters."""
from pathlib import Path

root = Path(r"D:\dataset-ml\DataExtractor")
deleted = []

for f in sorted(root.glob("202[45]/Q*/*/candles_M1.csv")):
    size = f.stat().st_size
    if size < 500:
        f.unlink()
        deleted.append((str(f), size))

print(f"Deleted {len(deleted)} stub files:")
for path, size in deleted:
    print(f"  {size:4d}b  {path}")

# Verify
remaining = [f for f in root.glob("202[45]/Q*/*/candles_M1.csv") if f.stat().st_size < 500]
print(f"\nRemaining stubs: {len(remaining)}")
