"""
CLI: Restore Titans memory from snapshot.

Usage:
    python -m memory.restore                    # restore latest snapshot
    python -m memory.restore --snapshot latest   # same
    python -m memory.restore --snapshot <path>   # specific file
    python -m memory.restore --list              # list available snapshots
"""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description="Restore Titans memory from snapshot")
    parser.add_argument(
        "--snapshot", default="latest",
        help="Snapshot path or 'latest' (default: latest)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available snapshots and exit"
    )
    parser.add_argument(
        "--snapshot-dir", default=None,
        help="Directory containing snapshots (default: models/titans/)"
    )
    args = parser.parse_args()

    from memory.titans import TitansMemory
    titans = TitansMemory(snapshot_dir=args.snapshot_dir)

    if args.list:
        snapshots = titans._list_snapshots()
        if not snapshots:
            print("No snapshots found.")
            return
        print(f"Found {len(snapshots)} snapshot(s):")
        for i, s in enumerate(snapshots):
            import os
            size = os.path.getsize(s) / 1024
            print(f"  [{i+1}] {os.path.basename(s)} ({size:.1f} KB)")
        return

    path = None if args.snapshot == "latest" else args.snapshot
    success = titans.restore_snapshot(path)
    if success:
        print("✅ Titans memory restored successfully.")
        stats = titans.get_stats()
        print(f"   Updates: {stats['total_updates']}, Surprises: {stats['total_surprises']}")
    else:
        print("❌ Restore failed. Use --list to see available snapshots.")
        sys.exit(1)


if __name__ == "__main__":
    main()
