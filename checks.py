"""Complete normal suite, count-event audit and a small resource exercise."""
import subprocess
import sys
import unittest
from pathlib import Path
from bench import verify_frozen, run_suite


def main():
    verify_frozen()
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    audit = subprocess.run([sys.executable, "-B", str(root / "count_audit.py")])
    if audit.returncode:
        return audit.returncode
    run_suite("compare", sizes=(32, 128), rounds=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
