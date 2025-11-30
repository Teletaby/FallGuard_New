#!/usr/bin/env python3
# Wrapper to run main.py with system Python

import subprocess
import sys

# Run main.py
result = subprocess.run([sys.executable, "main.py"], cwd=__file__.rsplit('\\', 1)[0] if '\\' in __file__ else '.')
sys.exit(result.returncode)
