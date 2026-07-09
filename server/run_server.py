#!/usr/bin/env python3

import subprocess
import sys


result = subprocess.run([sys.executable, "-m", "app.main"], cwd=__file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
sys.exit(result.returncode)