#!/usr/bin/env python3

import subprocess
import sys


try:
	result = subprocess.run([sys.executable, "-m", "app.main"], cwd=__file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
except KeyboardInterrupt:
	sys.exit(130)

sys.exit(result.returncode)