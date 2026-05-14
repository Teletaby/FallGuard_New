#!/usr/bin/env python3
import subprocess
import sys
import re

# Run main.py and capture output
proc = subprocess.Popen([sys.executable, 'main.py'], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.STDOUT, 
                       text=True, 
                       bufsize=1)

debug_lines = []
line_count = 0
max_lines = 200

try:
    for line in proc.stdout:
        line_count += 1
        if line_count > max_lines:
            break
            
        # Print all lines
        print(line, end='', flush=True)
        
        # Also collect DEBUG/DETECTION lines
        if re.search(r'(DEBUG|DETECTION|Frame|Box|SUCCESS|ERROR|No people)', line):
            debug_lines.append(line.rstrip())
            
except KeyboardInterrupt:
    proc.terminate()
except Exception as e:
    print(f"Error: {e}")

proc.terminate()
proc.wait(timeout=2)

print("\n\n=== FILTERED DEBUG LINES ===")
for line in debug_lines[-40:]:
    print(line)
