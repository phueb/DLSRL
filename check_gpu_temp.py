import subprocess
import re


lines = subprocess.getoutput('nvidia-smi -q').split()
for n, line in enumerate(lines):
    match=re.search('Temp', line)
    if match:
        print(lines[n:n+5])