# convert all svg files in current directory to pdf files

import os
from pathlib import Path

# install cairosvg if not installed
try:
    import cairosvg
except ImportError:
    os.system('pip install cairosvg')
    import cairosvg
    
# convert all svg files to pdf files
currentDir = Path(__file__).parent

for file in currentDir.glob("*.svg"):
    cairosvg.svg2pdf(url=str(file), write_to=str(file.with_suffix(".pdf")))
    print(f"Converted {file.name} to {file.with_suffix('.pdf').name}")
