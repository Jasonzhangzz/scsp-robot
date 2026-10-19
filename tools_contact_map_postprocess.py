"""Create compact contact-map SVGs and standalone temperature bars."""
from pathlib import Path
import re

# Tight common viewport around the 3-D contact-map scene (excludes colorbar).
VIEWBOX = (95.0, 55.0, 310.0, 370.0)

def compact_svg(src: Path, dst: Path):
    s = src.read_text()
    s = re.sub(r'<text\b[^>]*>\s*[xyz]\s*</text>\s*', '', s)
    vb = ' '.join(f'{v:g}' for v in VIEWBOX)
    s = re.sub(r'viewBox="[^"]+"', f'viewBox="{vb}"', s, count=1)
    s = re.sub(r'width="[^"]+pt"', 'width="310pt"', s, count=1)
    s = re.sub(r'height="[^"]+pt"', 'height="370pt"', s, count=1)
    dst.write_text(s)

def temperature_bar(dst: Path):
    dst.write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="90" height="420" viewBox="0 0 90 420">\n<defs><linearGradient id="temperature" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#3b4cc0"/><stop offset="0.25" stop-color="#8db0fe"/><stop offset="0.5" stop-color="#dddcdc"/><stop offset="0.75" stop-color="#f4987a"/><stop offset="1" stop-color="#b40426"/></linearGradient></defs>\n<rect x="20" y="20" width="28" height="360" fill="url(#temperature)" stroke="#333" stroke-width="1"/>\n<g font-family="Arial, sans-serif" font-size="13" fill="#222"><text x="55" y="385">0.2</text><text x="55" y="295">0.4</text><text x="55" y="205">0.6</text><text x="55" y="115">0.8</text><text x="55" y="25">1.0</text><text x="14" y="415" font-size="14">temperature</text></g></svg>\n''')

def process(folder: Path):
    for src in sorted(folder.glob('*_contact_map.svg')):
        # Skip already processed names.
        if src.stem.endswith('_clean'): continue
        compact_svg(src, src.with_name(src.stem + '_clean.svg'))
    temperature_bar(folder / 'temperature_bar.svg')

if __name__ == '__main__':
    for name in ('elephant', 'stanford_bunny2'):
        process(Path('/home/zz/scsp-robot/output/contact_maps_20261019/final') / name)
