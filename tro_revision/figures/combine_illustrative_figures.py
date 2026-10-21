"""Crop the two existing vector PDFs below their axis labels and join horizontally."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

FIGURES = Path(__file__).resolve().parent
INPUTS = ('fig_verify_acceptance_illustrative.pdf', 'fig_ema_illustrative.pdf')
OUTPUT = FIGURES/'fig_acceptance_ema_illustrative_combined.pdf'
PADDING_BP = 2.5
GAP_BP = 8.0


def bounds(path):
    xml = subprocess.check_output(['pdftotext', '-bbox', str(path), '-'], text=True)
    root = ET.fromstring(xml)
    page = next(n for n in root.iter() if n.tag.rsplit('}', 1)[-1] == 'page')
    words = [n for n in page.iter() if n.tag.rsplit('}', 1)[-1] == 'word'
             and n.text not in ('Illustrative', 'curve')]
    assert words
    return dict(width=float(page.attrib['width']), height=float(page.attrib['height']),
                top=min(float(n.attrib['yMin']) for n in words),
                bottom=max(float(n.attrib['yMax']) for n in words))


def main():
    pages = [bounds(FIGURES/name) for name in INPUTS]
    top = max(0., min(p['top'] for p in pages)-PADDING_BP)
    bottom = max(p['bottom'] for p in pages)+PADDING_BP
    height = bottom-top
    width = sum(p['width'] for p in pages)+GAP_BP
    graphics = []
    for name, page in zip(INPUTS, pages):
        graphics.append(r'\includegraphics[trim=0bp %.8fbp 0bp %.8fbp,clip]{%s}' %
                        (page['height']-bottom, top, (FIGURES/name).as_posix()))
    source = '\n'.join([
        r'\documentclass{article}',
        r'\usepackage{graphicx}',
        r'\usepackage[paperwidth=%.8fbp,paperheight=%.8fbp,margin=0bp]{geometry}' % (width, height),
        r'\pagestyle{empty}\setlength{\parindent}{0pt}\setlength{\topskip}{0pt}',
        r'\pdfinfo{/Title (Acceptance threshold and EMA - illustrative curves) '
        r'/Subject (User-specified illustrative curves; not experimental measurements)}',
        r'\begin{document}',
        r'\noindent\hbox{'+(r'\hspace{%.8fbp}' % GAP_BP).join(graphics)+r'}',
        r'\end{document}', ''])
    with tempfile.TemporaryDirectory(prefix='scsp_figure_crop_') as directory:
        temporary = Path(directory)
        (temporary/'combined.tex').write_text(source)
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                                 'combined.tex'], cwd=temporary, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(result.stdout[-6000:])
        shutil.copyfile(temporary/'combined.pdf', OUTPUT)
    metadata = dict(inputs=list(INPUTS), output=OUTPUT.name, order='acceptance left, EMA right',
                    top_from_page_top_bp=top, bottom_from_page_top_bp=bottom,
                    label_padding_bp=PADDING_BP, horizontal_gap_bp=GAP_BP,
                    output_width_bp=width, output_height_bp=height,
                    vector_content_preserved=True, data_type='illustrative',
                    experimental_measurements=False)
    OUTPUT.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(OUTPUT)


if __name__ == '__main__':
    main()
