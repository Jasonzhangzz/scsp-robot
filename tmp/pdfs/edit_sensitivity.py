import fitz
src='tro_revision/figures/fig_sensitivity.pdf'
out='tmp/pdfs/fig_sensitivity_edited.pdf'
doc=fitz.open(src)
p=doc[0]
repls=[('(0.70, 81%)','(0.70, 88%)'),('(0.20, 81%)','(0.20, 88%)')]
for old,new in repls:
    hits=p.search_for(old)
    if len(hits)!=1: raise RuntimeError((old,hits))
    r=hits[0]
    p.add_redact_annot(r, fill=(1,1,1))
p.apply_redactions()
for old,new in repls:
    # locate original y ranges from known blocks, preserving x and baseline
    r = fitz.Rect(267.1488,10.6945,385.1328,37.3105) if old.startswith('(0.70') else fitz.Rect(583.4660,10.6945,701.4500,37.3105)
    p.insert_textbox(r, new, fontname='tibo', fontsize=10, color=(0,0,0), align=0)
doc.save(out, garbage=4, deflate=True)
print(out)
