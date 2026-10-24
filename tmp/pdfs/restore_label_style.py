import fitz
src='tmp/pdfs/fig_sensitivity_original.pdf'
out='tmp/pdfs/fig_sensitivity_restored.pdf'
doc=fitz.open(src)
p=doc[0]
font='tibo'
items=[('(0.70, 81%)','(0.70, 88%)'),('(0.20, 81%)','(0.20, 88%)')]
rects=[]
for old,new in items:
    hits=p.search_for(old)
    if len(hits)!=1: raise RuntimeError((old,hits))
    rects.append((hits[0],new))
    p.add_redact_annot(hits[0],fill=(1,1,1))
p.apply_redactions()
for r,new in rects:
    rv=p.insert_text((r.x0,32.102539),new,fontname=font,fontsize=24,color=(176/255,107/255,0))
    if rv <= 0: raise RuntimeError(f'text insertion failed {new}: {rv}')
doc.save(out,garbage=4,deflate=True)
