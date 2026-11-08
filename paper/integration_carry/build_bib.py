"""Assemble refs.bib from the verified prior and new entry files, keeping only cited keys."""
import re
from pathlib import Path

HERE = Path(__file__).parent


def entries(text):
    return {m.group(1).strip(): m.group(0) for m in re.finditer(r"@\w+\{([^,]+),.*?\n\}", text, re.S)}


tex = (HERE / "main.tex").read_text()
cited = {k.strip() for group in re.findall(r"\\cite\{([^}]*)\}", tex) for k in group.split(",")}
prior = entries((HERE / "refs_prior.bib").read_text())
new = {**entries((HERE / "refs_new.bib").read_text()), **entries((HERE / "refs_new2.bib").read_text()),
       **entries((HERE / "refs_new3.bib").read_text())}
out, missing = [], []
for k in sorted(cited):
    src = new.get(k) or prior.get(k)
    (out.append(src) if src else missing.append(k))
(HERE / "refs.bib").write_text("\n\n".join(out) + "\n")
print(len(out), "entries; missing:", missing)
