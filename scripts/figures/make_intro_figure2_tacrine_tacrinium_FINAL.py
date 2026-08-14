from pathlib import Path
import re
import cairosvg
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

outdir = Path(".")
svg_path = outdir / "Figure2_Tacrine_Tacrinium_THESIS_FINAL.svg"
png_path = outdir / "Figure2_Tacrine_Tacrinium_THESIS_FINAL_600dpi.png"
pdf_path = outdir / "Figure2_Tacrine_Tacrinium_THESIS_FINAL.pdf"

neutral_smiles = "Nc1c2CCCCc2nc3ccccc13"
cation_smiles  = "Nc1c2CCCCc2[nH+]c3ccccc13"

def make_mol_svg(smiles: str, width=560, height=330) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.035
    opts.bondLineWidth = 3.2
    opts.minFontSize = 24
    opts.maxFontSize = 34
    opts.annotationFontScale = 1.0
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def svg_inner(svg: str) -> str:
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>", "", svg)
    svg = re.sub(r"^\s*<svg[^>]*>", "", svg, count=1)
    svg = re.sub(r"</svg>\s*$", "", svg, count=1)
    return svg.strip()

neutral_svg = svg_inner(make_mol_svg(neutral_smiles))
cation_svg = svg_inner(make_mol_svg(cation_smiles))

W, H = 1500, 500
parent_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="100%" height="100%" fill="white"/>
<style>
.title {{ font-family:"DejaVu Sans",Arial,sans-serif; font-size:31px; font-weight:700; fill:#202020; }}
.panel {{ font-family:"DejaVu Sans",Arial,sans-serif; font-size:24px; font-weight:700; fill:#4a4a4a; }}
.formula {{ font-family:"DejaVu Sans",Arial,sans-serif; font-size:21px; fill:#4a4a4a; }}
.arrowlabel {{ font-family:"DejaVu Sans",Arial,sans-serif; font-size:22px; font-weight:600; fill:#303030; }}
.subarrow {{ font-family:"DejaVu Sans",Arial,sans-serif; font-size:20px; fill:#4a4a4a; }}
</style>
<text x="55" y="48" class="panel">(A)</text>
<text x="300" y="52" text-anchor="middle" class="title">Tacrine</text>
<text x="300" y="82" text-anchor="middle" class="formula">C₁₃H₁₄N₂</text>
<text x="905" y="48" class="panel">(B)</text>
<text x="1200" y="52" text-anchor="middle" class="title">Tacrinium (+1)</text>
<text x="1200" y="82" text-anchor="middle" class="formula">C₁₃H₁₅N₂⁺</text>
<g transform="translate(35,105)">{neutral_svg}</g>
<g transform="translate(905,105)">{cation_svg}</g>
<defs>
<marker id="arrowhead" markerWidth="12" markerHeight="12" refX="10" refY="6"
        orient="auto" markerUnits="strokeWidth">
<path d="M0,0 L12,6 L0,12 z" fill="#3f3f3f"/>
</marker>
</defs>
<text x="750" y="205" text-anchor="middle" class="arrowlabel">Ring-N protonation</text>
<text x="750" y="238" text-anchor="middle" class="subarrow">+ H⁺</text>
<line x1="625" y1="285" x2="875" y2="285" stroke="#3f3f3f" stroke-width="5"
      marker-end="url(#arrowhead)"/>
</svg>"""

svg_path.write_text(parent_svg, encoding="utf-8")
cairosvg.svg2png(bytestring=parent_svg.encode("utf-8"), write_to=str(png_path),
                 output_width=3000, output_height=1000)
cairosvg.svg2pdf(bytestring=parent_svg.encode("utf-8"), write_to=str(pdf_path))

print("Created:")
print(svg_path)
print(png_path)
print(pdf_path)
