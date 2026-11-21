#!/usr/bin/env python3
"""Render the Marp-style clean deck to standalone HTML, PDF, and review images.

Optional CPU tools: markdown-it-py, pyyaml, playwright, pymupdf.
This renderer supports the Markdown/HTML subset used by this deck. It does not
require Marp or change the research environment. Alternatively use Marp --html.
Local isolated setup used for this export lives in scratch/presentation-tools/.
"""
import base64
import html
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "scratch/presentation-tools"
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(LOCAL / "browsers"))
os.environ.setdefault("TMPDIR", str(LOCAL))
libs = LOCAL / "sysroot/usr/lib/x86_64-linux-gnu"
if libs.exists():
    os.environ["LD_LIBRARY_PATH"] = str(libs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

from markdown_it import MarkdownIt
import yaml
from playwright.sync_api import sync_playwright
import pymupdf


def data_uri(path):
    mime = {".svg": "image/svg+xml", ".png": "image/png", ".ttf": "font/ttf"}[path.suffix]
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    source = ROOT / "docs/presentation_clean.md"
    _, front, body = source.read_text().split("---", 2)
    config = yaml.safe_load(front)
    parts = re.split(r"\n---\s*\n", body.strip())
    md = MarkdownIt("default", {"html": True}).enable("table")
    sections = []
    for i, part in enumerate(parts, 1):
        classes = re.search(r"<!--\s*_class:\s*(.*?)\s*-->", part)
        cls = classes.group(1) if classes else ""
        content = md.render(part)
        def embed(match):
            path = (source.parent / match.group(1)).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return 'src="' + data_uri(path) + '"'
        content = re.sub(r'src="(figures/[^"]+)"', embed, content)
        sections.append(f'<section class="{cls}" id="slide-{i}" data-page="{i}">{content}</section>')
    fonts = ROOT / ".venv/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf"
    # Minimal servers may have neither fontconfig nor a system font directory.
    # Keep its configuration/cache in the same isolated workspace directory.
    LOCAL.mkdir(parents=True, exist_ok=True)
    font_config = LOCAL / "fonts.conf"
    font_config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd">'
                          + f'<fontconfig><dir>{fonts}</dir><cachedir>{LOCAL / "font-cache"}</cachedir></fontconfig>')
    os.environ["FONTCONFIG_FILE"] = str(font_config)
    font_css = ""
    for name, weight in [("DejaVuSans.ttf",400),("DejaVuSans-Bold.ttf",700)]:
        path = fonts / name
        if path.exists():
            font_css += f"@font-face {{font-family:'DejaVu Sans';font-weight:{weight};src:url('{data_uri(path)}');}}\n"
    base = """
      * { box-sizing: border-box; }
      html { scroll-behavior: smooth; }
      body { margin: 0; background: #DCE3ED; }
      section { width: 1280px; height: 720px; display: flex; flex-direction: column;
                position: relative; overflow: hidden; margin: 24px auto; flex-shrink: 0; }
      section > * { flex-shrink: 0; }
      section::after { content: attr(data-page); position: absolute; }
      .nav { position: fixed; z-index: 10; bottom: 14px; right: 16px; background: #152A43;
             color: white; border-radius: 10px; padding: 7px 12px; font: 14px Arial; }
      .nav button { color: white; background: transparent; border: 0; cursor: pointer; font-size: 21px; padding: 3px 12px; }
      body.export .nav { display: none; }
      @page { size: 1280px 720px; margin: 0; }
      @media print {
        body { background: white; }
        section { margin: 0; break-after: page; break-inside: avoid; }
        section:last-of-type { break-after: auto; }
        .nav { display: none; }
        * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      }
    """
    navigation = """
    <div class="nav"><button aria-label="Previous slide" onclick="move(-1)">←</button><span id="counter"></span><button aria-label="Next slide" onclick="move(1)">→</button></div>
    <script>
    const slides=[...document.querySelectorAll('section')];
    const current=()=>slides.reduce((best,s,i)=>Math.abs(s.getBoundingClientRect().top)<Math.abs(slides[best].getBoundingClientRect().top)?i:best,0);
    function move(delta){const i=Math.max(0,Math.min(slides.length-1,current()+delta));slides[i].scrollIntoView({behavior:'smooth',block:'start'});}
    function update(){document.getElementById('counter').textContent=`${current()+1} / ${slides.length}`;}
    addEventListener('scroll',update,{passive:true});
    addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();move(1);}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();move(-1);}});
    if(location.search.includes('export'))document.body.classList.add('export');update();
    </script>
    """
    out = source.with_suffix(".html")
    out.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=1280">'
                   + '<title>'+html.escape(config["title"])+'</title><style>'+font_css+base+config["style"]
                   + '</style></head><body>'+"\n".join(sections)+navigation+'</body></html>')
    review = ROOT / "scratch/presentation-review"
    review.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox","--disable-dev-shm-usage","--disable-gpu"])
        page = browser.new_page(viewport={"width":1400,"height":850},device_scale_factor=1)
        page.goto(out.as_uri()+"?export",wait_until="load")
        page.evaluate("document.fonts.ready")
        font_errors = page.evaluate("[...document.fonts].filter(f=>f.status==='error').map(f=>f.family)")
        assert not font_errors, f"Fonts failed to render: {font_errors}"
        assert page.locator("h1").first.bounding_box()["height"] > 20, "Text has no visible height"
        page.wait_for_function("[...document.images].every(i => i.complete && i.naturalWidth > 0)")
        audit = page.evaluate("""() => [...document.querySelectorAll('section')].map(s=>{
          const bounds=s.getBoundingClientRect();
          const children=[...s.children].filter(e=>e.getBoundingClientRect().height>0);
          const bottom=Math.max(...children.map(e=>e.getBoundingClientRect().bottom-bounds.top));
          const bad=[...s.querySelectorAll('*')].filter(e=>{
            const b=e.getBoundingClientRect();return b.width>0&&b.height>0&&(b.left<bounds.left-1||b.right>bounds.right+1||b.bottom>bounds.bottom-36);
          }).map(e=>({tag:e.tagName,text:e.innerText?.slice(0,80)}));
          return {slide:s.dataset.page,title:s.querySelector('h1')?.innerText,bottom,overflow:bad};
        })""")
        (review/"layout-audit.json").write_text(json.dumps(audit,indent=2))
        for i, sec in enumerate(page.locator("section").all(),1):
            sec.screenshot(path=str(review/f"slide-{i:02d}.png"))
        page.pdf(path=str(source.with_suffix(".pdf")),print_background=True,prefer_css_page_size=True)
        browser.close()
    pdf=pymupdf.open(source.with_suffix(".pdf"))
    assert len(pdf)==len(parts),f"PDF has {len(pdf)} pages; expected {len(parts)}"
    assert all(len(p.get_text().strip())>60 for p in pdf),"Blank or raster-only page"
    print(f"Rendered {len(parts)} slides: {out} and {source.with_suffix('.pdf')}")
    issues=[a for a in audit if a["overflow"]]
    print(json.dumps({"layout_issues":issues,"review_images":str(review)},indent=2))
    if issues: raise SystemExit(1)


if __name__=="__main__": main()
