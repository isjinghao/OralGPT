# OralGPT-Family Technical Report Template

LaTeX template for OralGPT-Family technical reports on arXiv. All body text is
placeholder content; every number is a placeholder.

## Files

| File | What to edit |
|---|---|
| `main.tex` | Entry point. Leave as is; delete `\input{appendix}` if you need no appendix |
| `frontmatter.tex` | Title, authors, affiliations, abstract, date, Github / Hugging Face links |
| `content.tex` | Main body (sections, figures, tables, equations) and bibliography call |
| `appendix.tex` | Appendix (sections A, B, ...; Table A.1, Figure B.1, (A.1)) |
| `refs.bib` | BibTeX references (OralGPT-Family papers included) |
| `template_style.sty` | All styling; common settings are in the CONFIGURATION AREA at the top |
| `assets/` | `oralgpt_logo.png` and placeholder figures |

Common settings in `template_style.sty`: `ogPrimary` (heading/caption/link blue),
`ogBoxBg` (title-box background), `\ogHeaderLogoHeight`, `\ogHeaderText`,
`\ogHeaderRuleRaise`, and `headsep` inside `\geometry{...}`.

## Compile

Use **pdfLaTeX** (not XeLaTeX/LuaLaTeX):

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

On Overleaf: Menu → Compiler: pdfLaTeX, Main document: main.tex.

Requires standard TeX Live packages only, including `tcolorbox`, `fontawesome5`,
`twemojis`, `mathpazo`, `helvet`, `natbib`, `float`. No shell-escape.

## License notes

- OralGPT logo © OralGPT project, used with permission.
- GitHub icon: Font Awesome Free (CC BY 4.0); calendar and 🤗 icons: Twemoji (CC BY 4.0).
