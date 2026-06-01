# SpriteCraft Report Template

This directory contains a compact academic report/thesis template for the SpriteCraft project.
It is adapted from the structure of the 0BSD-licensed
[`latextemplates/scientific-thesis-template`](https://github.com/latextemplates/scientific-thesis-template),
but keeps the local dependency set small enough to build with the TeX installation currently available in this workspace.

The upstream template defaults to `lualatex` with `biblatex`/`biber`.
This project-local version uses KOMA-Script, `pdflatex`, `natbib`, and BibTeX because those packages are installed here.

## Build

From the repository root:

```sh
make -C report
```

Or from this directory:

```sh
latexmk -pdf main.tex
```

The generated PDF and auxiliary files are written to `report/build/`, which is ignored by Git.

## Edit

- `metadata.tex`: title, author, department, date, and PDF metadata.
- `frontmatter/`: abstract and optional front matter.
- `chapters/`: report body.
- `references.bib`: BibTeX bibliography.
- `figures/`: source figures that should be versioned.

Generated outputs should stay out of version control.
Keep source figures in `figures/`; avoid writing generated experiment artifacts directly into `report/`.
