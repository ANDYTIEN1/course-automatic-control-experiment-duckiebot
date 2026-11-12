---
title: "course-automatic-control-experiment-duckiebot"
author: "Torbjörn E. M. Nordling"
date: "2026-11-12"
version: "0.1"
license: "Apache-2.0"
purpose: "Student reports with their data, code and figures."
type: "student"
---

# course-automatic-control-experiment-duckiebot

Automatic Control Student Material for the Duckiebot Experiments

Write everything in this repository in English: reports, code, comments, file names, and commit messages.

## Folders

| Folder | Contents |
|---|---|
| `doc/` | One report folder per student, and the example report `template_Nordling2026_LatexExample/`. |
| `data/` | The data the reports use, committed to the repository. |
| `scripts/` | The code that processes the data and draws the figures. |
| `build/` | Every build artefact: PDFs, LaTeX auxiliary files, and other generated output. Git ignores everything in it. |

## Your report

Place your report in its own folder in `doc/`, named

```text
doc/report_FamilynameYEAR_TwoTitleWords_<StudentID>/
```

- `Familyname` is your family name, in ASCII letters.
- `YEAR` is the four-digit year of the report.
- `TwoTitleWords` is two words from the report title, each starting with a capital letter.
- `<StudentID>` is your student ID.

For example, `doc/report_Smith2026_GaitAnalysis_A12345678/`.

Put every figure of the report in a `figures/` subfolder of the report folder:

```text
doc/report_Smith2026_GaitAnalysis_A12345678/
├── main.tex
├── references.bib
└── figures/
    └── fig-gait-cycle.pdf
```

Commit the data your report uses in `data/` and the code in `scripts/`, so that anyone can regenerate every result and figure from the repository.

## Writing the report

We recommend LaTeX with the standard `article` class and APA references through `biblatex-apa`.
The report must compile with `pdflatex`, or with LuaLaTeX when it contains Chinese text, as Appendix B of the example report explains.

Start from the example report in `doc/template_Nordling2026_LatexExample/`: copy the folder to your report folder and replace its text.
The example has the structure of a technical report, and its text explains how to write each section and how to format figures, tables, equations, units, citations, and code.
Its minimal skeleton is:

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{csquotes}
\usepackage[backend=biber,style=apa,annotation=false]{biblatex}
\addbibresource{references.bib}

\title{Gait Analysis of ...}
\author{Firstname Familyname}

\begin{document}
\maketitle
...
\begin{figure}
  \centering
  \includegraphics[width=\linewidth]{figures/fig-gait-cycle.pdf}
  \caption{...}
\end{figure}
...
\printbibliography
\end{document}
```

Compile it into `build/` from the repository root, which runs `pdflatex` and `biber` as often as needed:

```bash
latexmk -pdf -cd \
  -outdir=../../build/report_Smith2026_GaitAnalysis_A12345678 \
  doc/report_Smith2026_GaitAnalysis_A12345678/main.tex
```
