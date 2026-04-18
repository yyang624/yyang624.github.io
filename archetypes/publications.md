---
title: "{{ replace .Name "-" " " | title }}"
date: {{ .Date }}
authors: ["J. Doe", ""]
preprint: false       # set true for arXiv-only; omits journal/DOI fields
status: ""            # optional: e.g. "submitted", "under review"
journal: ""
volume: ""
pages: ""
year: 
doi: ""
arxiv: ""
pdf: ""
code: ""
math: true
---

## Abstract

Paste abstract here. Equations render inline: $\hat{H}|\psi\rangle = E|\psi\rangle$.
