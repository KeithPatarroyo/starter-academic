---
title: "Rapid Exploration of the Assembly Chemical Space of Molecular Graphs"
authors:
- Ian Seet
- admin
- Gage Siebert
- Sara I. Walker
- Leroy Cronin
date: "2025-11-21T00:00:00Z"
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: "2017-10-30:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["2"]

# Publication name and optional abbreviated publication name.
publication: "[Journal of Chemical Information and Modeling](https://pubs.acs.org/journal/jcisd8?ref=breadcrumb)"
publication_short: "JCIM"

abstract: "Quantifying how hard it is to build a molecular graph matters for biosignature detection, chemical complexity, and cheminformatics. We present an exact, scalable algorithm to compute the molecular assembly index (MA), which prioritizes the largest duplicate subgraphs, represents fragmentation with an array of edge-lists, and prunes the search with both dynamic programming via a hash table of assembly states and a branch-and-bound heuristic guided by a conditional addition-chain lower bound. For organic molecules in the greater-than-500 Da range, our approach is up to 6 orders of magnitude faster than prior methods and yields exact MAs where previous algorithms would have timed out. We compute MAs to convergence for ∼300k COCONUT natural products with <50 bonds, profiling time and memory scaling. Finally, we exploit the speed of our algorithm to calculate joint assembly spaces and introduce the Joint Assembly Overlap (JAO), a Jaccard-like metric that emphasizes global scaffold reuse, and show that the JAO yields substantially different rankings from Tanimoto similarity with ECFP fingerprints and MCS (e.g., in steroids 270–380 Da and short peptides), accounting for substructural similarity beyond local environments. Together, these advances turn the molecular assembly index into a practical tool for large-scale exploration of chemical space."

# Summary. An optional shortened abstract.
summary: "Journal Paper containing a novel algorithm to calculate assembly indices of large molecules, allowing the exploration of the complexity of molecules in chemical space."

tags:
- dynamic-programming
- assembly-theory
- chemical-space
featured: false

links:
- name: 'Online Version'
  url: https://pubs.acs.org/doi/full/10.1021/acs.jcim.5c01964
url_pdf: 'https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.5c01964?ref=article_openPDF'
url_code: 'https://github.com/croningp/assemblycpp-v5' 
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: 'Conference Paper Figure 3: Assembly Theory measures and approximations in CA'
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
- dynamic-programming
- assembly-theory
- chemical-space

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides:
---
