// Hand-written IMS L1-sweep figure; iterate on layout here. Per-seed curves
// (nz, auroc, content_frac as %) and the column-wise mean come from
// figs/aggregated.json, emitted by scripts/aggregate.py.

#import "@preview/lilaq:0.6.0" as lq
#import "@preview/subpar:0.2.2"

#let _ims     = json("aggregated.json").ims_sweep
#let _seeds   = _ims.seeds
#let _mean    = _ims.mean
#let _n_seeds = _seeds.len()

// Palette: green for AUROC, red for content fraction. Faint per-seed lines
// (alpha 60) underneath a bold mean (alpha 255) with markers.
#let _auroc_faint = rgb(44,  160, 44,  60)
#let _auroc_bold  = rgb(44,  160, 44)
#let _frac_faint  = rgb(214, 39,  40,  60)
#let _frac_bold   = rgb(214, 39,  40)

// Plain-integer x-tick labels: force no auto-exponent (no `×10^3`).
// A hidden second line in the (a) caption matches (b)'s two-line height so
// the top row of subfigures stays vertically aligned.

#subpar.grid(
  figure(
    lq.diagram(
      width: 100%, height: 3.2cm,
      xlabel: [$n_"dicts,sel"$ active],
      ylabel: [AUROC],
      xaxis: (exponent: 0),
      ..for s in _seeds { (
        lq.plot(s.nz, s.auroc, mark: none,
                stroke: (paint: _auroc_faint, thickness: 0.6pt)),
      ) },
      lq.plot(_mean.nz, _mean.auroc, mark: "o",
              stroke: (paint: _auroc_bold, thickness: 1.5pt)),
    ),
    caption: [AUROC vs sparsity. \ #hide[.]],
  ), <fig:ims-auroc>,
  figure(
    lq.diagram(
      width: 100%, height: 3.2cm,
      xlabel: [$n_"dicts,sel"$ active],
      ylabel: [% contentful],
      xaxis: (exponent: 0),
      ..for s in _seeds { (
        lq.plot(s.nz, s.content_frac, mark: none,
                stroke: (paint: _frac_faint, thickness: 0.6pt)),
      ) },
      lq.plot(_mean.nz, _mean.content_frac, mark: "o",
              stroke: (paint: _frac_bold, thickness: 1.5pt)),
      lq.hlines(50, stroke: (paint: gray.darken(80%), dash: "dotted", thickness: 1pt)),
    ),
    caption: [Content fraction of L1-selected atoms.],
  ), <fig:ims-content-frac>,
  columns: (1fr, 1fr),
  caption: [OOMS detector $L_1$ sweep, #_n_seeds seed#if _n_seeds == 1 [] else [s] (faint) with mean (bold).],
  label: <fig:ims-sweep>,
  placement: top,
)
