// Hand-written CV histogram; iterate on layout here. Data (binned already,
// 40 equal-width bins up to the 99th percentile) comes from
// figs/aggregated.json, emitted by scripts/aggregate.py. Single seed
// (pretrained_seed0); see the `cv_hist` block in that JSON.

#import "@preview/lilaq:0.6.0" as lq
#import "@preview/tiptoe:0.3.1"

#let _cv = json("aggregated.json").cv_hist

// Arrow annotations narrate the content/style split: both start on the
// median-CV vline at ~90% of bar height and point outward. Lengths scale with
// the visible x-range so the layout survives data changes.
#let _y_max  = _cv.bin_counts.fold(0, calc.max)
#let _y_ann  = _y_max * 0.9
#let _x_lo   = _cv.bin_centers.first() - _cv.bin_width / 2
#let _x_hi   = _cv.bin_centers.last()  + _cv.bin_width / 2
#let _arrow  = (_x_hi - _x_lo) * 0.28

#lq.diagram(
  width: 7.5cm, height: 4.5cm,
  xlabel: [Cross-subset CV],
  ylabel: [#sym.hash atoms],
  title: none,
  lq.bar(_cv.bin_centers, _cv.bin_counts, width: _cv.bin_width,
         fill: rgb(50, 110, 180, 180)),
  lq.vlines(_cv.median_cv,
            stroke: (paint: gray.darken(80%), dash: "dashed", thickness: 1pt)),
  lq.path((_cv.median_cv, _y_ann), (_cv.median_cv - _arrow, _y_ann),
          tip: tiptoe.stealth, stroke: 0.8pt + black),
  lq.path((_cv.median_cv, _y_ann), (_cv.median_cv + _arrow, _y_ann),
          tip: tiptoe.stealth, stroke: 0.8pt + black),
  lq.place(_cv.median_cv - _arrow / 2, _y_ann, align: bottom,
      pad(bottom: 0.1em, box(fill: white.transparentize(10%), inset: 0.2em, radius: 0.2em)[#text(size: 1.00em)[contentful]])),
  lq.place(_cv.median_cv + _arrow / 2, _y_ann, align: bottom,
      pad(bottom: 0.1em, box(inset: 0.2em, text(size: 1.00em)[stylistic]))),
)
