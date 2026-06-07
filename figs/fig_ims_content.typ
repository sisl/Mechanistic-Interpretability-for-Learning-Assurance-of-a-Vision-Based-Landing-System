// Content-fraction half of fig_ims_sweep, rendered standalone for LaTeX subfigure use.

#import "@preview/lilaq:0.6.0" as lq

#let _ims     = json("aggregated.json").ims_sweep
#let _seeds   = _ims.seeds
#let _mean    = _ims.mean

#let _frac_faint  = rgb(214, 39,  40,  60)
#let _frac_bold   = rgb(214, 39,  40)

#lq.diagram(
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
)
