// AUROC half of fig_ims_sweep, rendered standalone for LaTeX subfigure use.

#import "@preview/lilaq:0.6.0" as lq

#let _ims     = json("aggregated.json").ims_sweep
#let _seeds   = _ims.seeds
#let _mean    = _ims.mean

#let _auroc_faint = rgb(44,  160, 44,  60)
#let _auroc_bold  = rgb(44,  160, 44)

#lq.diagram(
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
)
