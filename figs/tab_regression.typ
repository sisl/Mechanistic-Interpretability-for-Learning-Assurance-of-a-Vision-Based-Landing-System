// Hand-written table; iterate on layout here. Values come from
// figs/aggregated.json, emitted by scripts/aggregate.py.
//
// Formatting of individual cells goes through the `zero` package's `num`,
// which understands `"1.86+-0.00"` as "mean ± uncertainty" and honors the
// global `set-round` below so rounding is controlled in one place.

#import "@preview/zero:0.6.1": num

#let _agg = json("aggregated.json")
#let _n_pre = _agg.n_seeds.pretrained
#let _n_scr = _agg.n_seeds.scratch
#let _n_max = calc.max(_n_pre, _n_scr)

// Format a [mean, std] pair via zero's `num`. One decimal place, pad
// trailing zero. Setting round inline (rather than via `set-round` at the
// module top level) avoids emitting a content token that breaks the
// table row structure when the file is #included into main.typ.
// zero's `num` only rounds the mean side of a "a+-b" string; pre-round
// both parts in typst before handing them over so the spread doesn't
// leak its full float precision.
#let _round = (mode: "places", precision: 1, pad: true)
#let _fmt(pair) = {
  if pair == none { return text(style: "italic")[pending] }
  let m = calc.round(pair.at(0), digits: 1)
  let s = calc.round(pair.at(1), digits: 1)
  num(str(m) + "+-" + str(s), round: _round)
}

// Bold whichever variant has the lower (= better) value for a metric.
// Lives inside the file so the hand-written layout below can just call
// _pair("test_mae") and unpack two already-styled cells.
#let _pair(key) = {
  let pr = _agg.regression.pretrained
  let sc = _agg.regression.scratch
  let pv = if pr != none { pr.at(key, default: none) } else { none }
  let sv = if sc != none { sc.at(key, default: none) } else { none }
  if pv == none or sv == none { return (_fmt(pv), _fmt(sv)) }
  if pv.at(0) <= sv.at(0) { ([*#_fmt(pv)*], _fmt(sv)) }
  else                     { (_fmt(pv),       [*#_fmt(sv)*]) }
}

// Booktabs flush-edge cell helpers (local copies; typst #include doesn't
// share scope with the parent module).
#let lcell(it) = table.cell(inset: (left: 0pt))[#it]
#let rcell(it) = table.cell(inset: (right: 0pt))[#it]

#let (pre_trmae, scr_trmae) = _pair("train_mae")
#let (pre_trmed, scr_trmed) = _pair("train_median")
#let (pre_temae, scr_temae) = _pair("test_mae")
#let (pre_temed, scr_temed) = _pair("test_median")

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    inset: 6pt, stroke: none,
      column-gutter: (0.5em, 0em, 1.0em, 0em),

    table.hline(stroke: 0.8pt),

    // Two-row header with grouped Train / Test supercolumns. Each super
    // cell carries its own bottom stroke so there is a visible gap at
    // the col 2/3 boundary (typst hline only takes integer col indices,
    // so a single table.hline can't leave a trim between the two groups).
      table.cell(rowspan: 2, inset: (left: 0pt), align(left)[#v(0.8em) *Variant*]),
    table.cell(colspan: 2, align: center,
               // stroke: (bottom: 0.4pt),
               inset: (bottom: 3pt, right: 6pt))[*Train*],
    table.cell(colspan: 2, align: center,
               // stroke: (bottom: 0.4pt),
               inset: (bottom: 3pt, left: 6pt, right: 0pt))[*Test*],
    table.hline(start: 1, end: 3, stroke: 0.5pt),
    table.hline(start: 3, end: 5, stroke: 0.5pt),
      lcell[*MAE [px]*], rcell[*Median [px]*], lcell[*MAE [px]*], rcell[*Median [px]*],

    table.hline(stroke: 0.5pt),

    // Data rows: pretrained and scratch, or a spanned "pending" cell
    // when that variant has not produced final metrics yet.
    ..if _agg.regression.pretrained == none {
      (lcell[Pretrained], table.cell(colspan: 4, align: center)[_pending_])
    } else {
      (lcell[Pretrained], pre_trmae, pre_trmed, pre_temae, rcell[#pre_temed])
    },
    ..if _agg.regression.scratch == none {
      (lcell[Scratch], table.cell(colspan: 4, align: center)[_pending_])
    } else {
      (lcell[Scratch], scr_trmae, scr_trmed, scr_temae, rcell[#scr_temed])
    },

    table.hline(stroke: 0.8pt),
  ),
  caption: [Regression performance on LARDv2 (#_n_max seed#if _n_max == 1 [] else [s]).],
  kind: table,
  placement: top,
) <tab:training>
