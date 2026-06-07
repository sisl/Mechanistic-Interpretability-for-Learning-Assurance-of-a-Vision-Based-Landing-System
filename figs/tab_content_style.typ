// Hand-written content/style split table. Values come from
// figs/aggregated.json, emitted by scripts/aggregate.py.

#import "@preview/zero:0.6.1": num

#let _agg = json("aggregated.json")
#let _n_pre = _agg.n_seeds.pretrained
#let _n_scr = _agg.n_seeds.scratch
#let _n_max = calc.max(_n_pre, _n_scr)

// Content/style is a fraction; two decimals is enough to see that the two
// columns sum to ~1.00 without cluttering the cell.
// zero's `num` only rounds the mean side of "a+-b"; pre-round both parts
// so the spread doesn't leak full float precision.
#let _round = (mode: "places", precision: 2, pad: true)
#let _fmt(pair) = {
  if pair == none { return text(style: "italic")[pending] }
  let m = calc.round(pair.at(0), digits: 2)
  let s = calc.round(pair.at(1), digits: 2)
  num(str(m) + "+-" + str(s), round: _round)
}

#let lcell(it) = table.cell(inset: (left: 0pt))[#it]
#let rcell(it) = table.cell(inset: (right: 0pt))[#it]

// Pretrained is our reference row; bold both of its cells to echo its role
// as the variant the paper narrative leans on. Scratch stays plain weight.
#let _row(label, variant, bold: false) = {
  let cs = _agg.content_style.at(variant, default: none)
  let lab = if bold { strong(label) } else { [#label] }
  if cs == none {
    (lcell[#lab], table.cell(colspan: 2, align: center)[_pending_])
  } else {
    let c = _fmt(cs.content); let s = _fmt(cs.style)
    if bold { (lcell[#lab], strong(c), rcell[#strong(s)]) }
    else    { (lcell[#lab], c,         rcell[#s]) }
  }
}

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    inset: 6pt, stroke: none,
    table.hline(stroke: 0.8pt),
    lcell[*Variant*], [*Contentful*], rcell[*Stylistic*],
    table.hline(stroke: 0.5pt),
    .._row("Pretrained", "pretrained", bold: true),
    .._row("Scratch",    "scratch"),
    table.hline(stroke: 0.8pt),
  ),
  caption: [Head-weight split across the dictionary (#_n_max seed#if _n_max == 1 [] else [s]).],
  kind: table,
  placement: top,
) <tab:content-style>
