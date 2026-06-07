// Atom visualization: top contentful (left) vs top stylistic (right)
// Curated for paper: 4 content atoms = runway keypoint regions,
// 4 style atoms = rendering-specific patterns. Real patches with 3-patch
// context window, red border on target patch. Two-column wide (scope: parent).

#{
  let cell-w = 1.55cm
  let cell-h = 1.55cm
  let gap = 1.5pt
  let n-patches = 4

  // Okabe-Ito palette (colorblind-safe): blue, vermillion
  let c-content = rgb("#0072b2")
  let c-style = rgb("#d55e00")

  // Content: runway-relevant structure (low CV)
  let content = (
    ("atoms_content_top/rank01_atom111", "Threshold"),
    ("atoms_content_top/rank00_atom033", "Runway edge"),
    ("atoms_content_extra/rank07_atom209", "Far endpoint"),
    ("atoms_content_extra/rank04_atom063", "Grass boundary"),
  )

  // Style: rendering-specific patterns (high CV)
  let style = (
    ("atoms_style_top/rank481_atom354", "Blocky sky"),
    ("atoms/rank511_atom271", "Motion blur"),
    ("atoms_style_top/rank484_atom327", "Satellite grass"),
    ("atoms/rank509_atom123", "Aerial forest"),
  )

    set text(size: 0.9em)

  grid(
    columns: 2,
    column-gutter: 2em,
    {
        align(center)[#text(fill: c-content, weight: "bold", size: 1.2em)[#h(5em) Contentful atoms (low CV)]]
        v(-1.0em)
      for (dir, label) in content {
        grid(
            columns: (7.5em, ..range(n-patches).map(_ => cell-w)),
          column-gutter: gap,
          row-gutter: gap,
            align(horizon+right)[#text(size: 1em, fill: c-content)[#label]#h(1em)],
          ..range(n-patches).map(j => {
            let fname = if j < 9 { "0" + str(j + 1) } else { str(j + 1) }
            image(dir + "/" + fname + ".png", width: cell-w, height: cell-h, fit: "cover")
          }),
        )
        v(gap)
      }
    },
    {
        align(center)[#text(fill: c-style, weight: "bold", size: 1.2em)[#h(5em) Stylistic atoms (high CV)]]
        v(-1.0em)
      for (dir, label) in style {
        grid(
            columns: (7.5em, ..range(n-patches).map(_ => cell-w)),
          column-gutter: gap,
          row-gutter: gap,
            align(horizon+right)[#text(size: 1em, fill: c-style)[#label]#h(1em)],
          ..range(n-patches).map(j => {
            let fname = if j < 9 { "0" + str(j + 1) } else { str(j + 1) }
            image(dir + "/" + fname + ".png", width: cell-w, height: cell-h, fit: "cover")
          }),
        )
        v(gap)
      }
    },
  )
}
