# Sketch alignment: simulation results

Images: `landscape` (the planner tests' synthetic scene, simple shapes), `print2` (the author's own artwork on a mauve ground, not in the repo), `rainbow` (a busy photo: "Rainbow" by andyarthur on Flickr, CC BY 2.0, not in the repo).

Errors are in mm on a 24 x 18 in canvas, averaged over the parts of the image within 4% of the diagonal of something sketched; the median over trials is shown. **Found** = the right placement (error within twice the sketch's own drift + 3 mm). **Drift** = the best any single affine (or, for a raw photo, perspective) map could do: how far the freehand sketch strays from the image. **Placed** = the fitted placement alone; **Bent** = with the smooth bending on top.

## E1: sketch style (snap from a rough placement)

| condition | image | found | drift | placed | bent | bent, 90th pct (whole image) | time (s) |
|---|---|---|---|---|---|---|---|
| careful | landscape | 10/10 | 1.7 | 1.6 | 1.2 | 5.2 | 9.1 |
| careful | print2 | 10/10 | 1.8 | 2.0 | 1.4 | 5.3 | 10.6 |
| careful | rainbow | 10/10 | 1.9 | 2.1 | 1.3 | 3.0 | 12.0 |
| typical | landscape | 10/10 | 4.3 | 4.1 | 3.0 | 12.4 | 9.2 |
| typical | print2 | 10/10 | 5.0 | 5.7 | 4.7 | 11.6 | 10.8 |
| typical | rainbow | 9/10 | 4.7 | 6.3 | 4.6 | 8.9 | 12.5 |
| loose | landscape | 4/10 | 9.7 | 10.4 | 9.9 | 25.0 | 10.0 |
| loose | print2 | 4/10 | 9.2 | 11.0 | 10.0 | 26.0 | 10.8 |
| loose | rainbow | 8/10 | 9.1 | 14.3 | 14.0 | 25.8 | 11.1 |

## E2: finding the image with no help (typical sketch)

| condition | image | found | drift | placed | bent | bent, 90th pct (whole image) | time (s) |
|---|---|---|---|---|---|---|---|
| upright | landscape | 10/10 | 4.5 | 4.3 | 3.0 | 11.9 | 12.9 |
| upright | print2 | 8/10 | 4.6 | 5.3 | 4.0 | 11.9 | 15.1 |
| upright | rainbow | 9/10 | 4.6 | 6.9 | 4.8 | 9.1 | 15.8 |
| any | landscape | 10/10 | 4.8 | 4.9 | 3.7 | 14.2 | 118.2 |
| any | print2 | 7/10 | 4.5 | 4.3 | 3.0 | 9.9 | 112.9 |
| any | rainbow | 9/10 | 4.7 | 6.5 | 5.2 | 10.3 | 110.8 |

## E3: extra lines on the canvas (typical sketch, snap)

| condition | image | found | drift | placed | bent | bent, 90th pct (whole image) | time (s) |
|---|---|---|---|---|---|---|---|
| none | landscape | 10/10 | 4.9 | 4.7 | 3.2 | 13.2 | 8.4 |
| none | print2 | 10/10 | 4.3 | 4.7 | 3.4 | 10.1 | 11.2 |
| none | rainbow | 10/10 | 5.0 | 5.9 | 4.6 | 9.9 | 10.9 |
| grid | landscape | 10/10 | 4.9 | 4.8 | 3.3 | 11.7 | 8.3 |
| grid | print2 | 10/10 | 4.5 | 5.0 | 4.4 | 11.8 | 9.5 |
| grid | rainbow | 9/10 | 4.9 | 7.0 | 4.6 | 10.3 | 10.8 |

## E4: raw photo, no tracking (typical sketch, snap)

| condition | image | found | drift | placed | bent | bent, 90th pct (whole image) | time (s) |
|---|---|---|---|---|---|---|---|
| raw photo | landscape | 8/10 | 4.6 | 4.5 | 3.3 | 12.7 | 6.5 |
| raw photo | print2 | 8/10 | 4.5 | 5.1 | 3.6 | 13.7 | 8.6 |
| raw photo | rainbow | 7/10 | 4.6 | 6.8 | 5.5 | 10.3 | 9.6 |

## E5: graphite on white canvas (typical sketch, snap)

| condition | image | found | drift | placed | bent | bent, 90th pct (whole image) | time (s) |
|---|---|---|---|---|---|---|---|
| white canvas, graphite | landscape | 9/10 | 4.9 | 5.0 | 3.6 | 13.0 | 8.2 |
| white canvas, graphite | print2 | 9/10 | 4.8 | 5.6 | 4.5 | 13.3 | 10.2 |
| white canvas, graphite | rainbow | 10/10 | 4.9 | 6.8 | 5.7 | 10.9 | 7.3 |

## Telling a good result from a bad one

Score margin (best clearly different placement minus the chosen one; bigger = more certain): found median 0.167 (5th pct 0.084), missed median 0.095 (95th pct 0.429).
- accept when margin >= 0.05: 262/270 accepted, 25 of them wrong
- accept when margin >= 0.08: 248/270 accepted, 18 of them wrong
- accept when margin >= 0.1: 230/270 accepted, 16 of them wrong
