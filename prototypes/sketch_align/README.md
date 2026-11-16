# Aligning the image to a sketch (prototype)

For the phone app: many painters sketch the composition on the canvas before they paint. If the app can find the image in that sketch, it can:

1. **place the plan for you**, instead of you dragging and pinching it into place;
2. **bend the plan to follow your drawing** where the freehand sketch strayed from the image, so the colour regions land on your lines.

This folder is a **simulation**, like the X-tracking prototype next to it. It draws a freehand sketch of an image on a virtual canvas, photographs it with a virtual phone, and tries to find the image in the photo. The placement part (1) is now in the app as **Snap to drawing**; see [In the app](#in-the-app-snap-to-drawing).

## How it works

**The simulated sketch** (`salign/sketch.py`, `salign/photo.py`):

- **What gets drawn.** The outlines are the boundaries of the image's main colour regions. The drawing is broken into pen strokes, and some strokes are left out, more often where the colour contrast is low. Detail is simplified, ends overshoot, and some lines are drawn twice.
- **Freehand error.** The whole drawing drifts smoothly, with proportions stretched or sheared. Each stroke is also nudged on its own.
- **Lines that aren't in the image.** The canvas also carries:
  - tracking X's every ~10 cm;
  - construction lines;
  - patches of shading hatches;
  - optionally, a transfer grid.
- **The photo.** White pencil on the mauve ground, or graphite on white canvas, catching on the canvas tooth. It is photographed from 60 to 72 cm, tilted up to 20°, with the X-tracking prototype's lighting, blur, noise and JPEG.
- **Three sketch styles:**

  | Style | Smooth drift (share of the image diagonal) | Proportion error | Strokes left out |
  |---|---|---|---|
  | careful | 0.4% | 1% | 20% |
  | typical | 1% | 2.5% | 35% |
  | loose | 2% | 5% | 50% |

  On this canvas, the typical style comes out at about 4 mm off on average, against the best affine fit.

**The aligner** (`salign/lines.py`, `salign/align.py`):

1. **Straighten.** In X or marker mode, the app already knows where the canvas is, so the photo is resampled flat onto the canvas at 2 px/mm. Without tracking (a phone on a stand), the raw photo is used and perspective is fitted along with everything else.
2. **Find the pencil lines.** A pixel is on a line if the image curves like a thin ridge across it and it is lighter (or darker, for graphite) than the ground on *both* sides. The both-sides test ignores the edge of a painted area. Thresholds are in units of the photo's own noise.
3. **Describe the image.** Its colour edges are sorted into 8 directions, each with a distance map ("how far to the nearest edge running this way").
4. **Find the image.** There are two ways:
   - **Snap:** start from a rough placement made with the usual drag and pinch, then search around it. That covers ±25% in size, ±12° of turn, and a shift of up to 25% of the image size.
   - **Search with no help:** every size and every turn, either kept upright (±15°) or at any angle.

   Either way, a blurred copy of the image's edges slides over the sketch lines (FFT correlation), and the best 6 distinct placements go on to step 5.
5. **Refine the placement.** Each sketch point is matched to the nearest image edge running the same way (within about 30°). The placement is solved by robust least squares while the match radius shrinks. It goes through similarity, then affine (with tracking) or perspective (raw photo). The placement whose lines sit closest to image edges wins.
6. **Bend.** A smooth displacement grid goes on top, refined coarse to fine (2, then 4, then 8 cells across). Matching runs both ways: each sketch point looks for an image edge, and the image's strongest edges look for a sketch line, since a painter draws those. The reverse direction is what stops a drifted outline from snapping onto the wrong one of several parallel edges.

## Findings

Full tables are in [`results/summary.md`](results/summary.md). Example pictures are in [`results/samples.jpg`](results/samples.jpg): the photo, then the pencil lines found (yellow) with the image's edges where they were placed (green), then the image laid on the canvas. Each condition ran 10 times on each of 3 images: a simple synthetic landscape, the author's own artwork on mauve, and a busy photo.

Errors below are the median, in mm, over the parts of the image near something sketched.

**Snapping from a rough placement works.** With careful or typical sketches, in X/marker mode, the right placement was found in **146 of 150** tries. That covers styles, extra lines and graphite on white.

| Sketch | Found | Sketch's own drift | Placed | Bent |
| --- | --- | --- | --- | --- |
| careful | 30/30 | 1.7 to 1.9 | 1.6 to 2.1 | **1.2 to 1.4** |
| typical | 29/30 | 4.3 to 5.0 | 4.1 to 6.3 | **3.0 to 4.7** |
| loose | 16/30 | 9.1 to 9.7 | 10 to 14 | 10 to 14 |

**The placement is only as good as the drawing.** Placing the image without bending lands about as far off as the freehand sketch drifts.

**Bending follows the drawing part of the way.** It cuts the error near the content by about a quarter to a third, and it beats any affine fit.

**It doesn't follow everywhere.** It is weaker in large empty areas, where there is nothing to follow. The 90th percentile over the whole image stays at 9 to 13 mm for typical sketches.

**Loose sketches** (5% proportion errors) are not reliable: about half were found, and those were still about 10 mm off.

**Finding the image with no help:**

- **Kept upright** (±15°): 27 of 30 found, about 1.5 times as slow as a snap.
- **At any rotation:** 26 of 30, and about 10 times slower. All three gross failures were on the near-circular artwork, which also fits when turned.

So the app should keep the drag-roughly-then-snap flow, with an upright search as an extra.

**Extra lines don't matter.** X's, guide lines, hatching and a transfer grid made no difference: 59 of 60 found without the extras or with a grid.

**Graphite on white** behaves like white pencil on mauve: 28 of 30.

**A raw photo, without tracking** (the phone on a stand, perspective fitted directly): 23 of 30. When it missed, it usually landed near the right place but stopped short; tracking makes it more reliable.

**Knowing when it failed is not solved.** The gap between the best placement and the next-best clearly different one does not separate good results from bad cleanly. The app should show the result and let the painter accept it, nudge it, or snap again.

**Time.** On this PC, one thread, a snap takes about 3 s:

| Step | Time |
| --- | --- |
| Finding the lines | 0.2 s |
| Search | 1 s |
| Refining | 0.6 s |
| Bending | 1 s |

Expect a few times that on a mid-range phone, which is fine for a one-off button. The search and bending can be run at lower resolution if needed.

## In the app: Snap to drawing

The app now has a **Snap to drawing** button on the paint screen, a Kotlin port of steps 2 to 5 above (`android/app/src/main/java/.../align/`). The root README covers how to use it and why it works this way. Compared with the prototype:

- **It works on one 1280 x 960 camera frame**, not a full-resolution still. The search runs on a 256-pixel copy anyway, and a real chalk sketch was found at that size: the image's centre landed within 3 to 6 px of the full-resolution result.
- **It always fits full perspective** (the raw frame), as in the "raw photo" experiment. It searches around wherever the image is drawn now, *and* the whole frame upright on screen. With no X's there is often no useful starting placement.
- **There's no bending yet.** On the real sketch, a plain placement was already within about 1 to 2 mm of the drawn lines.
- **The result is held like a locked view**: phone on a stand, snap once. When X's or markers are tracking, the snapped spot also becomes the image's placement.
- **The search pads the frame by 0.6 of the image's size, not 1.25.** Positions with under half the image in view are rejected anyway. This change is in the prototype too, with identical results, and cut the search on a Galaxy A13 from 17.5 s to 6.5 s. Running the trial positions on all cores brought it to about 2.3 s. A snap takes about 4.5 s in all.

`export_test_case.py` writes test cases for the app's `SketchAlignTest`: a frame, the image, and this prototype's answer. The test checks the Kotlin port matches.

- **On the synthetic case:** 0.0 px apart.
- **On the real sketch:** median 2.4 px, 90th percentile 4.4 px. The real sketch stays out of git (`assets/local/`).

Still to do:

- **Bending** the plan to follow the drawing (step 6), drawn with `Canvas.drawBitmapMesh`.
- **A full-resolution still** for very fine drawings.
- **A way to tell when a snap is wrong.**

## What this doesn't tell us yet

- **Real sketches.** Simulated sketches are not real ones. The simulator draws region outlines; real painters also draw form lines, shadow shapes, and notes to themselves, and leave out things the simulator keeps. The real test is a photo of an actual sketch.
- **Real canvas, chalk and phone camera.** The X detector worked on the real mauve canvas, which is encouraging, but this hasn't been tried.
- **Following each line exactly.** Bending follows the smooth trend of the drawing, not each stroke. A stroke drawn 3 mm off on its own stays 3 mm off.
