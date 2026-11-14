# X-mark canvas tracking (prototype)

For the phone app: keep a guide image locked to the canvas while the phone moves, even when the camera is close enough to see only a small part of the canvas. The canvas carries small hand-drawn X marks; the camera finds them in each frame and works out exactly where it is looking.

This folder is a **simulation**. It renders realistic phone-camera frames of a virtual canvas and runs the tracking on them. Nothing here runs on a phone yet. It is separate from the planner and does not change it.

## How it works

1. **Map.** One wide view of the whole canvas records each X's position, stroke angles and size. In the simulation, the map is the true positions plus 0.5 mm of measurement error.
2. **Detect** (`xtrack/detect.py`). Each frame, find marks that are two roughly straight strokes crossing with arms on both sides. The crossing point is located to about 0.2 pixels.
3. **Lock on** (`xtrack/locate.py`, `Locator.cold_start`). With no idea where the camera is, match groups of detected X's to the map by their spacing, stroke angles and size, then confirm with a full camera model. The lock is refused unless:
   - at least 4 X's agree (5 in a cluttered scene);
   - they span the view;
   - the answer doesn't predict X's where there are none;
   - no different answer fits equally well;
   - the estimated overlay uncertainty is under 10 mm.
4. **Track** (`Tracker`). Follow the camera frame to frame from its last position. While locked, even 1 to 3 visible X's keep the overlay in place (the last pose holds the tilt steady). If the view is lost or jumps, it re-locks automatically. Every estimate carries an uncertainty in mm that the app could show.

The simulated camera (`xtrack/render.py`) is a 66° phone lens at 1280 × 960, at any distance, tilt and rotation. It adds:

- canvas weave and uneven lighting;
- defocus, motion blur, sensor noise and video compression;
- half the time, a hand holding a brush, out of focus and casting a shadow.

The X's are wobbly, bowed, uneven in size and angle, and broken up by the weave.

## Findings

Full tables are in [`results/summary.md`](results/summary.md); example frames are in [`results/samples.jpg`](results/samples.jpg) and [`results/session.jpg`](results/session.jpg).

**Spacing.** Space the X's at **a third to half of your working distance**. Locking on needs four X's in view; once locked, tracking keeps going on fewer.

Share of frames with at least four X's fully in view, with a hand in the way 60% of the time:

| Phone distance from canvas | X every third of the distance | X every half of the distance |
| --- | --- | --- |
| 15 cm | 5 cm: 98% | 7.5 cm: 60% |
| 20 cm | 7.5 cm: 92% | 10 cm: 67% |
| 30 cm | 10 cm: 98% | 15 cm: 46% |

At 10 cm, X's every 5 cm give 56%. The ultra-wide lens helps a lot up close: with X's every 10 cm at 10 cm away, it sees four in 72% of frames, against 0% for the main lens.

On a 24 × 18 in canvas, 10 cm spacing is about 30 X's and 7.5 cm about 48. Four X's near the corners never had four in view closer than 45 cm.

**Accuracy.** Once locked, the overlay was typically **0.3 to 0.8 mm** from where it belongs, averaged over the camera view.

**No confident wrong answers.** Across 1,920 lock-on attempts on three grounds, the app never locked onto the wrong place. When unsure, it refuses and waits for a better view.

**A painting session.** A simulated 30-second session (wide view, move in, close-up with a hand, fast pan):

- with X's every 7.5 cm: tracked **100%** of the time, typically 0.2 to 1.2 mm off;
- with X's every 10 cm: lost lock up to 3.5 s at a time while the hand covered the view.

**Rules that fell out of testing:**

- **Four X's to lock on.** Locking on three was sometimes hundreds of mm wrong, because three points can always be fitted.
- **Fewer can keep a lock.** Once locked, 1 to 3 X's are enough, with the last pose holding the tilt steady.
- **Contrasting marks.** Graphite on the mid-dark mauve ground missed a quarter of the X's. Use dark marks on light grounds and white pencil on dark grounds. The app should be told which, so it only looks for that.
- **Clutter beside the canvas** (a textured wall) produced dozens of false detections at 20 to 30 cm. The checks above handled it, but it is the main source of risk.
- **Hand occlusion** is the hardest case. A phone's gyroscope would bridge it, because it measures tilt changes directly.

## Limitations

- **Synthetic images.** Real cameras add lens distortion, rolling shutter, auto-exposure changes and glare on wet paint. The real pencil and canvas will differ from the simulation.
- **A simple hand model:** a forearm, palm and brush shape.
- **No motion sensors.** The real app would fuse the gyroscope.
- **The map is assumed from a good wide scan,** with 0.5 mm error.
- **The error measure includes off-canvas areas.** It is averaged over the whole view, including parts outside the canvas, which overstates error near the edges.
- **Python speed isn't representative.** Detection takes about 0.3 s and lock-on about 2 s per frame on one core. Tracking only needs detection, and an on-phone implementation in C++ or on the GPU would be far faster.

## Next steps

1. **Test on real phone video** of a canvas with drawn X's: walk around, move close, paint.
2. **Track the painting itself,** so X's can be erased as painted areas build up texture.
3. **Fuse the gyroscope** for hand occlusion and fast moves.
4. **Port to Android.** OpenCV runs on Android, and the logic here maps directly.

## Run it

```
.venv/bin/pip install -e ".[tracking]"                      # adds OpenCV
.venv/bin/python -m pytest prototypes/x_tracking/tests       # unit tests
cd prototypes/x_tracking
../../.venv/bin/python run_experiments.py --out results/     # full run takes about half an hour on 16 cores
```

`--quick` runs a few minutes' worth. `--experiments`, `--looks` and `--session-spacing` choose what to run.
