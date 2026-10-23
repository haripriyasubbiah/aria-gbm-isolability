# Second draw of real Fermi-GBM triggers (disjoint from the first 100)

Copy the two files into the repo (same relative paths) and add this line to
package.json "scripts", next to "test:gbm":

    "test:gbm:heldout": "npx tsx src/lib/aria/real_eval_heldout.ts",

## Run (repo root; needs `pip install astropy numpy`, ~0.8 GB download, 10-30 min)

    python3 experiments/gbm/prepare_draw2.py --per-class 25 --draw-seed 2
    # -> experiments/gbm/cubes_draw2.jsonl, triggers_draw2.csv

    # Train on draw 1, score on draw 2 (true held-out; aborts on any shared trigger)
    npm run test:gbm:heldout -- experiments/gbm/cubes.jsonl experiments/gbm/cubes_draw2.jsonl --out experiments/gbm/heldout_1to2.json
    # Reverse direction
    npm run test:gbm:heldout -- experiments/gbm/cubes_draw2.jsonl experiments/gbm/cubes.jsonl --out experiments/gbm/heldout_2to1.json

    # Pooled 200-cube run through your existing evaluator
    cat experiments/gbm/cubes.jsonl experiments/gbm/cubes_draw2.jsonl > experiments/gbm/cubes_pooled.jsonl
    npm run test:gbm -- experiments/gbm/cubes_pooled.jsonl

## Reading the output
- The held-out file reports `heldout` (disjoint draw) and `inSample` (same-draw) side by side. Their gap
  is direct evidence on the open issue that trainOnCubes/evaluateReal share known-class cubes.
- Training-seed std is 0 on real cubes (your finding), so it is not an uncertainty estimate. The
  uncertainty is which triggers were drawn. Two draws give you two points, not error bars. For error
  bars add draws: `--draw-seed 3 --tag draw3 --exclude experiments/gbm/triggers.csv experiments/gbm/triggers_draw2.csv`
  and report mean +/- std across draws (3+ draws, say so in the paper).
- If a class pool runs out (TGF is the likely one) the script prints WARNING and keeps what it found;
  lower --per-class then. Raw FITS are cached in experiments/gbm/raw/, so reruns are resumable.

## What was tested here
Held-out evaluator and overlap abort were run on a split of your 100 cubes. The downloader's exclusion,
shuffling and per-class quota logic were run against a mocked catalog. The real HEASARC download could not be run from my sandbox.
