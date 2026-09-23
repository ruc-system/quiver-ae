# Pre-executed logs

The results we obtained on the AE server, shipped with the repository so you can
see every figure before running anything. They are used only when you point the
plotter at this directory; a plain `./ae/scripts/plot_all.py` plots your own runs
and skips the experiments you have not run.

This snapshot is the 2026-09-23 AE-server run that produced the current PDFs in
`ae/figures/`. One directory per experiment, holding the archived CSV layout:

```
Pre-executed-logs/
  latency_qps/    env.txt  metrics.csv
  io_latency/     env.txt  breakdown.csv  hop_samples.csv  metrics.csv
  e2e/            env.txt  metrics.csv
  fusion/         env.txt  metrics.csv
  ablation/       env.txt  metrics.csv  breakdown.csv
  q_sensitivity/  env.txt  metrics.csv  breakdown.csv  cta_samples.csv
```

To plot this set alone, ignoring any run of your own:

```bash
./ae/scripts/plot_all.py ae/results/Pre-executed-logs -o /tmp/pre-executed
```

## Refreshing these logs (authors only)

Copy the CSVs of a completed run into the matching directory:

```bash
cp ae/results/e2e/{env.txt,metrics.csv} ae/results/Pre-executed-logs/e2e/
```
