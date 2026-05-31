# Experiment 10

This directory contains four self-contained experiment groups for Experiment 10:

- `01_stdp_only`
- `02_stdp_rate_target_ip`
- `03_stdp_spikl_ip`
- `04_spikl_ip_only`

Each group contains:

- `run.py`
- `group_settings.json`
- `requirements.txt`
- `src/`
- `output/`

Run a single group:

```bash
cd 03_stdp_spikl_ip
python run.py
```

Run all groups:

```bash
python run_all_groups.py
```
