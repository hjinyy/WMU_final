# WMU_final — figure and resistance-generalization package

This repository is organized by file responsibility. It contains the final figure deliverables, frozen inputs, a self-contained Step 3 resistance-generalization analysis, the small reusable calculations needed to reproduce the figures, and validation/documentation. It is not a duplicate of the working Simulink research repository.

## Repository map

- `data/frozen_inputs/` — immutable inputs used by the renderers:
  - `waveforms_selected/`: six reduced-channel waveforms for the matched SSO comparison.
  - `features/`: frozen feature tables for IEEE14 and IEEE30.
  - `predictions/`: archived model predictions used in the figures.
  - `metrics/`: archived evaluation summaries used in the figures.
  - `manifests/`: case metadata for matching the selected waveform files.
- `data/reference/placements/` — fixed WMU placement lookup tables.
- `data/provenance/` — source case hashes and the package SHA-256 inventory.
- `analysis/fault_generalization_step3_v1/` — frozen Step 3 train/selection/test analysis, its model outputs, input tables, portable scripts, tests, and runbook. It uses 0.1/1 Ω for training/model selection and reports the new34 10 Ω cohort separately.
- `src/wmu_final/` — reusable topology constants and frozen-feature/spatial-normalization functions. It does not train models or run simulations.
- `scripts/render/` — figure generation entry points: the eight-figure set and the separate audited Fig. 4 replacement.
- `scripts/audit/` — pre-fault persistence diagnostic entry point.
- `outputs/` — generated/delivered artifacts only. Each product has its own directory and contains image/PDF, caption, plot data, metadata, and diagnostics as applicable.
- `tests/` — layout, provenance, and output checks.
- `docs/` — project notes.

## Final deliverables

- `outputs/paper_figures_v1/` — eight figures in PNG/PDF plus captions, plot data, and render metadata. Its Fig. 4 overlay is retained as historical context only.
- `outputs/fig04_sso_audited_replacement_v1/` — separate audited replacement. Use this instead of Fig. 4 in the eight-figure set for SSO-background evidence.
- `outputs/fig04_pre_fault_persistence_audit_v1/` — supporting pre-fault persistence diagnostic.
- `analysis/fault_generalization_step3_v1/` — Step 3 ML evaluation on new34 10 Ω cases, with existing representative 10 Ω cases excluded from scoring.

Only six matched raw waveforms are packaged for the figure analyses. The full raw waveform corpus and Simulink models are not redistributed. The separate Step 3 package includes derived Step 2 feature tables and the integrated manifest, but no raw waveform corpus; its runner filters out the 360 previously examined representative 10 Ω cases before model-matrix construction and scoring. No Simulink execution or waveform regeneration is part of this repository.

## Reproduce and verify

Python 3.11+ is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q tests
```

To intentionally regenerate the outputs from the packaged frozen inputs:

```bash
python scripts/render/render_resistance_generalization_figures_v1.py --force
python scripts/render/render_fig04_sso_audited_replacement_v1.py --force
python scripts/audit/audit_fig04_sso_pre_fault_persistence_v1.py --force
```

`--force` replaces only the specific output directory owned by each figure script. Those figure scripts do not invoke Simulink, retrain models, or modify frozen input tables.

For the separately scoped Step 3 analysis:

```bash
python -m unittest discover -s analysis/fault_generalization_step3_v1/tests -v
python analysis/fault_generalization_step3_v1/scripts/evaluate_step3_unseen10_fixed3wmu.py --output outputs/reproduction_v1
python analysis/fault_generalization_step3_v1/scripts/select_step3_models_trainonly_condition_grouped.py --output outputs/reproduction_condition_grouped_v1
python analysis/fault_generalization_step3_v1/scripts/verify_step3_outputs.py
```

The Step 3 evaluator refuses to overwrite an existing output directory. A reproduction run uses the same archived 10 Ω new34 cases and is not an independent second holdout. The canonical checked-in outputs are under `analysis/fault_generalization_step3_v1/outputs/run_v1/`; see its `README.md` for the scope and limitations.

## SSO provenance and claim boundary

The current 1,080-case corpus lineage identifies IEEE14 Bus 7 and documents IEEE30 Bus 30 as SSO execution-reference buses. IEEE30 Bus 30 is not claimed to be a physically connectivity-validated PCC. IEEE14 Bus 14 belongs to a separate later 540-case run and is not used here. The figures document archived waveform behavior; they do not establish physical transfer functions, modal stability, or sustained 25-Hz oscillation.

## Font

Times New Roman was requested but was unavailable on the render host. The committed PDFs use Liberation Serif, with the fallback recorded in metadata. The generators detect and use Times New Roman automatically when it is installed.
