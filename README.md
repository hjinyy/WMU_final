# WMU_final — Paper Figure Package

This repository contains only the resistance-generalization paper-figure deliverables and their reproducible source package. It is a clean, standalone publication repository; unrelated experiments, model audits, temporary smoke outputs, and the original research repository are not included.

## Final outputs

- `results/paper_figures_resistance_generalization_v1/` — Figures 1–8, PNG/PDF, per-figure captions, plotted-data CSVs, source metadata.
- `results/paper_figures_resistance_generalization_v1_fig04_sso_audited_replacement/` — separately versioned corrected Fig. 4, selected raw-case hashes, plot data, and provenance. **Use this Fig. 4 instead of `fig04_sso_waveform_characteristics` in the original eight-figure folder.** The earlier overlay is retained as history, not endorsed as the SSO evidence figure.
- `results/fig04_sso_pre_fault_persistence_audit_v1/` — supporting pre-fault persistence diagnostic and numerical table.

The project has not run Simulink or regenerated waveform data. Only six matched archived raw cases are included, reduced to the time and voltage/current channels needed by these figures. Their original full-file SHA-256 hashes and included-channel lists are in `data/source_case_provenance.csv`. The complete 1,080-case corpus is not redistributed.

## Reproduce / verify

Python 3.11+ is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
```

The committed plot artifacts are the publication record. To intentionally regenerate the three figure/audit roots from the included frozen feature/prediction tables and selected waveform channels:

```bash
python scripts/render_resistance_generalization_figures_v1.py --force
python scripts/render_fig04_sso_audited_replacement_v1.py --force
python scripts/audit_fig04_sso_pre_fault_persistence_v1.py --force
```

`--force` replaces only the exact output directory owned by each script. The scripts do not invoke Simulink, retrain models, or alter the source tables. `scripts/run_unseen_resistance_spatial_generalization_v2.py` is included as the feature-table/schema helper used by the renderer; do not execute it for figure reproduction because it performs model evaluation.

## SSO bus provenance and claim boundary

For the current 1,080-case corpus, execution lineage identifies IEEE14 Bus 7 and documents IEEE30 Bus 30 as the SSO reference/injection buses. IEEE30 Bus 30 is **not** claimed to be a physically connectivity-validated PCC. IEEE14 Bus 14 belongs to a separate later 540-case run and is not used here. The corrected Fig. 4 reports background-dependent signatures in archived waveforms; it does not validate a physical transfer function, modal stability, or a sustained 25-Hz mode.

## Font

Times New Roman was requested. The render host did not have it installed, so committed PDFs embed Liberation Serif (the available Times-compatible serif). The generators detect Times New Roman and use it automatically when installed; rendered-font provenance is recorded in each result root.

## Repository contents

- `data/source_corpus/` — manifest, frozen feature tables, and six selected-channel waveforms used for figure regeneration.
- `data/placements/` and selected frozen predictions/summary tables under `results/` — compact source tables required by the figure code.
- `scripts/` — rendering and audit code.
- `src/wmu_project/basic_v1/` — topology constants needed by the renderer.
- `tests/` — artifact and provenance checks.
- `data/package_sha256.csv` — SHA-256 inventory of packaged code, data, and result artifacts.
