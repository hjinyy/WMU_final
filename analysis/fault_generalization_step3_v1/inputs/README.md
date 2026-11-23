# Frozen inputs for Step 3

These are derived-analysis inputs copied without modification from the completed Step 1/2 audit in the working research repository. The full integrated manifest and Step 2 feature tables are retained to preserve source provenance. The Step 3 runner constructs its model matrices only from (a) all 0.1/1 Ω training cases and (b) new34 10 Ω cases, and then only from the fixed three classifier WMUs per network. The 360 representative-bus 10 Ω feature rows are excluded from matrices, selection, prediction, and scoring.

Files:

- `integrated_manifest_4752_final.csv` — 4,752-case provenance/target manifest. Its SHA-256 is `6188e07acb7c1d680cc933c6587c841771d5c9c00e98987100c88c2c1fcb2032`.
- `features_manifest_reference.csv.gz`, `features_detected.csv.gz` — frozen Step 2 per-bus C1–C4 features. `detected` is the actual D1 waveform-only onset arm; manifest timing appears only in the explicitly labelled reference arm.
- `onset_predictions.csv` — raw-only detector status and post-hoc diagnostics, including the runaway flags.
- `STEP2_VERIFICATION_FINAL.json`, `STEP2_REPORT.md` — upstream validation and feature definitions.
- `legacy_final_summary_table.csv` — prior result rows verifying the fixed placements; the repository-native `../../../data/reference/placements/wmu_count_comparison_*.csv` tables are the primary placement evidence.

No raw waveform corpus or Simulink models are included here. Every Step 2 table file has an exact SHA-256 recorded in `SHA256SUMS.csv` at this package root.
