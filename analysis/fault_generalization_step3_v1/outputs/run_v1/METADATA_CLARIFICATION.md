# Run metadata clarification

The preserved source execution wrote `representative_10ohm_holdout_cases_not_read: 360` in `run_metadata.json`. More precisely, the complete integrated manifest was loaded for cohort/count checks, but those 360 representative-bus 10 Ω rows were excluded from the feature matrices, predictions, and scored test set. The Step 3 runner also asserts that no representative 10 Ω rows enter either feature-mode matrix. This note clarifies the field wording; it does not alter the original execution log or results.
