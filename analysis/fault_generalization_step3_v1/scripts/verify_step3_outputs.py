#!/usr/bin/env python3
"""Fail-closed audit of packaged Step 3 inputs, splits, scores, and hashes."""
from pathlib import Path
import csv
import hashlib
import json
import math

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
INPUTS = PACKAGE / "inputs"
RUN = PACKAGE / "outputs/run_v1"
NETWORKS = {"ieee14": 14, "ieee30": 30}
TARGETS = ("FaultType", "FaultBusLabel")
MODES = ("manifest_reference", "detected")
MODELS = ("ExtraTrees", "RandomForest")
FEATURES = (
    "C1_modified_PMU_like",
    "C2_WMU_Baseline",
    "C3_Baseline_plus_NormVI",
    "C4_C3_plus_time_domain",
)


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, msg):
    if not ok:
        raise SystemExit("FAIL: " + msg)


def f1_macro(records, labels):
    scores = []
    for label in labels:
        tp = sum(r["TruthLabel"] == label and r["PredictedLabel"] == label for r in records)
        fp = sum(r["TruthLabel"] != label and r["PredictedLabel"] == label for r in records)
        fn = sum(r["TruthLabel"] == label and r["PredictedLabel"] != label for r in records)
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return sum(scores) / len(scores) if scores else 0.0


def pct(value, digits=2):
    return f"{100 * float(value):.{digits}f}"


def verify_report(report_path, metrics, predictions, onset_rows):
    report = Path(report_path).read_text(encoding="utf-8")
    indexed = {(r["NetworkID"], r["Target"], r["WindowMode"], r["Model"], r["FeatureSet"]): r for r in metrics}
    for net in NETWORKS:
        for target, label in (("FaultType", "유형"), ("FaultBusLabel", "위치")):
            r = indexed[(net, target, "detected", "ExtraTrees", FEATURES[0])]
            metric_label = "Accuracy" if target == "FaultType" else "정확도"
            expected = f"  - {label}: 조건부 {metric_label} {pct(r['ConditionalAccuracy'])}%, Macro-F1 {pct(r['ConditionalMacroF1'])}%, end-to-end 정답률 {pct(r['EndToEndCorrectRate'])}%."
            require(expected in report, f"README selected-model metric mismatch: {net}/{target}")
            pieces = []
            for feature in (FEATURES[1], FEATURES[2]):
                vals = []
                for model in MODELS:
                    alt = indexed[(net, target, "detected", model, feature)]
                    vals.append(f"{model} `{pct(alt['ConditionalAccuracy'], 1)} / {pct(alt['ConditionalMacroF1'], 1)} / {pct(alt['EndToEndCorrectRate'], 1)}%`")
                prefix = "C2" if feature == FEATURES[1] else "C3"
                pieces.append(f"{prefix} {vals[0]}, {vals[1]}")
            expected_alt = f"- **{net.upper()} {label}:** " + "; ".join(pieces) + "."
            require(expected_alt in report, f"README alternative-model metric mismatch: {net}/{target}")

    selected = [r for r in predictions if r["Target"] == "FaultType" and r["WindowMode"] == "detected" and r["Model"] == "ExtraTrees" and r["FeatureSet"] == FEATURES[0] and r["ValidFeatureWindow"] == "False"]
    bad_cases = {int(r["UnifiedCaseID"]): r for r in selected}
    require(len(bad_cases) == 23, "invalid detected-window case count differs from the report")
    from collections import Counter
    by_bus = Counter((r["NetworkID"], r["FaultBusLabel"]) for r in bad_cases.values())
    require(by_bus == Counter({("ieee14", "1"): 20, ("ieee30", "13"): 3}), "invalid-window bus breakdown differs from the report")
    onset = {int(r["UnifiedCaseID"]): r for r in onset_rows}
    statuses = Counter(onset[u]["DetectorStatus"] for u in bad_cases)
    require(statuses == Counter({"NO_EVENT_DETECTED": 22, "DETECTED": 1}), "invalid-window detector-status breakdown differs from the report")
    late = [onset[u] for u in bad_cases if onset[u]["DetectorStatus"] == "DETECTED"]
    require(len(late) == 1 and math.isclose(float(late[0]["OnsetAbsErrorMs_posthoc"]), 58.75, abs_tol=1e-9), "late-detection error differs from the report")
    require("유효하지 않은 23건은 IEEE14 Bus 1에서 검출 실패 20건, IEEE30 Bus 13에서 검출 실패 2건 및 검출 지연 1건(오차 +58.75 ms)입니다." in report, "README detector-miss narrative mismatch")


def main():
    manifest_rows = rows(INPUTS / "integrated_manifest_4752_final.csv")
    manifest = {int(r["UnifiedCaseID"]): r for r in manifest_rows}
    require(len(manifest_rows) == len(manifest) == 4752, "manifest must contain 4,752 unique cases")
    train = {u for u, r in manifest.items() if float(r["FaultResistanceOhm"]) in (0.1, 1.0)}
    new_test = {u for u, r in manifest.items() if float(r["FaultResistanceOhm"]) == 10 and r["SourceDataset"] == "new"}
    representative = {u for u, r in manifest.items() if float(r["FaultResistanceOhm"]) == 10 and r["SourceDataset"] == "baseline"}
    require((len(train), len(new_test), len(representative)) == (3168, 1224, 360), "resistance/cohort split counts mismatch")

    placements = {"ieee14": [2, 6, 9], "ieee30": [6, 1, 2]}
    for net, buses in placements.items():
        p = REPO / f"data/reference/placements/wmu_count_comparison_{net}.csv"
        evidence = [r for r in rows(p) if r["PlacementObjective"] == "localization" and int(r["k"]) == 3]
        require(len(evidence) == 1, f"{net}: expected one fixed localization k=3 row")
        require([int(x) for x in evidence[0]["SelectedWMUBuses"].split(";")] == buses, f"{net}: fixed placement mismatch")

    step2 = json.loads((INPUTS / "STEP2_VERIFICATION_FINAL.json").read_text())
    require(step2["status"] == "PASS_WITH_REVIEW_FLAGS" and step2["ml_evaluation"]["status"] == "PASS", "upstream Step 2 status invalid")
    runmeta = json.loads((RUN / "run_metadata.json").read_text())
    require(runmeta["status"] == "COMPLETE", "canonical Step 3 run metadata is incomplete")
    require(runmeta["integrated_manifest_sha256"] == sha(INPUTS / "integrated_manifest_4752_final.csv"), "manifest hash mismatch in run metadata")
    require(runmeta["step2_detected_features_sha256"] == sha(INPUTS / "features_detected.csv.gz"), "detected-feature hash mismatch")
    require(runmeta["step2_manifest_reference_features_sha256"] == sha(INPUTS / "features_manifest_reference.csv.gz"), "reference-feature hash mismatch")
    require(runmeta["step2_onset_predictions_sha256"] == sha(INPUTS / "onset_predictions.csv"), "onset input hash mismatch")
    source_meta = PACKAGE / "source_snapshot/run_v1/run_metadata.json"
    source_script = PACKAGE / "source_snapshot/step3_scripts/evaluate_step3_unseen10_fixed3wmu.py"
    require(runmeta == json.loads(source_meta.read_text()), "run metadata differs from preserved original snapshot")
    require(runmeta["step3_evaluator_sha256"] == sha(source_script), "original evaluator/source snapshot hash mismatch")

    cv = rows(RUN / "tables/train_only_cv.csv")
    selected = rows(RUN / "tables/model_selection.csv")
    folds = rows(RUN / "splits/train_cv_fold_assignments.csv")
    oof = rows(RUN / "predictions/train_cv_oof_predictions.csv")
    metrics = rows(RUN / "tables/new34_10ohm_test_metrics.csv")
    predictions = rows(RUN / "predictions/new34_10ohm_test_predictions.csv")
    require((len(cv), len(selected), len(folds), len(oof), len(metrics), len(predictions)) == (64, 8, 12672, 101376, 64, 39168), "Step 3 artifact row counts mismatch")

    # Every fold-assignment group covers exactly its network's 0.1/1-ohm training cases.
    for net in NETWORKS:
        net_train = {u for u in train if manifest[u]["NetworkID"] == net}
        require(len(net_train) == (1008 if net == "ieee14" else 2160), f"{net}: train split count mismatch")
        for target in TARGETS:
            for mode in MODES:
                group = [r for r in folds if r["NetworkID"] == net and r["Target"] == target and r["WindowMode"] == mode]
                ids = [int(r["UnifiedCaseID"]) for r in group]
                require(len(group) == len(net_train) and set(ids) == net_train and len(set(ids)) == len(ids), f"{net}/{target}/{mode}: fold assignment coverage/leakage error")
                require({int(r["Fold"]) for r in group} == {1, 2, 3, 4, 5}, f"{net}/{target}/{mode}: fold IDs mismatch")
                for r in group:
                    source = manifest[int(r["UnifiedCaseID"])]
                    truth = source["FaultType"] if target == "FaultType" else str(int(source["FaultBus"]))
                    require(r["TruthLabel"] == truth, f"{net}/{target}: fold truth label mismatch")

    require(all(r["SelectedModel"] == "ExtraTrees" and r["SelectedFeatureSet"] == FEATURES[0] for r in selected), "train-CV selected model/feature mismatch")
    require({int(r["UnifiedCaseID"]) for r in oof} == train, "OOF predictions are not restricted to training resistances")
    require(not ({int(r["UnifiedCaseID"]) for r in oof} & new_test), "train OOF overlaps new34 final test")
    require(not ({int(r["UnifiedCaseID"]) for r in predictions} & representative), "representative 10-ohm cases were scored")
    require(all(r["Cohort"] == "new34" and float(r["FaultResistanceOhm"]) == 10.0 for r in predictions), "test predictions include wrong cohort/resistance")

    grouped = {}
    for r in predictions:
        key = (r["NetworkID"], r["Target"], r["WindowMode"], r["Model"], r["FeatureSet"])
        grouped.setdefault(key, []).append(r)
    require(len(grouped) == 64, "unexpected test candidate group count")
    for (net, target, mode, model, feature), group in grouped.items():
        total = 324 if net == "ieee14" else 900
        valid = [r for r in group if r["ValidFeatureWindow"] == "True"]
        expected_valid = ({"ieee14": 304, "ieee30": 897}[net] if mode == "detected" else total)
        require(len(group) == total and len(valid) == expected_valid, f"test denominator/coverage mismatch: {(net, target, mode, model, feature)}")
        require(all(r["Cohort"] == "new34" and r["FaultBusLabel"] and r["TruthLabel"] for r in group), "test prediction truth/cohort fields missing")

    metric_index = {}
    for r in metrics:
        key = (r["NetworkID"], r["Target"], r["WindowMode"], r["Model"], r["FeatureSet"])
        metric_index[key] = r
        group = grouped[key]
        valid = [x for x in group if x["ValidFeatureWindow"] == "True"]
        correct = sum(x["TruthLabel"] == x["PredictedLabel"] for x in valid)
        total = len(group)
        require(int(r["TestCasesTotal"]) == total and int(r["TestCasesWithValidWindow"]) == len(valid), f"metric denominators mismatch: {key}")
        require(math.isclose(float(r["WindowCoverage"]), len(valid) / total, abs_tol=1e-12), f"coverage mismatch: {key}")
        require(math.isclose(float(r["ConditionalAccuracy"]), correct / len(valid), abs_tol=1e-12), f"conditional accuracy mismatch: {key}")
        classes = ("SLG", "LL", "LLG", "ThreePhase") if r["Target"] == "FaultType" else tuple(str(i) for i in range(1, NETWORKS[r["NetworkID"]] + 1))
        require(math.isclose(float(r["ConditionalMacroF1"]), f1_macro(valid, classes), abs_tol=1e-12), f"conditional macro-F1 mismatch: {key}")
        require(math.isclose(float(r["CorrectValidPredictions"]), correct, abs_tol=1e-12), f"correct count mismatch: {key}")
        require(math.isclose(float(r["EndToEndCorrectRate"]), correct / total, abs_tol=1e-12), f"end-to-end accuracy mismatch: {key}")

    leak = rows(RUN / "tables/feature_leakage_audit.csv")
    require(len(leak) == 16 and all(r["Status"] == "PASS" for r in leak), "feature leakage audit failed")
    sensitivity = rows(RUN / "tables/runaway_training_sensitivity.csv")
    require(len(sensitivity) == 8, "runaway sensitivity row count mismatch")
    require(sum(int(r["RunawayTrainCasesExcluded"]) for r in sensitivity if r["WindowMode"] == "detected") == 36, "runaway excluded-case count mismatch")
    for r in sensitivity:
        key = (r["NetworkID"], r["Target"], r["WindowMode"], r["SelectedModel"], r["SelectedFeatureSet"])
        base = metric_index[key]
        for field in ("ConditionalAccuracy", "ConditionalMacroF1", "EndToEndCorrectRate"):
            require(math.isclose(float(r[field]), float(base[field]), abs_tol=1e-12), f"runaway sensitivity changed held-out score: {key}/{field}")

    onset = rows(INPUTS / "onset_predictions.csv")
    runaway = {int(r["UnifiedCaseID"]) for r in onset if r["RawOnlyVoltageRunawayFlag_gt_1e6"].lower() == "true"}
    require(len(runaway) == 18 and runaway.issubset(train) and not (runaway & new_test), "runaway flags/cohort provenance mismatch")
    verify_report(PACKAGE / "README.md", metrics, predictions, onset)

    # Condition-grouped CV is train-only and has a byte-identical portable rerun.
    grouped_cv = rows(PACKAGE / "outputs/train_only_condition_grouped_v1/tables/train_only_condition_grouped_cv.csv")
    grouped_cv_rerun = PACKAGE / "outputs/train_only_condition_grouped_reproduction_v1/tables/train_only_condition_grouped_cv.csv"
    require(len(grouped_cv) == 64, "condition-grouped CV candidate count mismatch")
    require(sha(PACKAGE / "outputs/train_only_condition_grouped_v1/tables/train_only_condition_grouped_cv.csv") == sha(grouped_cv_rerun), "condition-grouped CV rerun hash mismatch")
    require(all(r["TrainCVRank"] == "1" and r["SelectedByConditionGroupedCV"] == "True" and r["Model"] == "ExtraTrees" and r["FeatureSet"] == FEATURES[0] for r in grouped_cv if r["SelectedByConditionGroupedCV"] == "True"), "condition-grouped selection differs")
    require(sum(r["SelectedByConditionGroupedCV"] == "True" for r in grouped_cv) == 8, "condition-grouped selection group count mismatch")

    # Every packaged file is covered by the non-recursive SHA inventory.
    sums = rows(PACKAGE / "SHA256SUMS.csv")
    listed = set()
    for r in sums:
        rel = Path(r["Path"])
        require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe inventory path {rel}")
        p = PACKAGE / rel
        require(p.is_file(), f"missing inventory path {rel}")
        require(p.stat().st_size == int(r["Bytes"]) and sha(p) == r["SHA256"], f"hash mismatch: {rel}")
        listed.add(rel.as_posix())
    actual = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*") if p.is_file() and p.name != "SHA256SUMS.csv" and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    generated = {name for name in actual - listed if name.startswith("outputs/reproduction_") or name.startswith("outputs/train_only_condition_grouped_reproduction_") and name != "outputs/train_only_condition_grouped_reproduction_v1"}
    require(listed == actual - generated, f"hash inventory coverage mismatch (missing={sorted(actual-listed-generated)}, extra={sorted(listed-actual)})")

    print(json.dumps({"status": "PASS", "manifest_cases": len(manifest), "train_cases": len(train), "new34_test_cases": len(new_test), "representative_10ohm_not_scored": len(representative), "cv_candidate_rows": len(cv), "fold_assignment_rows": len(folds), "oof_prediction_rows": len(oof), "test_metric_rows": len(metrics), "test_prediction_rows": len(predictions), "detected_valid_cases": {"ieee14": 304, "ieee30": 897}, "runaway_flags": len(runaway), "sha_inventory_entries": len(sums)}, indent=2))


if __name__ == "__main__":
    main()
