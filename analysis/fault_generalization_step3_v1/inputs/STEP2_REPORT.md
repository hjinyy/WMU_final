# Step 2 — 고장시점 검출 및 특징 추출 감사 보고서

**작성:** 2026-11-23  
**상태:** 추출·ML·최종 감사 완료 (`PASS_WITH_REVIEW_FLAGS`)  
**결과 경로:** `results/fault_generalization_step2_v1/full_corpus_v2/`

## 1. 목적과 비교 설계

실제 WMU 파형만을 입력으로 onset을 검출하고, 검출 기준 window와 manifest-reference window에서 같은 파형으로 C1–C4 feature를 계산했다. FaultType 및 FaultBusLabel을 대상으로 네 feature set, 두 window condition, 두 네트워크를 같은 classifier와 고정 CaseID-grouped folds로 비교했다. Manifest event timing은 검출기에 전달하지 않았다.

| Arm | 입력 특징 | 특징 수 / 측정 버스 | 역할 |
|---|---|---:|---|
| C1 | modified PMU-like sequence phasors | 6 | 위상자 기반 비교군 |
| C2 | WMU Baseline | 40 | 기존 baseline feature 정의 |
| C3 | C2 + NormVI | 44 | 제안 방식 |
| C4 | C3 + time-domain features | 56 | 시간영역 추가 실험 |

검출된 window와 manifest-reference window 모두 사전 고정 조건인 100 ms pre-event + 3 cycles at 50 Hz + 100 ms post-event를 사용했다. Event endpoint는 baseline 호환을 위해 inclusive 처리했다. C2–C4의 `sso_frequency_energy`는 기존 baseline 재현을 위해 manifest의 `SSOFrequencyHz`를 **주파수 band selector로만** 사용한다. 그 값은 검출기나 numeric ML input에는 넣지 않았다.

## 2. 데이터 및 산출물 무결성

- 총 4,752건: IEEE14 1,512건(버스 라벨 14개), IEEE30 3,240건(버스 라벨 30개). 각 fault-bus label별 108건이다.
- Raw processing errors: **0**. 원본 raw CSV와 manifest는 읽기 전용으로 사용했고, 기존 Step 1 산출물은 덮어쓰지 않았다.
- Manifest SHA-256: `6188e07acb7c1d680cc933c6587c841771d5c9c00e98987100c88c2c1fcb2032`.
- Manifest-reference features: 118,368 bus rows, 4,752건 모두 완전.
- Detected-window features: 117,998 bus rows, 4,729건 모두 완전.
- 각 feature table은 62개 waveform feature의 union을 보관한다. CaseID별 모든 네트워크 버스 행이 완결됐고 값은 finite임을 검증했다.
- Step 1과 일치하는 `|V| > 1e6` raw voltage flags 18건을 보존하고 표시했다(IEEE30). Primary ML cohort에는 포함했고, 별도 sensitivity cohort에서는 제외했다.

## 3. Onset 검출 및 window coverage

D1 detector는 각 케이스의 raw aligned V/I 파형만 받으며 측정 bus label 6, 1, 2의 관측 채널을 사용한다. 현재 구현은 full-record per-channel median scaling 및 centered score를 쓰므로 **offline timestamp**다. Centered kernel의 look-ahead는 7.5 ms이며, causal decision availability/real-time alarm latency는 주장하지 않는다.

- Detector status: 4,730/4,752 detected (**99.537%**); 22 no-event detections.
- Valid detected feature windows: 4,729/4,752 (**99.516%**). 추가 1건은 detector status가 DETECTED였지만 window가 bounds/grid 검증을 통과하지 못했다.
- 검출 4,730건 중 4,729건(99.979%)의 post-hoc absolute onset error는 10 ms 이내, 359건(7.590%)은 5 ms 이내였다. Median absolute error 6.90 ms.
- IEEE14: 1,492/1,512 detected/valid(98.677%), 20 미검출; 검출건 median absolute error 6.45 ms, 최대 7.35 ms.
- IEEE30: 3,238/3,240 detected(99.938%), 3,237 valid(99.907%), 2 미검출. 18 voltage-runaway 사례는 모두 표시했다.
- 주요 검출 이상: UnifiedCaseID 3044 (IEEE30, SLG, bus label 13)의 detected onset은 0.35875 s, manifest onset은 0.30000 s로 **+58.75 ms** 늦었다. 이 window는 무효 처리해 detected-window ML에서 제외했고 manifest-reference arm 및 all-case oracle에는 보존했다.
- 미검출 22건은 IEEE14 20건( SLG 9, LL 7, LLG 4 )과 IEEE30 SLG 2건이다. 세 데이터셋 arm 모두 고장 사례만 포함하므로 false-alarm rate는 추정할 수 없다.

검출 timestamps의 median signed error는 음수 방향이며, centered score의 look-ahead와 부합한다. 이 수치를 실제 경보 전달 지연으로 해석하면 안 된다.

## 4. ML 프로토콜 및 누수 점검

- Classifier: `ExtraTreesClassifier(n_estimators=120, class_weight='balanced', n_jobs=-1)`, random state 42.
- Network별 별도 모델. FaultType(4 classes)과 FaultBusLabel(IEEE14 14 classes / IEEE30 30 classes) 두 target을 평가했다.
- 5-fold `StratifiedGroupKFold`; group은 `UnifiedCaseID`. Network/target당 fold assignment를 한 번 만든 뒤 C1–C4와 모든 window scope에서 재사용했다.
- 64 metric groups, 320 fold rows, 9,504 fold assignments, 165,184 OOF predictions, 37,832 same-case paired correctness rows를 생성했다.
- Feature-name leakage audit 16/16 PASS. Fault/event labels, case IDs, paths, event timing, SSO frequency/magnitude, fault resistance, inception angle 등 metadata는 model input columns에서 제외했다. Train/test CaseID overlap 0이며 C1–C4가 동일 split hashes를 재사용했음을 재검증했다.
- 모든 bus sensor의 feature vector를 한 case-level row로 펼쳤다. FaultBusLabel은 model fault-block label로 다뤘으며, 이를 verified physical topology localization으로 해석하지 않는다.

### 핵심 pooled OOF accuracy (같은 검출 유효 사례로 paired 비교)

| 네트워크 / target | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| IEEE14 / FaultType, manifest → detected | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| IEEE14 / FaultBusLabel, manifest → detected | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 1.000 |
| IEEE30 / FaultType, manifest → detected | 1.000 → 1.000 | 0.999691 → 1.000 | 1.000 → 1.000 | 0.999691 → 0.999691 |
| IEEE30 / FaultBusLabel, manifest → detected | 1.000 → 1.000 | 1.000 → 0.999691 | 1.000 → 0.999691 | 1.000 → 1.000 |

비교는 manifest-reference와 detected windows가 **동일한 유효 사례·fold**를 쓴 paired 결과다. IEEE14은 모든 arm/target에서 점수가 같았다. IEEE30도 대부분 포화 수준이며, 37,832 paired rows 중 37,827은 correctness가 같고 5 rows에서만 달랐다(2 detected-only correct, 3 manifest-only correct). 이 5 rows는 3개 고유 사례(IDs 4433, 4435, 4245)에 해당하며 모두 non-runaway, onset absolute error 약 6.9–7.2 ms였다. 따라서 검출 window의 작은 이동이 일부 사례를 바꿨지만 평균 영향은 매우 작았고, 0–10 ms 범위 밖의 유효 window 효과는 평가할 수 없다.

Timing-error bins에서도 유효 detected windows는 전부 ≤10 ms에 있었다. IEEE14의 ≤5 ms 159건 및 >5–10 ms 1,333건은 모든 arm/target에서 paired accuracy 1.0이었다. IEEE30의 ≤5 ms 200건도 1.0이었다. >5–10 ms 3,037건에서는 일부 C2/C3/C4 비교가 1건 차이(약 0.000329)였고, 방향은 target/feature set에 따라 달랐다. No-valid-window는 IEEE14 20건, IEEE30 3건으로, detected accuracy는 정의하지 않았다.

### Runaway 제외 sensitivity

IEEE30의 18 flagged records를 제외한 paired cohort는 3,219건이다. FaultType은 네 feature set 모두 manifest/detected accuracy 1.0. FaultBusLabel도 C1/C3/C4는 1.0이고, C2 detected는 0.999689(1건 차이)였다. 따라서 flagged runaway가 전체 결과의 포화를 유일하게 설명하지는 않는다. 그러나 이것은 별도 물리검증이 아니며, 18건은 원인 규명 전 정상 파형으로 승인하지 않는다.

## 5. 해석 및 한계

1. 이 조건에서는 C1도 이미 천장에 가까워 NormVI(C3)나 시간영역 추가(C4)의 일관된 개선을 입증하지 못했다. “제안 특징이 일반화 성능을 개선했다”는 결론은 지지되지 않는다.
2. 5-fold split은 case-level 누수를 막지만 같은 저항·각도·배경/실험 설계가 train/test 양쪽에 남는 **in-corpus random CV**다. Held-out resistance/bus/operating-condition generalization은 평가하지 않았다. 매우 높은 점수는 이 제한된 조건의 결과로만 해석한다.
3. Manifest-reference는 알려진 onset을 이용한 상한 참고 arm이지 운영 성능이 아니다. 실제 detector arm에는 23건의 valid-window abstention이 있다.
4. 고장 사례만 있어 normal/no-fault false-alarm 성능은 측정할 수 없다.
5. IEEE30 Bus 30 PCC 연결과 current sensor direction이 별도 확인되지 않았으므로 topology, power-flow 방향, 물리적 위치 추론은 하지 않는다.
6. `SSOFrequencyHz`는 C2–C4의 legacy SSO spectral-band selector에만 쓰였으므로, 해당 baseline feature computation이 완전한 metadata-independent pipeline은 아니다. Onset detector 자체에는 사용되지 않았다.

## 6. 산출물과 검증 파일

- `onset_predictions.csv`, `onset_summary_by_condition.csv`
- `window_feature_status.csv`
- `features_manifest_reference.csv.gz`, `features_detected.csv.gz`
- `ml_evaluation_v1/metrics.csv`, `fold_metrics.csv`, `fold_assignments.csv`, `fold_split_audit.csv`
- `ml_evaluation_v1/predictions_oof.csv`, `paired_case_correctness.csv`, `timing_error_performance.csv`
- `ml_evaluation_v1/feature_leakage_audit.csv`, `model_feature_columns.csv`, `run_metadata.json`
- `STEP2_VERIFICATION_PREML.json`, `STEP2_VERIFICATION_FINAL.json`
- `STEP2_ARTIFACT_SHA256_PREML.json`, `STEP2_ARTIFACT_SHA256_FINAL.json`
- `requirements-step2.lock`

최종 verifier 상태는 `PASS_WITH_REVIEW_FLAGS`; ML audit 상태는 `PASS`다. Extractor, detector, evaluator, manifest, feature tables 및 ML tables의 SHA-256은 `STEP2_ARTIFACT_SHA256_FINAL.json`에 기록했다. Tests: **12 passed**; Python compilation 및 `git diff --check` 통과.
