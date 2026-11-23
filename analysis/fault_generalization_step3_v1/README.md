# Step 3 — 10 Ω 저항 미관측 일반화 평가 (신규 34개 모선)

**상태: 분석 완료, 검토 주의점 포함.** 이 결과는 시뮬레이션·물리 모델 검증이 아니라, 보존된 Step 1/2 파형 특징을 이용한 ML 일반화 평가입니다. 기존 대표 모선의 10 Ω 사례는 이번 평가에서 예측·점수화하지 않았습니다.

## 목적과 고정 조건

- 학습 및 모델 선택: `FaultResistanceOhm ∈ {0.1, 1.0}`만 사용.
- 최종 평가: `FaultResistanceOhm = 10.0`이며 `SourceDataset=new`인 신규 34개 모선만 사용.
  - IEEE14: 신규 9개 모선, 324건.
  - IEEE30: 신규 25개 모선, 900건.
  - 총 최종 평가 대상: 1,224건.
- 10 Ω 대표 모선 360건은 과거 평가에서 이미 반복 사용되었으므로 이번 결과의 학습/모델 선택/점수화에서 제외했습니다. 원본 기준 산출물은 변경하지 않았습니다.
- 학습 사례는 IEEE14 1,008건, IEEE30 2,160건(총 3,168건)입니다.
- 기존 위치 선정 기록을 고정했습니다: IEEE14 `[2, 6, 9]`, IEEE30 `[6, 1, 2]`. 근거는 저장소의 `data/reference/placements/wmu_count_comparison_ieee14.csv`, `...ieee30.csv`이며, 각 파일의 localization 목적 `k=3` 행과 일치합니다.
- 위치 추정(FaultBusLabel)과 4종 고장 유형(FaultType: SLG, LL, LLG, ThreePhase)은 **별도 분류기**로 학습·평가했습니다. 토폴로지 기반 거리 점수는 계산하지 않았습니다.

## 입력 특징과 선택 절차

고정 WMU 세 곳의 각 특징 벡터를 결합했습니다. C1–C4는 Step 2 정의(센서당 각각 6/40/44/56개 특징)를 유지했습니다. C3/C4의 NormVI 네 특징은 각 사례에서 **선택된 세 WMU만으로** 분모를 다시 계산했습니다. 저항, 모선 정답, 고장유형 정답, CaseID, 시간 메타데이터는 입력 특징에서 제외했습니다. SSO 주파수는 기존 spectral-band feature 계산의 선택자로만 사용했고 숫자형 예측변수로 넣지 않았습니다.

- ExtraTrees: 120 trees, `random_state=42`, `class_weight=balanced`.
- RandomForest: 120 trees, `random_state=42`, `class_weight=balanced_subsample`.
- 각 네트워크·타깃·윈도 모드별로 5-fold StratifiedGroupKFold를 학습 저항 데이터에서만 수행했습니다. 후보는 두 모델 × C1–C4이며, Macro-F1 평균 → 정확도 평균 → 적은 특징 수 → ExtraTrees → C1–C4 순으로 선택했습니다.
- 선택된 후보는 모든 네트워크/타깃에서 ExtraTrees+C1이었습니다. 일부 학습 CV 점수는 1.0으로 포화되어 모델 간 차이를 구분하지 못했으며, 이 선택은 성능 우월성의 증거가 아니라 사전 tie-break 결과입니다.
- 저항쌍 누수 가능성을 점검하기 위해 (고장 모선, 유형, 각도, 배경)으로 0.1/1 Ω 짝을 같은 fold에 두는 **훈련 전용 조건 그룹 CV 민감도 분석**도 별도로 실행했습니다. 모델/특징 선택은 동일하게 ExtraTrees+C1이었고, 이 분석도 10 Ω 자료를 읽거나 선택에 사용하지 않았습니다.

## 신규 34개 모선 — 실제 검출 윈도우 결과

`detected` 윈도우를 주 결과로 보고합니다. Accuracy/Macro-F1은 유효 윈도우가 있는 사례에 대한 조건부 점수이며, End-to-end 정답률은 검출 실패·유효하지 않은 윈도우를 오답 처리한 비율입니다.

### 훈련 CV만으로 선택된 후보: ExtraTrees + C1

- **IEEE14, 신규 9개 모선 (324건):** 유효 윈도우 304건(93.83%).
  - 유형: 조건부 Accuracy 26.64%, Macro-F1 10.52%, end-to-end 정답률 25.00%.
  - 위치: 조건부 정확도 10.86%, Macro-F1 5.10%, end-to-end 정답률 10.19%.
- **IEEE30, 신규 25개 모선 (900건):** 유효 윈도우 897건(99.67%).
  - 유형: 조건부 Accuracy 25.08%, Macro-F1 10.03%, end-to-end 정답률 25.00%.
  - 위치: 조건부 정확도 3.68%, Macro-F1 0.24%, end-to-end 정답률 3.67%.

선택된 C1 모델은 두 네트워크에서 유효 사례의 FaultType을 전부 `ThreePhase`로 예측했습니다. IEEE30 위치 모델은 유효 897건을 모두 Bus 13으로 예측했습니다. 따라서 10 Ω에서는 학습 CV와 최종 일반화가 크게 달랐습니다. 이 불리한 결과를 그대로 보존했습니다.

### 사전 지정된 대안 비교(모델 선택에는 사용하지 않음)

다음은 동일한 신규 10 Ω 검증집합에서의 `detected` 윈도우 점수입니다. 최종 평가 점수를 보고 후보를 고르지 않았으며, 전체 64개 후보의 결과는 `outputs/run_v1/tables/new34_10ohm_test_metrics.csv`에 저장했습니다. 아래는 WMU Baseline(C2)과 Baseline+NormVI(C3)의 **조건부 Accuracy / Macro-F1 / end-to-end 정답률**입니다.

- **IEEE14 유형:** C2 ExtraTrees `67.1 / 58.4 / 63.0%`, RandomForest `56.6 / 46.9 / 53.1%`; C3 ExtraTrees `52.0 / 40.6 / 48.8%`, RandomForest `56.9 / 45.3 / 53.4%`.
- **IEEE14 위치:** C2 ExtraTrees `13.8 / 3.0 / 13.0%`, RandomForest `12.2 / 1.9 / 11.4%`; C3 ExtraTrees `69.4 / 42.5 / 65.1%`, RandomForest `71.4 / 47.0 / 67.0%`.
- **IEEE30 유형:** C2 ExtraTrees `64.0 / 55.9 / 63.8%`, RandomForest `28.0 / 15.7 / 27.9%`; C3 ExtraTrees `64.0 / 55.6 / 63.8%`, RandomForest `27.3 / 14.5 / 27.2%`.
- **IEEE30 위치:** C2 ExtraTrees `34.4 / 30.4 / 34.3%`, RandomForest `28.0 / 23.4 / 27.9%`; C3 ExtraTrees `42.8 / 34.0 / 42.7%`, RandomForest `41.8 / 34.6 / 41.7%`.

C2/C3가 선택된 C1보다 나은 일부 결과가 있지만, 이는 최종 검증 점수에서 관찰된 대안 비교일 뿐 새 모델 선택 근거로 사용하지 않았습니다. 따라서 해당 비교 점수에 맞춘 재학습·재선정은 수행하지 않았습니다.

## 검출 오차·윈도 영향

- 신규 34개 모선 10 Ω 대상 1,224건 중 Step 2 raw-only D1 검출에서 유효 detected window는 1,201건(98.12%)입니다. manifest-timed reference arm은 1,224건 모두 존재하며 상한선 비교용으로 별도 저장했습니다.
- 유효하지 않은 23건은 IEEE14 Bus 1에서 검출 실패 20건, IEEE30 Bus 13에서 검출 실패 2건 및 검출 지연 1건(오차 +58.75 ms)입니다. 실제 검출 arm의 분류 조건부 점수와 end-to-end 점수를 모두 함께 보고했습니다.
- 대표 모선 10 Ω 360건은 이번 예측 파일에 포함되지 않습니다.

## 이상치 민감도·한계

- Step 1에서 runaway로 표시된 18건은 주 분석 학습에 포함하고 제거하지 않았습니다. 별도 민감도에서는 이 18건을 **학습에서만** 제외해 선택된 후보를 재적합했습니다. 신규 10 Ω 결과는 동일했습니다. 원본 사례 및 플래그는 보존했습니다.
- 현재 Step 2 onset detector는 원시 파형에서 모선 `[6,1,2]`를 사용합니다. IEEE14 분류기 특징 입력은 고정 배치 `[2,6,9]`이므로, detector 입력까지 포함한 물리적 센서 수를 정확히 3대로 제한하려는 해석과는 일치하지 않습니다. 본 Step 3의 “고정 3-WMU”는 **분류기 특징 입력**에 적용했습니다. 검출기까지 같은 세 센서로 제한하는 end-to-end 결과로 확대 해석하면 안 됩니다.
- 학습 fold는 저항별 고유 `UnifiedCaseID`로 그룹화했습니다. 조건 그룹 CV 민감도에서 선택은 동일했지만, 일반화 결론은 CV가 아니라 held-out 10 Ω 신규34 결과를 기준으로 합니다.
- no-fault 사례가 없어 false-alarm rate는 측정하지 않았습니다. 실제 Simulink 연결·센서 방향·물리적 토폴로지는 검증하지 않았으므로 물리적 위치/전력흐름 주장을 하지 않습니다.

### 재현 환경

검증한 환경은 Python 3.11.15, NumPy 2.4.6, scikit-learn 1.9.1입니다. 저장소 루트에서 `python -m pip install -r analysis/fault_generalization_step3_v1/requirements.txt`로 필요한 패키지를 설치할 수 있습니다. 테스트와 결과 생성 명령은 저장소 루트 README의 Step 3 section을 참고하세요.

## 산출물과 재현 정보

- 실행 산출물: `outputs/run_v1/`
- 조건 그룹 훈련 CV 민감도 기준 산출물: `outputs/train_only_condition_grouped_v1/`
- 조건 그룹 CV portable 재실행 결과: `outputs/train_only_condition_grouped_reproduction_v1/` (CV score CSV가 기준 산출물과 바이트 단위로 일치)
- 실행 스크립트 및 테스트: `scripts/`, `tests/`
- 원 실행 코드/결과 snapshot: `source_snapshot/` (이 snapshot의 `run_metadata.json`은 원 실행 source SHA와 연결됩니다.)
- portable package runner와 테스트는 저장소 내 입력·reference placement만 사용합니다. 코드의 경로/schema packaging adaptation은 모델·split·feature·metric 계산을 바꾸지 않았으며, 새 위치 source로 확인 후 canonical 결과와 계산 결과를 대조했습니다.
- 후보별 예측: `outputs/run_v1/predictions/new34_10ohm_test_predictions.csv`
- Fold별 OOF 예측: `outputs/run_v1/predictions/train_cv_oof_predictions.csv`
- 입력/실행 해시: `inputs/` 및 `outputs/run_v1/run_metadata.json`
- 대표 모선 기존 결과 참고: 저장소의 기존 `outputs/paper_figures_v1/` 및 `data/frozen_inputs/` 자료. 이번 Step 3 출력으로 덮어쓰지 않았습니다.

전체 지표·confusion matrix·모선별/유형별 분해는 CSV로 보존했습니다. 저장소 내 portable runner를 별도 scratch 출력에 재실행해 최종 지표·예측·OOF·confusion/per-bus 결과 9개 CSV의 SHA-256이 canonical `run_v1`과 일치하는지 확인했습니다. 조건 그룹 훈련 CV도 별도로 재실행해 결과 CSV가 원본과 바이트 단위로 일치했습니다. 이 재현 확인은 같은 10 Ω 신규34 사례를 다시 계산한 것이며 독립된 두 번째 test set이 아닙니다. 이 패키지는 10 Ω 신규 모선에 대한 부정적 일반화 결과도 포함한 감사 가능한 분석이며, 물리 검증 결과는 아닙니다.
