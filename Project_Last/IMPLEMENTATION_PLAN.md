# 삼성전자 익일 시가·종가 회귀 모델 구현 계획

## 1. 문서 목적

이 문서는 `PRD_NEXT_DAY_STOCK_REGRESSION.md`를 실제 코드와 산출물로 구현하기 위한 작업 순서와 완료 조건을 정의한다.

작업 범위는 정제된 삼성전자 일봉 데이터를 이용하여 다음 실제 거래일의 시가와 종가를 예측하는 회귀 모델을 완성하고, RSI와 MACD가 익일 종가 예측에 어떻게 반영되었는지 분석 문서로 남기는 것이다.

## 2. 현재 진행 상태

| 단계 | 상태 | 산출물 |
|---|---|---|
| 원본 데이터 검증 | 완료 | `reports/005930_data_quality.md` |
| 이상 행 분리 | 완료 | `reports/005930_excluded_rows.csv` |
| 숫자 형식 정규화 | 완료 | `prepare_stock_data.py` |
| 과거→미래 정렬 | 완료 | `data/processed/005930_2016to2026_clean.csv` |
| 변동률·RSI·MACD 재계산 | 완료 | 정제 CSV에 반영 |
| 파생 특성 및 목표 생성 | 완료 | Phase 2 |
| 기준 회귀 모델 | 완료 | Phase 3 |
| 후보 모델 비교 | 예정 | Phase 4 |
| 독립 테스트 평가 | 완료 | Phase 5 일부 |
| RSI·MACD 영향 분석 | 예정 | Phase 6 |

현재 정제 데이터는 2018-06-01 이후 1,961행이며 날짜 범위는 2018-06-01부터 2026-06-01까지다. 모델 데이터는 1,901행이며 32개 특성을 포함한다.

## 3. 구현 원칙

- 원본 CSV는 덮어쓰지 않는다.
- 2018-06-01 이전 데이터는 학습 및 예측 파이프라인에서 제외한다.
- 모든 학습 데이터는 날짜 오름차순으로 처리한다.
- 무작위 데이터 분할은 사용하지 않는다.
- 거래일 `t`의 입력에는 `t` 장 종료 시점까지 확정된 값만 사용한다.
- 목표값은 다음 실제 거래일 `t+1`에서 가져온다.
- 전처리기와 스케일러는 학습 구간에만 적합한다.
- 테스트 구간은 모델과 설정이 확정되기 전까지 모델 선택에 사용하지 않는다.
- 모델 성능은 반드시 Naive 기준 모델과 비교한다.
- 실행 설정과 랜덤 시드를 저장하여 결과를 재현할 수 있게 한다.

## 4. 목표 디렉터리 구조

```text
Project_Last/
├─ 005930_2016to2026.csv
├─ PRD_NEXT_DAY_STOCK_REGRESSION.md
├─ IMPLEMENTATION_PLAN.md
├─ prepare_stock_data.py
├─ requirements.txt
├─ config.yaml
├─ src/
│  ├─ feature_engineering.py
│  ├─ dataset.py
│  ├─ train_baseline.py
│  ├─ train_candidates.py
│  ├─ evaluate.py
│  ├─ predict.py
│  └─ analyze_indicators.py
├─ data/
│  └─ processed/
│     ├─ 005930_2016to2026_clean.csv
│     └─ 005930_model_dataset.csv
├─ models/
├─ reports/
│  ├─ metrics/
│  ├─ figures/
│  ├─ predictions/
│  └─ RSI_MACD_IMPACT_REPORT.md
└─ tests/
```

## 5. 단계별 구현 계획

### Phase 1. 데이터 검증 및 정제

상태: 완료

수행 내용:

- 날짜와 숫자 형식 정규화
- 주말 행 제거
- 거래량 결측 및 0 이하 행 제거
- 비정상 가격 단위 행 제거
- OHLC 논리 오류 행 제거
- 시간순 정렬
- 정제 이후 변동률, RSI 및 MACD 재계산
- 제외 행과 사유 기록

완료 조건:

- 정제 CSV가 날짜 오름차순이다.
- 날짜 중복, 주말 행 및 OHLC 오류가 없다.
- 원본 데이터가 변경되지 않는다.
- 제외된 모든 행에 사유가 기록된다.

### Phase 2. 파생 특성 및 목표값 생성

상태: 완료

구현 파일:

- `src/feature_engineering.py`
- `src/dataset.py`
- `tests/test_feature_engineering.py`

생성할 목표값:

- `target_next_open`
- `target_next_close`
- `target_next_open_return`
- `target_next_close_return`

수익률 정의:

```text
target_next_open_return  = next_open / current_close - 1
target_next_close_return = next_close / current_close - 1
```

생성할 주요 특성:

- 1·2·3·5·10·20일 수익률
- 5·10·20·60일 이동평균 대비 종가 비율
- 5·10·20일 이동 변동성
- 거래량 이동평균과 현재 거래량 비율
- 당일 시가 대비 종가 수익률
- 전일 종가 대비 당일 시가 갭
- 당일 고가와 저가의 범위
- 고가·저가 구간 내 종가 위치
- RSI 현재값과 1일 변화량
- MACD, Signal, Histogram 현재값과 1일 변화량
- 외국인소진율 현재값과 1일 변화량
- 요일 및 월

처리 규칙:

- 롤링 계산으로 발생한 초기 결측 행은 제거한다.
- 마지막 행은 목표값이 없으므로 최신 예측용으로 별도 보존한다.
- 가격 원값보다 수익률과 비율 특성을 우선 사용한다.
- 미래 데이터로 결측값을 채우지 않는다.

산출물:

- `data/processed/005930_model_dataset.csv`
- `reports/feature_summary.json`
- 최신 예측용 입력 행

완료 조건:

- 행 `t`의 목표값이 행 `t+1`의 실제 시가·종가와 일치한다.
- 입력 특성에 미래 행 정보가 포함되지 않는다.
- 특성명, 계산식, 결측 제거 행 수가 보고서에 기록된다.
- 관련 단위 테스트를 통과한다.

### Phase 3. 기준 모델 파이프라인

상태: 완료

구현 파일:

- `src/train_baseline.py`
- `src/evaluate.py`
- `tests/test_time_split.py`

데이터 분할:

- Train: 앞쪽 80%
- Validation: 다음 10%
- Test: 마지막 10%
- 구간 경계의 익일 목표값 누수를 방지하기 위해 Train과 Validation 마지막 행을 1개씩 제거

실제 분할 날짜와 행 수는 모델 데이터 생성 후 고정하여 `config.yaml`과 평가 보고서에 기록한다.

기준 모델:

1. Naive Open: 다음 시가를 현재 종가로 예측
2. Naive Close: 다음 종가를 현재 종가로 예측
3. 최근 N일 평균 수익률 모델
4. Ridge 회귀

학습 방식:

- 시가 수익률과 종가 수익률 모델을 각각 학습한다.
- Ridge에는 학습 구간에만 적합한 표준화를 적용한다.
- 랜덤 시드는 `42`를 기본값으로 고정한다.

평가 지표:

- 가격 MAE
- 가격 RMSE
- sMAPE
- 수익률 MAE
- 상승·하락 방향 정확도

산출물:

- 기준 모델 파일
- 전처리 파이프라인
- Validation 성능표
- 예측값과 실제값 비교 CSV

완료 조건:

- 한 명령으로 특성 생성부터 기준 모델 평가까지 실행된다.
- Naive 모델과 Ridge가 완전히 동일한 검증 행에서 비교된다.
- 데이터 분할 경계와 사용 특성이 저장된다.
- 테스트 구간은 모델 선택에 사용되지 않는다.

### Phase 4. 후보 모델 비교 및 선택

상태: 예정

구현 파일:

- `src/train_candidates.py`
- `tests/test_training_pipeline.py`

1차 후보 모델:

- Elastic Net
- Random Forest Regressor
- HistGradientBoostingRegressor

추가 의존성이 허용될 경우 검토:

- XGBoost
- LightGBM

검증 방법:

- 확장 윈도우 기반 `TimeSeriesSplit`
- 각 폴드에서 학습 기간은 과거에서 미래 방향으로 확장
- 모든 후보를 동일한 폴드와 지표로 비교
- 지나치게 큰 하이퍼파라미터 탐색은 피하고 제한된 후보군만 사용

선택 기준:

1. Validation 및 워크포워드 평균 MAE
2. 폴드별 성능 편차
3. Naive 모델 대비 개선 정도
4. 과적합 여부
5. 재현성과 추론 속도

산출물:

- `reports/model_comparison.csv`
- 폴드별 성능 기록
- 선택된 모델과 하이퍼파라미터
- 모델 선택 근거 문서

완료 조건:

- 독립 테스트 구간을 열어보기 전에 최종 모델이 결정된다.
- 시가 모델과 종가 모델을 각각 선택한다.
- 후보 모델의 실패 또는 제외 이유도 기록한다.

### Phase 5. 최종 테스트 평가 및 최신 예측

상태: 예정

구현 파일:

- `src/predict.py`
- `src/evaluate.py`

수행 내용:

- 확정된 모델과 설정으로 Train+Validation 구간 재학습
- 독립 Test 구간을 한 번 평가
- 예측 수익률을 원 단위 예상 가격으로 복원
- 실제값과 예측값의 시계열 및 잔차 분석
- 최신 정제 행을 기준으로 다음 거래일 시가·종가 예측

산출물:

- `models/final_open_model.*`
- `models/final_close_model.*`
- `reports/final_test_metrics.json`
- `reports/predictions/test_predictions.csv`
- `reports/predictions/latest_prediction.json`
- 실제값·예측값 그래프
- 잔차 그래프

최신 예측 결과에 포함할 항목:

- 데이터 기준일
- 예상 대상: 다음 실제 거래일
- 예상 시가
- 예상 종가
- 현재 종가 대비 예상 수익률
- 사용 모델과 버전
- 분석용 예측이며 투자 권유가 아니라는 설명

완료 조건:

- 최종 모델이 Naive 모델과 비교된다.
- 결과에 기준일과 가격 단위가 명확히 표시된다.
- 저장된 모델을 다시 불러와 같은 입력에 동일한 예측을 생성한다.

### Phase 6. RSI·MACD 영향 분석

상태: 예정

사용자 요청에 따라 모델 완성 후 별도 분석 문서를 작성한다.

구현 파일:

- `src/analyze_indicators.py`

분석 대상:

- RSI(14)
- RSI 1일 변화량
- MACD(12, 26)
- MACD Signal(9)
- MACD Histogram
- 각 MACD 지표의 1일 변화량

분석 방법:

- 선형 모델 계수와 계수 방향
- 트리 모델의 permutation importance
- 해당 특성을 제거한 모델과의 성능 비교
- RSI 구간별 실제 익일 종가 수익률 통계
- MACD Histogram 부호와 변화 방향별 익일 종가 수익률 통계
- 가능하면 부분 의존도 또는 SHAP 분석

주의 사항:

- 특성 중요도는 인과관계를 의미하지 않는다.
- RSI와 MACD는 가격으로부터 파생되므로 다른 가격 특성과 상관관계가 높을 수 있다.
- 단일 중요도 값만 사용하지 않고 제거 실험과 구간별 통계를 함께 제시한다.

산출물:

- `reports/RSI_MACD_IMPACT_REPORT.md`
- RSI 구간별 통계 CSV
- MACD 조건별 통계 CSV
- 특성 중요도 그래프
- 지표 제거 전후 성능 비교표

완료 조건:

- RSI와 MACD가 예측값에 미친 영향의 방향과 상대적 크기가 설명된다.
- 모델 성능에 실제로 기여했는지 제거 실험 결과가 포함된다.
- 분석의 한계와 인과관계로 해석할 수 없다는 점이 명시된다.

### Phase 7. 통합 실행 및 문서화

상태: 예정

수행 내용:

- 의존성 버전 고정
- 설정 파일 작성
- 전체 파이프라인 실행 명령 정리
- 테스트 실행
- 최종 모델 카드 작성
- 생성된 파일과 실행 결과 점검

예상 실행 흐름:

```powershell
python prepare_stock_data.py
python -m src.feature_engineering
python -m src.train_baseline
python -m src.train_candidates
python -m src.evaluate
python -m src.predict
python -m src.analyze_indicators
```

완료 조건:

- 빈 산출물 디렉터리에서 전체 파이프라인이 재실행된다.
- 같은 데이터와 설정으로 주요 성능 수치가 재현된다.
- 모든 테스트가 통과한다.
- 최종 모델, 평가 결과, 최신 예측, RSI·MACD 분석 문서가 존재한다.

## 6. 테스트 계획

### 데이터 테스트

- 필수 열 존재 여부
- 날짜 파싱 및 오름차순 정렬
- 날짜 중복 여부
- OHLC 논리 관계
- 가격과 거래량의 양수 여부

### 특성 테스트

- 수익률 계산식 검증
- 이동평균과 변동성 계산 검증
- RSI 및 MACD 변화량 계산 검증
- 현재 행보다 미래 데이터를 참조하지 않는지 검증

### 목표값 테스트

- `t`의 `target_next_open`이 `t+1` 시가와 같은지 검증
- `t`의 `target_next_close`가 `t+1` 종가와 같은지 검증
- 마지막 행의 목표값이 결측인지 검증

### 분할 테스트

- `max(train_date) < min(validation_date)`
- `max(validation_date) < min(test_date)`
- 같은 날짜가 여러 구간에 중복되지 않음
- 스케일러가 학습 구간에만 적합됨

### 모델 테스트

- 학습 완료 및 모델 저장
- 저장 모델 재로딩
- 예측값의 개수와 대상 행 수 일치
- NaN 또는 무한대 예측 없음
- 동일 시드에서 동일 결과 재현

## 7. 주요 의사결정

| 항목 | 결정 |
|---|---|
| 예측 시점 | 거래일 장 종료 직후 |
| 예측 대상 | 다음 실제 거래일 시가와 종가 |
| 학습 목표 | 현재 종가 대비 익일 시가·종가 수익률 |
| 최종 출력 | 원 단위 예상 시가와 종가 |
| 데이터 순서 | 과거에서 미래 |
| 모델 구성 | 시가·종가 개별 모델 |
| 기본 모델 | Naive 및 Ridge |
| 후보 모델 | Elastic Net, Random Forest, HistGradientBoosting |
| 데이터 분할 | 시간순 Train/Validation/Test |
| 모델 선택 | 워크포워드 Validation 결과 |
| 최종 확인 | 독립 Test 구간 |
| 지표 설명 | RSI·MACD 별도 영향 분석 문서 |

## 8. 위험 요소와 대응

| 위험 | 대응 |
|---|---|
| 거래량 K/M 단위의 수집 일관성 문제 | 모델링 전에 기간별 분포를 재점검하고 필요하면 거래량 특성을 제외한 실험 수행 |
| 공식 거래일 검증 미완료 | 외부 거래소 캘린더를 사용할 수 있을 때 추가 대조 |
| 최신 행 OHLC 오류로 2026-06-24 제외 | 원본 수집 로직 확인 후 수정 데이터가 확보되면 재정제 |
| 데이터 수가 약 2,447행으로 제한적 | 단순 모델을 우선하고 특성 수와 탐색 범위를 제한 |
| 장기간 시장 구조 변화 | 워크포워드 검증과 기간별 오차 분석 |
| 기술 지표 간 다중공선성 | Ridge, 제거 실험 및 permutation importance 병행 |
| 테스트 구간 반복 확인에 따른 과적합 | 최종 설정 확정 전 테스트 결과를 열지 않음 |

## 9. 전체 완료 기준

다음 조건이 모두 충족되면 프로젝트 구현을 완료한 것으로 본다.

- 정제 데이터로부터 파생 특성과 익일 목표값이 재현 가능하게 생성된다.
- Naive, 선형 및 트리 기반 모델이 동일한 시간 구간에서 비교된다.
- 최종 시가·종가 모델과 전처리기가 저장된다.
- 독립 테스트 성능과 Naive 대비 개선 정도가 기록된다.
- 최신 기준일의 다음 거래일 예상 시가와 종가가 출력된다.
- RSI와 MACD가 익일 종가 예측에 반영된 방식을 별도 문서로 설명한다.
- 원본 데이터는 보존된다.
- 전체 테스트와 재실행 검증이 통과한다.

## 10. 다음 작업

다음 구현 단계는 Phase 4다.

1. HistGradientBoostingRegressor 구현
2. RandomForestRegressor 구현
3. 학습 구간 내부 워크포워드 하이퍼파라미터 비교
4. 고정 Validation 구간에서 기준 모델과 동일 조건으로 평가
5. 시가·종가별 최종 후보 모델 선택
6. 테스트 구간을 열기 전 선택 결과와 근거 저장
