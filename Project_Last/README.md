# 삼성전자 익일 시가·종가 예측 프로젝트

삼성전자(005930) 일봉 데이터를 정제하고 파생 특성을 생성한 뒤, 다음 실제 거래일의 시가와 종가를 추정한 프로젝트입니다.

Streamlit 대시보드에서 다음 내용을 확인할 수 있습니다.

- 사용한 원본 데이터와 정제 결과
- 32개 파생 특성과 익일 목표값
- 시간순 8:1:1 데이터 분할
- Naive, 20일 평균 수익률, Ridge 모델
- 검증 구간 성능
- 독립 테스트 구간 성능
- 실제 가격과 예측 가격 비교
- PRD와 구현·평가 문서

## 실행 방법

Python 가상환경 사용을 권장합니다.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

실행 후 브라우저에서 기본적으로 다음 주소가 열립니다.

```text
http://localhost:8501
```

## 데이터 파이프라인 재실행

```powershell
python prepare_stock_data.py
python -m src.feature_engineering
python -m src.train_baseline
python -m src.evaluate_test
```

주의: 독립 테스트 구간은 이미 한 번 평가되었습니다. 해당 구간을 추가 모델 선택이나 하이퍼파라미터 조정에 사용하면 독립 테스트로 볼 수 없습니다.

## 테스트

```powershell
python -m unittest discover -s tests -v
```

## 주요 산출물

```text
streamlit_app.py
data/processed/005930_2016to2026_clean.csv
data/processed/005930_model_dataset.csv
reports/baseline_validation_report.md
reports/independent_test_report.md
reports/metrics/
reports/predictions/
```

## 한계

독립 테스트에서 검증 대비 가격 오차가 크게 증가했습니다. 현재 모델은 연구용 기준선이며 투자 판단이나 자동매매에 사용할 수준이 아닙니다.
