"""Streamlit dashboard for the 005930 next-day regression project."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent

PATHS = {
    "clean": ROOT / "data/processed/005930_2016to2026_clean.csv",
    "model_data": ROOT / "data/processed/005930_model_dataset.csv",
    "excluded": ROOT / "reports/005930_excluded_rows.csv",
    "quality": ROOT / "reports/005930_data_quality.json",
    "features": ROOT / "reports/feature_summary.json",
    "baseline_summary": ROOT / "reports/metrics/baseline_run_summary.json",
    "validation_metrics": ROOT
    / "reports/metrics/baseline_validation_metrics.csv",
    "ridge_cv": ROOT / "reports/metrics/ridge_cv_results.csv",
    "validation_predictions": ROOT
    / "reports/predictions/baseline_validation_predictions.csv",
    "test_summary": ROOT / "reports/metrics/independent_test_summary.json",
    "test_metrics": ROOT / "reports/metrics/independent_test_metrics.csv",
    "test_predictions": ROOT
    / "reports/predictions/independent_test_predictions.csv",
}

MODEL_LABELS = {
    "naive": "Naive",
    "rolling_mean_20": "20일 평균 수익률",
    "ridge": "Ridge",
}
TARGET_LABELS = {"open": "익일 시가", "close": "익일 종가"}


@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require_files(paths: list[Path]) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        st.error("필요한 산출물 파일을 찾을 수 없습니다.")
        st.code("\n".join(missing))
        st.stop()


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def format_won(value: float) -> str:
    return f"{value:,.0f}원"


def format_period(frame: pd.DataFrame, column: str = "date") -> str:
    dates = pd.to_datetime(frame[column])
    return f"{dates.min():%Y-%m-%d} ~ {dates.max():%Y-%m-%d}"


def project_overview() -> None:
    quality = read_json(PATHS["quality"])
    features = read_json(PATHS["features"])
    split = read_json(PATHS["baseline_summary"])
    test = read_json(PATHS["test_summary"])

    st.subheader("프로젝트 개요")
    st.write(
        "삼성전자(005930) 일봉 데이터로 다음 실제 거래일의 시가와 종가를 "
        "추정하는 회귀 모델 프로젝트입니다. 가격 자체 대신 현재 종가 대비 "
        "익일 수익률을 학습하고, 예측 수익률을 다시 원 단위 가격으로 복원합니다."
    )

    columns = st.columns(4)
    with columns[0]:
        metric_card("정제 데이터", f"{quality['clean_rows']:,}행")
    with columns[1]:
        metric_card("모델 데이터", f"{features['model_rows']:,}행")
    with columns[2]:
        metric_card("입력 특성", f"{features['feature_count']}개")
    with columns[3]:
        metric_card("독립 테스트", f"{test['test']['rows']:,}행")

    st.markdown("#### 전체 흐름")
    flow = pd.DataFrame(
        [
            ["1", "데이터 정제", "2018-06-01 이전 제거, OHLC 검증"],
            ["2", "특성 생성", "수익률·이동평균·RSI·MACD 등 32개"],
            ["3", "시간순 분할", "Train 80% / Validation 10% / Test 10%"],
            ["4", "기준 모델 검증", "Naive·20일 평균·Ridge"],
            ["5", "독립 테스트", "검증에서 선택한 모델을 한 번 평가"],
        ],
        columns=["단계", "작업", "내용"],
    )
    st.dataframe(flow, hide_index=True, width="stretch")

    st.markdown("#### 시간순 데이터 구간")
    periods = pd.DataFrame(
        [
            [
                "Train",
                split["train"]["date_min"],
                split["train"]["date_max"],
                split["train"]["rows"],
            ],
            [
                "Validation",
                split["validation"]["date_min"],
                split["validation"]["date_max"],
                split["validation"]["rows"],
            ],
            [
                "Test",
                split["test"]["date_min"],
                split["test"]["date_max"],
                split["test"]["rows"],
            ],
        ],
        columns=["구간", "시작일", "종료일", "행 수"],
    )
    fig = px.timeline(
        periods,
        x_start="시작일",
        x_end="종료일",
        y="구간",
        color="구간",
        text="행 수",
        category_orders={"구간": ["Test", "Validation", "Train"]},
    )
    fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10))
    st.plotly_chart(fig, width="stretch")

    st.warning(
        "이 프로젝트는 연구용입니다. 독립 테스트에서 최근 고변동 시장에 대한 "
        "일반화 성능이 낮았으며 투자 또는 자동매매 판단에 사용할 수준이 아닙니다."
    )


def data_quality_page() -> None:
    quality = read_json(PATHS["quality"])
    clean = read_csv(PATHS["clean"])
    excluded = read_csv(PATHS["excluded"])
    clean["날짜"] = pd.to_datetime(clean["날짜"])

    st.subheader("사용 데이터와 정제 과정")
    columns = st.columns(4)
    with columns[0]:
        metric_card("원본 행", f"{quality['source_rows']:,}")
    with columns[1]:
        metric_card("정제 행", f"{quality['clean_rows']:,}")
    with columns[2]:
        metric_card("제외 행", f"{quality['excluded_rows']:,}")
    with columns[3]:
        metric_card("사용 기간", f"{quality['clean_date_min']} 이후")

    st.caption(f"원본 파일: `{quality['source_file']}`")

    price_fig = go.Figure()
    price_fig.add_trace(
        go.Scatter(
            x=clean["날짜"],
            y=clean["종가"],
            name="종가",
            line=dict(color="#3b82f6"),
        )
    )
    price_fig.update_layout(
        title="정제 후 종가",
        xaxis_title="날짜",
        yaxis_title="원",
        height=400,
        margin=dict(l=10, r=10),
    )
    st.plotly_chart(price_fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("#### 제외 사유")
        reason_data = pd.DataFrame(
            [
                {"사유": key, "행 수": value}
                for key, value in quality["excluded_by_reason"].items()
            ]
        )
        st.dataframe(reason_data, hide_index=True, width="stretch")
    with right:
        st.markdown("#### 검증 상태")
        st.write(f"- 날짜 중복: **{quality['duplicate_dates']}건**")
        st.write(
            f"- 시간순 정렬: **{'완료' if quality['chronological_order'] else '오류'}**"
        )
        st.write("- 2018-06-01 이전 데이터: **전부 제외**")
        st.write("- RSI·MACD: **정제 이후 재계산**")

    with st.expander("제외된 원본 행 보기"):
        st.dataframe(excluded, hide_index=True, width="stretch")

    with st.expander("정제 데이터 미리보기"):
        st.dataframe(clean.tail(200), hide_index=True, width="stretch")


def feature_page() -> None:
    summary = read_json(PATHS["features"])
    model_data = read_csv(PATHS["model_data"])

    st.subheader("파생 특성과 익일 목표값")
    columns = st.columns(4)
    with columns[0]:
        metric_card("특성 수", str(summary["feature_count"]))
    with columns[1]:
        metric_card("학습 가능 행", f"{summary['model_rows']:,}")
    with columns[2]:
        metric_card("롤링 제외 행", str(summary["warmup_or_incomplete_rows_removed"]))
    with columns[3]:
        metric_card("최신 특성 기준일", summary["latest_feature_date"])

    groups = {
        "수익률": [name for name in summary["feature_columns"] if name.startswith("return_")],
        "이동평균": [
            name for name in summary["feature_columns"] if name.startswith("close_to_ma_")
        ],
        "변동성": [
            name for name in summary["feature_columns"] if name.startswith("volatility_")
        ],
        "거래량": [
            name for name in summary["feature_columns"] if name.startswith("volume_to_ma_")
        ],
        "일봉 형태": [
            "intraday_return",
            "open_gap_return",
            "intraday_range",
            "close_position_in_range",
        ],
        "RSI": ["rsi_14", "rsi_change_1d"],
        "MACD": [
            "macd",
            "macd_signal",
            "macd_hist",
            "macd_change_1d",
            "macd_signal_change_1d",
            "macd_hist_change_1d",
        ],
        "외국인": ["foreign_ownership_pct", "foreign_ownership_change_1d"],
        "달력": ["day_of_week", "month"],
    }
    group_rows = [
        {"분류": group, "특성 수": len(columns), "특성": ", ".join(columns)}
        for group, columns in groups.items()
    ]
    st.dataframe(
        pd.DataFrame(group_rows),
        hide_index=True,
        width="stretch",
        column_config={"특성": st.column_config.TextColumn(width="large")},
    )

    st.markdown("#### 예측 목표값")
    target_table = pd.DataFrame(
        [
            ["target_next_open", "다음 실제 거래일 시가"],
            ["target_next_close", "다음 실제 거래일 종가"],
            [
                "target_next_open_return",
                "다음 시가 / 현재 종가 - 1",
            ],
            [
                "target_next_close_return",
                "다음 종가 / 현재 종가 - 1",
            ],
        ],
        columns=["열", "정의"],
    )
    st.dataframe(target_table, hide_index=True, width="stretch")

    st.markdown("#### 주요 특성 분포")
    selected = st.selectbox(
        "확인할 특성",
        ["rsi_14", "macd", "macd_hist", "return_20d", "volatility_20d"],
    )
    fig = px.histogram(model_data, x=selected, nbins=50)
    fig.update_layout(height=350, margin=dict(l=10, r=10))
    st.plotly_chart(fig, width="stretch")

    with st.expander("데이터 누수 방지 규칙"):
        for rule in summary["leakage_controls"]:
            st.write(f"- {rule}")


def metrics_table(frame: pd.DataFrame, include_selected: bool = False) -> pd.DataFrame:
    result = frame.copy()
    result["model"] = result["model"].map(MODEL_LABELS).fillna(result["model"])
    result["target"] = result["target"].map(TARGET_LABELS).fillna(result["target"])
    result["price_mae"] = result["price_mae"].round(1)
    result["price_rmse"] = result["price_rmse"].round(1)
    result["price_smape_pct"] = result["price_smape_pct"].round(3)
    result["return_mae"] = result["return_mae"].round(5)
    result["direction_accuracy"] = (
        result["direction_accuracy"] * 100
    ).round(2)
    columns = [
        "model",
        "target",
        "price_mae",
        "price_rmse",
        "price_smape_pct",
        "return_mae",
        "direction_accuracy",
    ]
    if include_selected:
        result["selected_from_validation"] = result[
            "selected_from_validation"
        ].map({True: "선택", False: ""})
        columns.insert(2, "selected_from_validation")
    result = result[columns]
    result.columns = [
        {
            "model": "모델",
            "target": "목표",
            "selected_from_validation": "검증 선택",
            "price_mae": "MAE(원)",
            "price_rmse": "RMSE(원)",
            "price_smape_pct": "sMAPE(%)",
            "return_mae": "수익률 MAE",
            "direction_accuracy": "방향 정확도(%)",
        }[column]
        for column in result.columns
    ]
    return result


def validation_page() -> None:
    summary = read_json(PATHS["baseline_summary"])
    metrics = read_csv(PATHS["validation_metrics"])
    ridge_cv = read_csv(PATHS["ridge_cv"])

    st.subheader("시간순 학습 및 검증")
    st.write(
        "전체 모델 데이터는 시간순 8:1:1로 나눴습니다. 구간 경계에서 "
        "익일 목표값이 다음 구간으로 넘어가는 것을 막기 위해 Train과 "
        "Validation의 마지막 행을 한 개씩 제거했습니다."
    )

    columns = st.columns(3)
    for column, key, label in zip(
        columns,
        ["train", "validation", "test"],
        ["Train", "Validation", "Test"],
    ):
        with column:
            item = summary[key]
            metric_card(label, f"{item['rows']:,}행")
            st.caption(f"{item['date_min']} ~ {item['date_max']}")

    st.markdown("#### 사용 모델")
    st.write(
        "- **Naive:** 다음 시가·종가 수익률을 0으로 가정\n"
        "- **20일 평균 수익률:** 최근 관측 수익률 평균을 순차적으로 사용\n"
        "- **Ridge:** 32개 특성 표준화 후 L2 규제 선형회귀"
    )
    st.caption(f"Ridge 내부 검증: {summary['ridge_cv']}")

    st.markdown("#### 검증 성능")
    st.dataframe(
        metrics_table(metrics),
        hide_index=True,
        width="stretch",
    )

    fig = px.bar(
        metrics.assign(
            모델=metrics["model"].map(MODEL_LABELS),
            목표=metrics["target"].map(TARGET_LABELS),
        ),
        x="모델",
        y="price_mae",
        color="목표",
        barmode="group",
        labels={"price_mae": "MAE(원)"},
        title="검증 구간 가격 MAE",
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Ridge alpha 워크포워드 결과"):
        display = ridge_cv.copy()
        display["target"] = display["target"].map(TARGET_LABELS)
        display["cv_return_mae"] = -display["mean_cv_negative_return_mae"]
        st.dataframe(
            display[
                [
                    "target",
                    "param_ridge__alpha",
                    "cv_return_mae",
                    "rank_test_score",
                ]
            ],
            hide_index=True,
            width="stretch",
        )


def prediction_chart(
    predictions: pd.DataFrame, target: str, model: str
) -> go.Figure:
    predictions = predictions.copy()
    predictions["next_date"] = pd.to_datetime(predictions["next_date"])
    actual_column = f"actual_{target}_price"
    predicted_column = f"predicted_{target}_price_{model}"
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=predictions["next_date"],
            y=predictions[actual_column],
            name="실제",
            line=dict(color="#2563eb", width=2),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=predictions["next_date"],
            y=predictions[predicted_column],
            name="예측",
            line=dict(color="#f97316", width=2),
        )
    )
    figure.update_layout(
        xaxis_title="대상 거래일",
        yaxis_title="가격(원)",
        height=430,
        margin=dict(l=10, r=10),
    )
    return figure


def independent_test_page() -> None:
    summary = read_json(PATHS["test_summary"])
    metrics = read_csv(PATHS["test_metrics"])
    predictions = read_csv(PATHS["test_predictions"])
    validation = read_csv(PATHS["validation_metrics"])

    st.subheader("독립 테스트 평가")
    st.error(
        "독립 테스트 구간은 이미 한 번 열었습니다. 이후 모델 선택이나 "
        "하이퍼파라미터 조정에 다시 사용하면 독립 테스트가 아닙니다."
    )

    selected = metrics.loc[metrics["selected_from_validation"]].copy()
    selected_map = {
        row["target"]: row for _, row in selected.iterrows()
    }
    columns = st.columns(4)
    with columns[0]:
        metric_card(
            "시가 테스트 MAE",
            format_won(selected_map["open"]["price_mae"]),
        )
    with columns[1]:
        metric_card(
            "시가 방향 정확도",
            f"{selected_map['open']['direction_accuracy'] * 100:.1f}%",
        )
    with columns[2]:
        metric_card(
            "종가 테스트 MAE",
            format_won(selected_map["close"]["price_mae"]),
        )
    with columns[3]:
        metric_card(
            "종가 sMAPE",
            f"{selected_map['close']['price_smape_pct']:.2f}%",
        )

    st.markdown("#### 테스트 성능")
    st.dataframe(
        metrics_table(metrics, include_selected=True),
        hide_index=True,
        width="stretch",
    )

    target = st.radio(
        "예측 대상",
        options=["open", "close"],
        format_func=lambda value: TARGET_LABELS[value],
        horizontal=True,
    )
    available_models = list(MODEL_LABELS)
    selected_default = summary["selected_models"][target]
    model = st.selectbox(
        "모델",
        options=available_models,
        index=available_models.index(selected_default),
        format_func=lambda value: MODEL_LABELS[value],
    )
    st.plotly_chart(
        prediction_chart(predictions, target, model),
        width="stretch",
    )

    actual_column = f"actual_{target}_price"
    predicted_column = f"predicted_{target}_price_{model}"
    error_frame = predictions[["next_date", actual_column, predicted_column]].copy()
    error_frame["절대 오차"] = (
        error_frame[predicted_column] - error_frame[actual_column]
    ).abs()
    error_frame["오차율(%)"] = (
        error_frame["절대 오차"] / error_frame[actual_column] * 100
    )
    error_fig = px.bar(
        error_frame,
        x="next_date",
        y="절대 오차",
        title="거래일별 절대 오차",
    )
    error_fig.update_layout(height=330, margin=dict(l=10, r=10))
    st.plotly_chart(error_fig, width="stretch")

    st.markdown("#### 검증 대비 테스트 성능 변화")
    comparison_rows = []
    for target_name, model_name in summary["selected_models"].items():
        validation_row = validation.loc[
            (validation["target"] == target_name)
            & (validation["model"] == model_name)
        ].iloc[0]
        test_row = metrics.loc[
            (metrics["target"] == target_name)
            & (metrics["model"] == model_name)
        ].iloc[0]
        comparison_rows.append(
            {
                "목표": TARGET_LABELS[target_name],
                "선택 모델": MODEL_LABELS[model_name],
                "검증 MAE": validation_row["price_mae"],
                "테스트 MAE": test_row["price_mae"],
                "증가 배율": test_row["price_mae"]
                / validation_row["price_mae"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    st.dataframe(
        comparison.style.format(
            {
                "검증 MAE": "{:,.0f}원",
                "테스트 MAE": "{:,.0f}원",
                "증가 배율": "{:.1f}배",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.warning(
        "검증 대비 테스트 MAE가 시가는 약 6.9배, 종가는 약 5.3배 "
        "증가했습니다. 최근 급등·고변동 구간에서 일반화 성능이 크게 "
        "저하됐으므로 모델 신뢰도는 낮습니다."
    )


def documents_page() -> None:
    st.subheader("프로젝트 문서")
    documents = {
        "PRD": ROOT / "PRD_NEXT_DAY_STOCK_REGRESSION.md",
        "구현 계획": ROOT / "IMPLEMENTATION_PLAN.md",
        "데이터 품질 보고서": ROOT / "reports/005930_data_quality.md",
        "기준 모델 검증 보고서": ROOT / "reports/baseline_validation_report.md",
        "독립 테스트 보고서": ROOT / "reports/independent_test_report.md",
    }
    selected = st.selectbox("문서 선택", list(documents))
    st.markdown(read_text(documents[selected]))


def main() -> None:
    st.set_page_config(
        page_title="005930 익일 가격 예측 프로젝트",
        page_icon="📈",
        layout="wide",
    )
    require_files(list(PATHS.values()))

    st.title("삼성전자 익일 시가·종가 예측")
    st.caption(
        "데이터 정제부터 시간순 검증과 독립 테스트까지 프로젝트 전체 과정을 "
        "Git 저장소의 산출물로 재구성한 대시보드"
    )

    pages = {
        "프로젝트 개요": project_overview,
        "데이터 정제": data_quality_page,
        "특성과 목표값": feature_page,
        "학습과 검증": validation_page,
        "독립 테스트": independent_test_page,
        "문서": documents_page,
    }
    selection = st.sidebar.radio("메뉴", list(pages))
    st.sidebar.divider()
    st.sidebar.caption("대상 종목: 삼성전자 005930")
    st.sidebar.caption("모델 사용 데이터: 2018-06-01 이후")
    st.sidebar.warning("연구용 결과이며 투자 권유가 아닙니다.")
    pages[selection]()


if __name__ == "__main__":
    main()
