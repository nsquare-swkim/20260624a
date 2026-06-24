"""Validate, clean, and chronologically sort the stock chart CSV.

The source file is never modified. Outputs are written under data/processed
and reports.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_PATH = Path("005930_2016to2026_수정됨.txt")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

CLEAN_PATH = PROCESSED_DIR / "005930_2016to2026_clean.csv"
EXCLUDED_PATH = REPORTS_DIR / "005930_excluded_rows.csv"
QUALITY_JSON_PATH = REPORTS_DIR / "005930_data_quality.json"
QUALITY_MD_PATH = REPORTS_DIR / "005930_data_quality.md"

PRICE_COLUMNS = ["종가", "시가", "고가", "저가"]
INDICATOR_COLUMNS = [
    "RSI(14)",
    "MACD(12,26)",
    "MACD_SIGNAL(9)",
    "MACD_HIST",
]
DATA_START_DATE = pd.Timestamp("2018-06-01")


def parse_number(series: pd.Series, suffix: str | None = None) -> pd.Series:
    text = series.astype("string").str.strip().str.replace(",", "", regex=False)
    if suffix:
        text = text.str.replace(suffix, "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def parse_volume(value: object) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan

    multipliers = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}
    unit = text[-1].upper()
    try:
        if unit in multipliers:
            return float(text[:-1]) * multipliers[unit]
        return float(text)
    except ValueError:
        return np.nan


def load_source(path: Path) -> pd.DataFrame:
    """Load the revised JSON-like text export or the legacy CSV."""
    if path.suffix.lower() == ".txt":
        rows: list[list[object]] = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip().rstrip(",")
            if not line.startswith('["'):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid data row at source line {line_number}"
                ) from error
            if len(row) == 7 and str(row[0]).isdigit():
                rows.append(row)

        if not rows:
            raise ValueError(f"No stock data rows found in {path}")

        return pd.DataFrame(
            rows,
            columns=[
                "날짜",
                "시가",
                "고가",
                "저가",
                "종가",
                "거래량",
                "외국인소진율",
            ],
        )

    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    values = np.full(len(close), np.nan)

    if len(close) <= period:
        return pd.Series(values, index=close.index)

    average_gain = gain.iloc[1 : period + 1].mean()
    average_loss = loss.iloc[1 : period + 1].mean()

    def value(avg_gain: float, avg_loss: float) -> float:
        if avg_gain == 0 and avg_loss == 0:
            return 50.0
        if avg_loss == 0:
            return 100.0
        strength = avg_gain / avg_loss
        return 100 - (100 / (1 + strength))

    values[period] = value(average_gain, average_loss)
    for position in range(period + 1, len(close)):
        average_gain = (
            average_gain * (period - 1) + gain.iloc[position]
        ) / period
        average_loss = (
            average_loss * (period - 1) + loss.iloc[position]
        ) / period
        values[position] = value(average_gain, average_loss)

    return pd.Series(values, index=close.index)


def add_reason(
    reasons: pd.Series, mask: pd.Series, reason: str
) -> None:
    reasons.loc[mask] = reasons.loc[mask].map(
        lambda current: f"{current};{reason}" if current else reason
    )


def build_markdown_report(report: dict[str, object]) -> str:
    reason_lines = "\n".join(
        f"| {reason} | {count} |"
        for reason, count in report["excluded_by_reason"].items()
    )
    warning_lines = "\n".join(
        f"- {warning}" for warning in report["warnings"]
    )

    return f"""# 005930 데이터 품질 보고서

## 처리 결과

| 항목 | 값 |
|---|---:|
| 원본 파일 | {report["source_file"]} |
| 원본 행 수 | {report["source_rows"]} |
| 정제 행 수 | {report["clean_rows"]} |
| 제외 행 수 | {report["excluded_rows"]} |
| 중복 날짜 수 | {report["duplicate_dates"]} |
| 정제 시작일 | {report["clean_date_min"]} |
| 정제 종료일 | {report["clean_date_max"]} |
| 시간순 정렬 | {report["chronological_order"]} |

## 제외 사유

한 행이 여러 사유에 해당할 수 있으므로 사유별 합계는 제외 행 수보다 클 수 있다.

| 사유 | 행 수 |
|---|---:|
{reason_lines}

## 정제 규칙

- 토요일과 일요일 행 제외
- 날짜, OHLC, 거래량 파싱 실패 행 제외
- 거래량 결측 또는 0 이하 행 제외
- 고가·저가가 시가·종가 범위를 위반하는 행 제외
- 중복 날짜는 원본에서 먼저 등장한 행만 유지
- 2018-06-01 이전 데이터 제외
- 제외 후 변동률, RSI(14), MACD(12, 26), Signal(9), Histogram 재계산
- 결과는 과거에서 미래 방향으로 정렬

## 주의 사항

{warning_lines}

상세 제외 행은 `{EXCLUDED_PATH.as_posix()}`에서 확인할 수 있다.
"""


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_PATH}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    source = load_source(SOURCE_PATH)
    required = ["날짜", *PRICE_COLUMNS, "거래량"]
    missing_columns = [column for column in required if column not in source]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = source.copy()
    data["_source_row"] = np.arange(2, len(data) + 2)
    normalized_date = data["날짜"].astype(str).str.replace(" ", "", regex=False)
    normalized_date = normalized_date.str.replace("-", "", regex=False)
    data["_date"] = pd.to_datetime(
        normalized_date,
        format="%Y%m%d",
        errors="coerce",
    )
    for column in PRICE_COLUMNS:
        data[f"_{column}"] = parse_number(data[column])
    data["_volume"] = data["거래량"].map(parse_volume)
    if "변동 %" in data:
        data["_source_change_pct"] = parse_number(data["변동 %"], suffix="%")
    else:
        data["_source_change_pct"] = np.nan
    if "외국인소진율" in data:
        data["_foreign_ownership_pct"] = parse_number(data["외국인소진율"])

    reasons = pd.Series("", index=data.index, dtype="string")
    add_reason(reasons, data["_date"].isna(), "invalid_date")

    invalid_price = data[[f"_{column}" for column in PRICE_COLUMNS]].isna().any(
        axis=1
    )
    add_reason(reasons, invalid_price, "invalid_price")

    nonpositive_price = (
        data[[f"_{column}" for column in PRICE_COLUMNS]] <= 0
    ).any(axis=1)
    add_reason(reasons, nonpositive_price, "nonpositive_price")

    weekend = data["_date"].dt.dayofweek >= 5
    add_reason(reasons, weekend.fillna(False), "weekend")

    before_start_date = data["_date"] < DATA_START_DATE
    add_reason(
        reasons,
        before_start_date.fillna(False),
        "before_2018_06_01",
    )

    missing_volume = data["_volume"].isna()
    add_reason(reasons, missing_volume, "missing_volume")
    add_reason(
        reasons,
        data["_volume"].notna() & (data["_volume"] <= 0),
        "nonpositive_volume",
    )

    close = data["_종가"]
    open_price = data["_시가"]
    high = data["_고가"]
    low = data["_저가"]
    invalid_ohlc = (
        (high < close)
        | (high < open_price)
        | (low > close)
        | (low > open_price)
        | (high < low)
    )
    add_reason(reasons, invalid_ohlc.fillna(False), "invalid_ohlc")

    mixed_price_unit = data["_source_change_pct"] > 100
    add_reason(
        reasons,
        mixed_price_unit.fillna(False),
        "mixed_price_unit_change_over_100pct",
    )

    duplicate_date = data["_date"].notna() & data["_date"].duplicated(
        keep="first"
    )
    add_reason(reasons, duplicate_date, "duplicate_date")
    data["_exclusion_reason"] = reasons

    excluded = data.loc[data["_exclusion_reason"] != ""].copy()
    clean = data.loc[data["_exclusion_reason"] == ""].copy()
    clean = clean.sort_values("_date", kind="stable").reset_index(drop=True)

    clean["날짜"] = clean["_date"].dt.strftime("%Y-%m-%d")
    for column in PRICE_COLUMNS:
        clean[column] = clean[f"_{column}"].round().astype("int64")
    clean["거래량"] = clean["_volume"]
    if "_foreign_ownership_pct" in clean:
        clean["외국인소진율"] = clean["_foreign_ownership_pct"]

    # Recompute values affected by removed non-trading or mixed-unit rows.
    clean["변동 %"] = clean["종가"].pct_change() * 100
    clean["RSI(14)"] = calculate_rsi(clean["종가"].astype(float), period=14)
    ema_12 = clean["종가"].ewm(span=12, adjust=False).mean()
    ema_26 = clean["종가"].ewm(span=26, adjust=False).mean()
    clean["MACD(12,26)"] = ema_12 - ema_26
    clean["MACD_SIGNAL(9)"] = clean["MACD(12,26)"].ewm(
        span=9, adjust=False
    ).mean()
    clean["MACD_HIST"] = (
        clean["MACD(12,26)"] - clean["MACD_SIGNAL(9)"]
    )

    output_columns = [
        "날짜",
        "종가",
        "시가",
        "고가",
        "저가",
        "거래량",
        "변동 %",
        *INDICATOR_COLUMNS,
    ]
    if "외국인소진율" in clean:
        output_columns.append("외국인소진율")
    clean_output = clean[output_columns].copy()
    numeric_float_columns = ["거래량", "변동 %", *INDICATOR_COLUMNS]
    clean_output[numeric_float_columns] = clean_output[
        numeric_float_columns
    ].round(6)
    clean_output.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")

    excluded_output = excluded[
        [
            "_source_row",
            "날짜",
            "종가",
            "시가",
            "고가",
            "저가",
            "거래량",
            *(["변동 %"] if "변동 %" in excluded else []),
            *(["외국인소진율"] if "외국인소진율" in excluded else []),
            "_exclusion_reason",
        ]
    ].rename(
        columns={
            "_source_row": "원본_행번호",
            "_exclusion_reason": "제외_사유",
        }
    )
    excluded_output.to_csv(
        EXCLUDED_PATH, index=False, encoding="utf-8-sig"
    )

    reason_counts: Counter[str] = Counter()
    for reason_text in excluded["_exclusion_reason"]:
        reason_counts.update(str(reason_text).split(";"))

    volume_text = source["거래량"].dropna().astype(str).str.strip()
    volume_units = (
        volume_text[volume_text.str[-1].str.upper().isin(["K", "M", "B"])]
        .str[-1]
        .str.upper()
        .value_counts()
        .to_dict()
    )
    report = {
        "source_file": str(SOURCE_PATH),
        "clean_file": str(CLEAN_PATH),
        "excluded_file": str(EXCLUDED_PATH),
        "source_rows": int(len(source)),
        "clean_rows": int(len(clean_output)),
        "excluded_rows": int(len(excluded_output)),
        "duplicate_dates": int(data["_date"].duplicated().sum()),
        "excluded_by_reason": dict(sorted(reason_counts.items())),
        "clean_date_min": clean_output["날짜"].min(),
        "clean_date_max": clean_output["날짜"].max(),
        "chronological_order": bool(
            pd.to_datetime(clean_output["날짜"]).is_monotonic_increasing
        ),
        "remaining_missing_values": {
            column: int(count)
            for column, count in clean_output.isna().sum().items()
        },
        "source_volume_unit_counts": {
            str(unit): int(count) for unit, count in volume_units.items()
        },
        "data_start_date": DATA_START_DATE.strftime("%Y-%m-%d"),
        "warnings": [
            "액면분할 이전 구간의 영향을 피하기 위해 2018-06-01 이전 데이터는 학습 대상에서 제외했다.",
            "거래소 공식 거래일 캘린더와의 대조는 아직 수행하지 않았다. 현재 단계에서는 주말, 거래량 결측 및 OHLC 논리 규칙을 사용했다.",
            "정제 후 첫 변동률과 초기 14개 RSI 값은 계산에 필요한 과거 데이터가 없어 결측인 것이 정상이다.",
        ],
    }

    QUALITY_JSON_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    QUALITY_MD_PATH.write_text(
        build_markdown_report(report),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
