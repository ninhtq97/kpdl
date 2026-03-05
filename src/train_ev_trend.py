from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_COL = "ev_share_percent"
TIME_COL = "time_index"
MIN_REQUIRED_ROWS = 8
DEFAULT_MAX_LAG = 3
DEFAULT_GROUP_COL = "vehicle_group"

TIME_ALIASES = ("thoi_gian", "date_time", "date", "year")
PRICE_ALIASES = ("gia_tien_vnd", "price_vnd", "vehicle_price_vnd", "vehicle_price_million_vnd")
GROUP_ALIASES = ("vehicle_group", "loai_phuong_tien")


@dataclass
class ModelResult:
    model_name: str
    mae: float
    rmse: float
    mape_percent: float
    r2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình dự đoán xu hướng EV.")
    parser.add_argument("--input", type=str, default="data/ev_adoption.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--test_years", type=int, default=5)
    parser.add_argument("--forecast_horizon", type=int, default=10)
    parser.add_argument("--group_col", type=str, default=DEFAULT_GROUP_COL)
    return parser.parse_args()


def build_models() -> dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=6,
            min_samples_leaf=1,
        ),
    }


def resolve_group_column(df: pd.DataFrame, requested_group_col: str) -> str | None:
    if requested_group_col in df.columns:
        return requested_group_col
    for col in GROUP_ALIASES:
        if col in df.columns:
            return col
    return None


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = {TIME_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

    if not np.issubdtype(df[TIME_COL].dtype, np.number):
        raise ValueError(f"Cột {TIME_COL} phải là kiểu số.")

    if not np.issubdtype(df[TARGET_COL].dtype, np.number):
        raise ValueError(f"Cột {TARGET_COL} phải là kiểu số.")

    if df[TIME_COL].isna().any() or df[TARGET_COL].isna().any():
        raise ValueError(f"Cột {TIME_COL}/{TARGET_COL} không được chứa giá trị rỗng.")

    if len(df) < MIN_REQUIRED_ROWS:
        raise ValueError(f"Cần tối thiểu {MIN_REQUIRED_ROWS} bản ghi để huấn luyện ổn định.")

    return df.sort_values(TIME_COL).reset_index(drop=True)


def normalize_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "loai_phuong_tien" in work.columns and "vehicle_group" not in work.columns:
        work["vehicle_group"] = work["loai_phuong_tien"]

    for price_col in PRICE_ALIASES:
        if price_col not in work.columns:
            continue
        if price_col == "vehicle_price_million_vnd":
            work["vehicle_price_vnd"] = (
                pd.to_numeric(work[price_col], errors="coerce") * 1_000_000
            )
        else:
            work["vehicle_price_vnd"] = pd.to_numeric(work[price_col], errors="coerce")
        break

    if "year" in work.columns:
        work[TIME_COL] = pd.to_numeric(work["year"], errors="coerce")
        if work[TIME_COL].isna().any():
            raise ValueError("Cột year chứa giá trị không hợp lệ.")
        return work

    for time_col in TIME_ALIASES:
        if time_col not in work.columns or time_col == "year":
            continue
        dt = pd.to_datetime(work[time_col], errors="coerce")
        if dt.isna().any():
            raise ValueError(f"Cột {time_col} chứa thời gian không hợp lệ.")
        work[TIME_COL] = dt.dt.year + (dt.dt.month - 1) / 12.0
        return work

    raise ValueError("Thiếu cột thời gian hợp lệ.")


def split_train_test(data: pd.DataFrame, test_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_years <= 0:
        raise ValueError("test_years phải > 0.")

    max_time = float(data[TIME_COL].max())
    cutoff_time = max_time - float(test_years)
    train_df = data[data[TIME_COL] <= cutoff_time].copy()
    test_df = data[data[TIME_COL] > cutoff_time].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Không đủ dữ liệu để tách train/test theo số năm yêu cầu."
            " Hãy giảm test_years hoặc thêm dữ liệu."
        )

    if len(train_df) < MIN_REQUIRED_ROWS:
        raise ValueError(
            "Tập train quá ít sau khi tách theo năm."
            " Hãy giảm test_years hoặc thêm dữ liệu."
        )

    return train_df, test_df


def evaluate(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> ModelResult:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    r2 = r2_score(y_true, y_pred)
    return ModelResult(model_name=model_name, mae=mae, rmse=rmse, mape_percent=mape, r2=r2)


def build_supervised_dataset(df: pd.DataFrame, max_lag: int = 3) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    for lag in range(1, max_lag + 1):
        work[f"{TARGET_COL}_lag_{lag}"] = work[TARGET_COL].shift(lag)

    work[f"{TARGET_COL}_rolling_mean_3"] = work[TARGET_COL].shift(1).rolling(3).mean()
    work = work.dropna().reset_index(drop=True)

    # Đã loại TIME_COL ra khỏi danh sách bị xóa, biến nó thành Feature đắc lực bắt Trend
    non_feature_cols = {
        TARGET_COL, "thoi_gian", "date_time", "date", "month", "year",
        "vehicle_group", "loai_phuong_tien"
    }
    raw_features = work.drop(columns=[c for c in non_feature_cols if c in work.columns])
    encoded_features = pd.get_dummies(raw_features, drop_first=False)

    model_df = pd.concat([work[[TARGET_COL]], encoded_features], axis=1)
    feature_cols = [c for c in model_df.columns if c != TARGET_COL]
    return model_df, feature_cols


def train_and_select_best_model(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series,
) -> tuple[ModelResult, object, list[ModelResult]]:
    models = build_models()
    results, trained_models =[], {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        results.append(evaluate(y_test, pred, model_name=model_name))
        trained_models[model_name] = model

    best_result = min(results, key=lambda result: result.rmse)
    return best_result, trained_models[best_result.model_name], results


def linear_extrapolation(x: np.ndarray, y: np.ndarray, x_future: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(x.astype(float), y.astype(float), deg=1)
    return slope * x_future.astype(float) + intercept


def is_flat_forecast(values: pd.Series, min_span: float = 0.5) -> bool:
    """Xác định xem dự báo có bị đường thẳng đi ngang (hoặc tịt về 0) hay không"""
    arr = values.to_numpy(dtype=float)
    if arr.size <= 1: return False
    return float(arr.max() - arr.min()) < min_span


def is_saturated_forecast(values: pd.Series, lower: float = 0.0, upper: float = 100.0, ratio: float = 0.3) -> bool:
    """Cảnh báo chuỗi dự báo bị dồn ở biên 0/100 quá nhiều điểm."""
    arr = values.to_numpy(dtype=float)
    if arr.size == 0:
        return False
    near_bound = np.isclose(arr, lower, atol=1e-9) | np.isclose(arr, upper, atol=1e-9)
    return float(near_bound.mean()) >= ratio


def forecast_future(model, original_df: pd.DataFrame, feature_cols: list[str], horizon: int) -> tuple[pd.DataFrame, str]:
    last_time = float(original_df[TIME_COL].max())

    # Đồng bộ bước nhảy thời gian dự báo đúng bằng bước nhảy lịch sử (e.g., 1 tháng)
    diffs = original_df[TIME_COL].diff().dropna()
    step_size = diffs[diffs > 0].median() if not diffs[diffs > 0].empty else 1.0

    num_steps = int(np.ceil(horizon / step_size))
    future_times = last_time + np.arange(1, num_steps + 1) * step_size
    x_train = original_df[TIME_COL].values

    exogenous_columns =[c for c in original_df.columns if c not in {TIME_COL, TARGET_COL}]
    numeric_exogenous =[c for c in exogenous_columns if np.issubdtype(original_df[c].dtype, np.number)]
    categorical_exogenous = [c for c in exogenous_columns if c not in numeric_exogenous]

    exogenous_future = {
        col: linear_extrapolation(x_train, original_df[col].values, future_times)
        for col in numeric_exogenous
    }
    categorical_defaults = {
        col: original_df[col].mode(dropna=True).iloc[0] if not original_df[col].mode().empty else "unknown"
        for col in categorical_exogenous
    }

    history_target = original_df[TARGET_COL].tolist()
    future_rows =[]

    # Dự báo cuốn chiếu từng bước (tháng)
    for idx, t in enumerate(future_times):
        row = {TIME_COL: t}
        for col in numeric_exogenous:
            row[col] = exogenous_future[col][idx]
        for col in categorical_exogenous:
            row[col] = categorical_defaults[col]

        for lag in (1, 2, 3):
            row[f"{TARGET_COL}_lag_{lag}"] = history_target[-lag]
        row[f"{TARGET_COL}_rolling_mean_3"] = float(np.mean(history_target[-3:]))

        x_row_raw = pd.DataFrame([row])
        x_row = pd.get_dummies(x_row_raw).reindex(columns=feature_cols, fill_value=0)

        y_hat = float(np.clip(model.predict(x_row)[0], 0.0, 100.0))
        history_target.append(y_hat)
        future_rows.append({"time_index": t, "predicted_ev_share_percent": y_hat})

    res_df = pd.DataFrame(future_rows)
    adjustment = "none"

    # Cứu hộ dự báo: Nếu RF hoặc LR bị phẳng/cắm về 0, chuyển qua ngoại suy Trend Line
    if is_flat_forecast(res_df["predicted_ev_share_percent"]) or is_saturated_forecast(
        res_df["predicted_ev_share_percent"]
    ):
        fallback_vals = linear_extrapolation(x_train, original_df[TARGET_COL].values, res_df["time_index"].values)
        res_df["predicted_ev_share_percent"] = np.clip(fallback_vals, 0.0, 100.0)
        adjustment = "fallback_linear_trend"

    res_df["year"] = np.floor(res_df["time_index"]).astype(int)
    start_year = int(np.floor(res_df["time_index"].min()))
    end_year = start_year + horizon - 1

    yearly_df = res_df.groupby("year")["predicted_ev_share_percent"].mean().reset_index()
    yearly_df = yearly_df[(yearly_df["year"] >= start_year) & (yearly_df["year"] <= end_year)].copy()

    return yearly_df, adjustment


def train_and_forecast_one_group(df_group: pd.DataFrame, test_years: int, forecast_horizon: int) -> tuple[dict, pd.DataFrame]:
    validated_df = validate_dataframe(df_group)
    supervised_df, feature_cols = build_supervised_dataset(validated_df, max_lag=DEFAULT_MAX_LAG)
    train_df, test_df = split_train_test(supervised_df, test_years)

    best_result, best_model, results = train_and_select_best_model(
        x_train=train_df[feature_cols],
        y_train=train_df[TARGET_COL],
        x_test=test_df[feature_cols],
        y_test=test_df[TARGET_COL],
    )

    forecast_df, adjustment = forecast_future(
        model=best_model, original_df=validated_df, feature_cols=feature_cols, horizon=forecast_horizon
    )

    metrics_payload = {
        "best_model": best_result.model_name,
        "forecast_adjustment": adjustment,
        "results":[asdict(r) for r in results],
    }
    return metrics_payload, forecast_df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_input_schema(pd.read_csv(input_path))
    group_col = resolve_group_column(df, requested_group_col=args.group_col)

    metrics_by_group = {}
    all_forecasts =[]

    if group_col is not None:
        for group_value in sorted(df[group_col].dropna().astype(str).unique()):
            group_df = df[df[group_col].astype(str) == group_value].copy()
            metrics, forecast_df = train_and_forecast_one_group(group_df, args.test_years, args.forecast_horizon)
            metrics_by_group[group_value] = metrics
            forecast_df[group_col] = group_value
            all_forecasts.append(forecast_df)
    else:
        metrics, forecast_df = train_and_forecast_one_group(df, args.test_years, args.forecast_horizon)
        metrics_by_group["tong_hop"] = metrics
        all_forecasts.append(forecast_df)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"metrics_by_group": metrics_by_group}, f, ensure_ascii=False, indent=2)

    forecast_path = output_dir / "forecast.csv"
    forecast_all_df = pd.concat(all_forecasts, ignore_index=True)
    sort_cols = ["year"]
    if group_col is not None and group_col in forecast_all_df.columns:
        sort_cols.append(group_col)
    forecast_all_df = forecast_all_df.sort_values(sort_cols).reset_index(drop=True)
    forecast_all_df.to_csv(forecast_path, index=False)

    print("=== Hoàn tất huấn luyện ===")
    print(f"Lưu metrics tại: {metrics_path}")
    print(f"Lưu dự báo tại: {forecast_path} (Đã fix lỗi sập 0% và đi ngang)")


if __name__ == "__main__":
    main()