from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ dự báo từ file forecast.csv")
    parser.add_argument("--input", type=str, default="outputs/forecast.csv", help="Đường dẫn file forecast.csv")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/forecast_plot.png",
        help="Đường dẫn file ảnh đầu ra (.png)",
    )
    parser.add_argument(
        "--group_col",
        type=str,
        default="vehicle_group",
        help="Tên cột nhóm trong forecast (ví dụ: vehicle_group)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    forecast_df = pd.read_csv(input_path)

    required_cols = {"year", "predicted_ev_share_percent"}
    missing = required_cols - set(forecast_df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong forecast: {sorted(missing)}")

    plt.figure(figsize=(10, 6))

    if args.group_col in forecast_df.columns:
        for group_name, group_df in forecast_df.groupby(args.group_col):
            group_df = group_df.sort_values("year")
            plt.plot(
                group_df["year"],
                group_df["predicted_ev_share_percent"],
                marker="o",
                linewidth=2,
                label=str(group_name),
            )
        plt.legend(title=args.group_col)
    else:
        forecast_df = forecast_df.sort_values("year")
        plt.plot(
            forecast_df["year"],
            forecast_df["predicted_ev_share_percent"],
            marker="o",
            linewidth=2,
            label="forecast",
        )
        plt.legend()

    plt.title("Dự báo tỷ lệ EV theo năm")
    plt.xlabel("Năm")
    plt.ylabel("Tỷ lệ EV dự báo (%)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Đã lưu biểu đồ tại: {output_path}")


if __name__ == "__main__":
    main()
