# EV Trend Forecast (Vietnam)

Dự án dùng machine learning để dự báo `ev_share_percent` theo chuỗi thời gian, hỗ trợ cả bài toán tổng hợp và bài toán theo nhóm phương tiện (ví dụ `o_to_dien`, `xe_may_dien`).

## 1) Cài đặt

```bash
pip install -r requirements.txt
```

## 2) Chạy huấn luyện và dự báo

Lệnh mặc định (đúng theo code hiện tại):

```bash
python src/train_ev_trend.py
```

Mặc định tương ứng:
- `--input data/ev_adoption.csv`
- `--output_dir outputs`
- `--test_years 5`
- `--forecast_horizon 10`
- `--group_col vehicle_group`

Ví dụ tùy chỉnh:

```bash
python src/train_ev_trend.py --input data/ev_adoption.csv --output_dir outputs --test_years 3 --forecast_horizon 8 --group_col vehicle_group
```

## 3) Cấu trúc dữ liệu đầu vào

Cần tối thiểu:
- Cột thời gian: một trong `date_time`, `thoi_gian`, `date`, `year`
- Cột mục tiêu: `ev_share_percent`

Nếu có cột nhóm phương tiện:
- Ưu tiên `vehicle_group` (vẫn hỗ trợ alias cũ `loai_phuong_tien`)

Với dữ liệu giá xe, code tự nhận các alias:
- `price_vnd`, `gia_tien_vnd`, `vehicle_price_vnd`, `vehicle_price_million_vnd`

## 4) Logic mô hình hiện tại

- So sánh 2 mô hình: `LinearRegression` và `RandomForestRegressor`
- Chọn mô hình tốt nhất theo `RMSE` trên tập test
- Tạo đặc trưng trễ: `lag_1`, `lag_2`, `lag_3`, `rolling_mean_3`
- Tách train/test theo thời gian thực:
  - `cutoff_time = max(time_index) - test_years`
  - `train <= cutoff`, `test > cutoff`
- Forecast theo bước thời gian lịch sử (suy ra từ median delta của `time_index`), sau đó tổng hợp về mức năm
- Có cơ chế fallback khi forecast bị phẳng hoặc bị dồn ở biên:
  - `forecast_adjustment = fallback_linear_trend`
- Giá trị forecast được chặn trong khoảng `[0, 100]`

## 5) Output

- `outputs/metrics.json`
  - Chứa `metrics_by_group`
  - Mỗi nhóm có: `best_model`, `forecast_adjustment`, `results` (`MAE`, `RMSE`, `MAPE`, `R2`)
- `outputs/forecast.csv`
  - Cột chính: `year`, `predicted_ev_share_percent`
  - Nếu có phân nhóm: thêm cột `vehicle_group`

## 6) Vẽ biểu đồ từ output

```bash
python src/plot_forecast.py --input outputs/forecast.csv --output outputs/forecast_plot.png --group_col vehicle_group
```

File biểu đồ sinh ra tại: `outputs/forecast_plot.png`
