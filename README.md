# Mô hình khai phá dữ liệu: Dự đoán xu hướng dùng xe điện

Project này là bộ khung để làm bài tập môn Khai Phá Dữ Liệu với đề tài dự đoán xu hướng sử dụng xe điện (EV) trong tương lai.

## 1) Ý tưởng mô hình

- **Bài toán**: dự đoán `ev_share_percent` (tỷ lệ sử dụng EV, %)
- **Dữ liệu theo chuỗi thời gian** theo mốc `year` (có thể là năm nguyên hoặc năm thập phân theo tháng)
- **Đặc trưng**:
  - Đặc trưng trễ (lag) của chính tỷ lệ EV: `lag_1`, `lag_2`, `lag_3`
  - Trung bình trượt 3 năm gần nhất
  - Biến ngoại sinh (nếu có): số trạm sạc, chỉ số giá nhiên liệu, GDP bình quân đầu người...
- **Mô hình so sánh**:
  - Hồi quy tuyến tính (`LinearRegression`)
  - Rừng ngẫu nhiên (`RandomForestRegressor`)
- Chọn mô hình tốt nhất theo **RMSE** trên tập kiểm tra.

## 2) Cấu trúc dữ liệu đầu vào

File CSV cần có tối thiểu:

- Một cột thời gian: `thoi_gian` (khuyến nghị, dạng `YYYY-MM`) hoặc `date` hoặc `year`
- `ev_share_percent`

Có thể thêm cột số khác làm biến giải thích (ví dụ: `charging_stations`, `fuel_price_index`, `gdp_per_capita`).
Với dữ liệu giá xe, ưu tiên dùng `gia_tien_vnd` hoặc `vehicle_price_vnd` (đơn vị VND).

Dữ liệu đang có trong thư mục `data/`:

- `data/ev_adoption_sample.csv`
- `data/ev_adoption_vietnam_monthly_enriched_1984_2025.csv`
- `data/ev_adoption_vietnam_monthly_enriched_2018_2026.csv`
- `data/ev_adoption_vietnam_cars_motorbikes.csv` (gồm cả `o_to_dien` và `xe_may_dien`)

Ngoài các biến ban đầu, dữ liệu hiện có thêm:

- `battery_price_index` (chỉ số giá pin, thường giảm dần theo thời gian)
- `urbanization_rate` (tỷ lệ đô thị hóa)

Việc thêm số năm lịch sử và biến giải thích giúp mô hình học xu hướng ổn định hơn, từ đó dự báo tốt hơn.

## 3) Cài đặt

```bash
pip install -r requirements.txt
```

## 4) Huấn luyện và dự báo

```bash
python src/train_ev_trend.py --input data/ev_adoption_sample.csv --output_dir outputs --test_years 3 --forecast_horizon 5
```

Chạy với bộ dữ liệu lớn hơn ~500 bản ghi:

```bash
python src/train_ev_trend.py --input data/ev_adoption_vietnam_monthly_enriched_1984_2025.csv --output_dir outputs_vn_500 --test_years 24 --forecast_horizon 12
```

Chạy với dữ liệu Việt Nam từ 2018 đến nay:

```bash
python src/train_ev_trend.py --input data/ev_adoption_vietnam_monthly_enriched_2018_2026.csv --output_dir outputs_vn_2018_present --test_years 12 --forecast_horizon 12
```

Chạy với dữ liệu gồm cả ô tô điện + xe máy điện (dự báo theo từng loại):

```bash
python src/train_ev_trend.py --input data/ev_adoption_vietnam_cars_motorbikes.csv --output_dir outputs_vn_all_types --test_years 12 --forecast_horizon 8
```

## 5) Kết quả đầu ra

- `outputs/metrics.json`: so sánh các mô hình + mô hình tốt nhất
- `outputs/forecast.csv`: dự báo tỷ lệ EV cho các năm tương lai. Nếu dữ liệu đầu vào kết thúc ở năm hiện tại nhưng chưa đủ năm (ví dụ mốc thời gian dạng `2026.x`), dự báo sẽ bao gồm cả năm hiện tại.

Với dữ liệu có cột `loai_phuong_tien`, file `forecast.csv` sẽ có thêm cột này để phân biệt dự báo `o_to_dien` và `xe_may_dien` theo từng năm.
