# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Chí Hướng

**MSSV**: 2A202601203

**Ngày**: 2026-08-21

**Tier**: T4

**Base model**: `unsloth/Qwen3.5-4B`

**GPU thực tế**: Tesla T4, 14.6 GB VRAM

Mọi số liệu trong báo cáo này được lấy từ các artifact trong `results/`. Tôi giữ nguyên verdict FAILED, không sửa tập đánh giá, prompt baseline hoặc ngưỡng regression gate.

---

## 1. Setup

| Hạng mục | Giá trị |
|---|---|
| Dataset | 250 ticket chăm sóc khách hàng, đầu ra JSON triage |
| Train / validation | 225 / 25, seed 42 |
| `max_length` | 1024; p95 đo được là 98, mức gợi ý là 256 |
| `MASK_MODE` | assistant-only |
| Epochs / max steps | 2 epoch / 30 optimizer steps |
| Precision | fp16 |
| Effective batch | 16 |

Chat template **có giữ khối `<think>`**. `template_check.json` ghi nhận thẻ mở và nội dung reasoning đều tồn tại. Tôi giữ `max_length=1024` theo cấu hình T4 thống nhất của pipeline dù p95 chỉ là 98 token; lựa chọn này tránh cắt mất mẫu dài nếu thay dữ liệu, đổi template hoặc mở rộng đầu ra, đổi lại là tốn thêm bộ nhớ đệm.

---

## 2. Mask proof (NB1)

| Hạng mục | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 |
| Số token được tính loss | 39 / 94 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi không nằm trong loss | true |

Đoạn đầu của phần được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh",
 "product": "balo laptop", "sentiment": "trung_tinh"}
```

Kết quả chứng minh loss chỉ giám sát phần trả lời của assistant. Tỷ lệ 0.4149 cũng loại trừ lỗi phổ biến là huấn luyện model trên toàn bộ prompt.

---

## 3. Ba baseline và fine-tune

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.0000 | 0.7578 | 0.0000 | 3097.9 |
| (b) base + optimized prompt | 0.7650 | 0.7578 | 1.0000 | 998.8 |
| (c) LoRA fine-tune | 0.9650 | 0.4556 | 1.0000 | 1488.6 |

Baseline (b) thực sự mạnh hơn (a): target tăng từ 0.000 lên 0.765, format tăng từ 0 lên 1, đồng thời latency thấp hơn. Tôi không sửa `OPTIMIZED_PROMPT`; gatekeeper xác nhận SHA là `719e74d3b6232053`. Vì vậy fine-tune được so với một prompt baseline mạnh và không được hưởng lợi từ việc cố tình làm yếu đối chứng.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | Vị trí | r | Trainable | LR | Train loss | Target | Thời gian (s) | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6265 | 0.965 | 996.4 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 0.0001 | 0.5366 | 0.970 | 797.5 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5704 | 0.000 | 938.4 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 991.5 | 7.09 |

Ba đối chứng NB4 được phục hồi từ lần chạy đầy đủ trước đó trên cùng T4, cùng model và ngân sách 30 step. Run `correct` được chạy lại để khôi phục adapter bị mất sau khi runtime cũ reset.

### 4.1 — Vị trí adapter so với rank

`attn_only` có 32,456,704 tham số huấn luyện, rất gần 32,464,896 của `correct`; sai lệch nhỏ hơn 5%, nên đây là đối chứng công bằng. Trên target, `attn_only` đạt 0.970 còn `correct` đạt 0.965, tức attention-only nhỉnh hơn 0.005 và về thực tế gần như hòa. Train loss cũng xếp `attn_only` tốt hơn, 0.5366 so với 0.6265. Kết quả cho thấy với tác vụ JSON triage hẹp này, gắn adapter vào toàn bộ text-linear không tạo ưu thế rõ ràng khi ngân sách tham số đã được cân bằng. Tuy nhiên, không thể kết luận rank tự nó là đòn bẩy, vì rank 283 ở `attn_only` được tăng nhằm cân bằng ngân sách; phép thử đang đo vị trí dưới cùng ngân sách chứ không phải quét rank độc lập.

### 4.2 — Learning rate sai

`wrong_lr` chỉ giảm learning rate từ 0.0001 xuống 0.00001 nhưng train loss tăng từ 0.6265 lên 1.5704 và target rơi xuống 0.000. Với cùng 30 step, learning rate theo thang full fine-tuning không đủ lớn để LoRA học được ánh xạ tác vụ. Nếu chỉ nhìn đường loss giảm mà không biết LR và không đánh giá target, tôi có thể kết luận sai rằng model vẫn đang học chậm và chỉ cần thêm epoch. Kết quả target bằng không cho thấy lỗi thật nằm ở thang learning rate, không phải rank hay vị trí adapter.

### 4.3 — QLoRA và VRAM

QLoRA dùng 7.09 GB thay vì 12.01 GB, tiết kiệm 4.92 GB, tương đương khoảng 41.0%. Đổi lại, target giảm từ 0.965 xuống 0.940, train loss tăng từ 0.6265 lên 0.7058, và thời gian huấn luyện cũng không tốt hơn. Kết quả này ủng hộ khuyến nghị không ưu tiên QLoRA cho model này khi T4 vẫn đủ chạy fp16. QLoRA chỉ đáng cân nhắc nếu giới hạn VRAM quan trọng hơn mức giảm chất lượng target.

---

## 5. Phán quyết

**Kết quả regression gate: FAILED**

- Target delta: +0.200
- Regression delta: -0.302
- Valid reasoning trace rate: 0.00
- Format: 1.000
- Fine-tune latency: 1488.6 ms

Fine-tune cải thiện target từ 0.765 lên 0.965, tức tăng 0.200, và duy trì format tuyệt đối 1.0. Tuy nhiên, regression giảm từ 0.7578 xuống 0.4556, tương ứng mức giảm khoảng 0.302, vượt xa dung sai 0.020. Vì vậy verdict FAILED là hợp lý: model đã học rất tốt tác vụ triage nhưng đồng thời quên đáng kể năng lực tổng quát. `valid_trace_rate=0` cũng cho thấy không có bằng chứng rằng reasoning trace được duy trì trong đầu ra đánh giá, dù template và mask ban đầu đúng. Tôi không nới ngưỡng hoặc sửa tập eval để biến kết quả thành PASSED. Với bài toán này, prompt baseline đã đạt 0.765 mà không làm giảm regression, nên fine-tune hiện tại chưa tạo ra lợi ích đủ an toàn để triển khai. Hướng sửa hợp lý là thêm khoảng 1–5% replay data tổng quát rồi huấn luyện và đánh giá lại trên đúng tập đóng băng.

---

## 6. Đánh giá định tính

| # | Ticket rút gọn | Nhãn đúng | Baseline (b) | Fine-tune | Nhận xét |
|---|---|---|---:|---:|---|
| 1 | Balo laptop, đổi size, “hỏi cho biết thôi” | `doi_tra / thap / balo laptop / tieu_cuc` | 0.50 | 1.00 | ✅ FT thắng |
| 2 | Máy xay sinh tố, muốn đổi, đã 3 ngày, bực mình | `doi_tra / trung_binh / máy xay sinh tố / tieu_cuc` | 0.50 | 1.00 | ✅ FT thắng |
| 3 | Bình giữ nhiệt, chưa thấy tiền, khi nào tiện | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | 0.75 | 0.75 | ❌ FT sai urgency |
| 4 | Nồi chiên không dầu, thiếu phụ kiện, khi nào tiện | `san_pham_loi / thap / nồi chiên không dầu / trung_tinh` | 0.50 | 0.75 | ❌ FT sai urgency nhưng vẫn hơn baseline |
| 5 | Đèn bàn LED, hoàn tiền, quá hạn rồi | `hoan_tien / cao / đèn bàn LED / tich_cuc` | 1.00 | 1.00 | ➖ Hòa |

Trong toàn bộ 50 mẫu có 7 ca fine-tune chưa đạt điểm tuyệt đối. Hai ca sai được đưa vào bảng thay vì chỉ chọn các ca thắng. Mẫu chung nổi bật là model thường xác định đúng intent, product và sentiment nhưng có thể phân loại sai urgency ở các câu mang sắc thái “khi nào tiện”, “không vội” hoặc tín hiệu ưu tiên gián tiếp. Điều này giải thích vì sao các ca kém nhất vẫn đạt 0.75 thay vì thất bại hoàn toàn.

---

## 7. Kết luận và điều tôi học được

Tôi chưa nên deploy adapter fine-tune này trong môi trường cần giữ năng lực tổng quát. Kết quả target 0.965 và format 1.0 chứng minh LoRA đã học rất hiệu quả cấu trúc JSON triage, nhưng regression giảm 0.302 là một chi phí quá lớn so với dung sai 0.020. Baseline dùng prompt tối ưu đã đạt target 0.765, regression 0.7578 và latency thấp hơn, nên đây vẫn là phương án an toàn hơn nếu yêu cầu chất lượng target chưa bắt buộc gần tuyệt đối. Đòn bẩy rõ nhất trong thí nghiệm không phải chỉ là rank: `attn_only` sau khi khớp ngân sách tham số gần như hòa với `correct`, trong khi learning rate thấp hơn 10 lần khiến target về 0.000. Mask assistant-only đã đúng và không phải nguồn lỗi. Rủi ro chính của cấu hình hiện tại là dữ liệu huấn luyện quá chuyên biệt, dẫn đến quên thảm họa. Bước tiếp theo nên là bổ sung 1–5% replay data tổng quát, giữ nguyên tập eval và prompt baseline, sau đó đo lại cả bốn nhóm. Chỉ nên deploy nếu target vẫn tăng nhưng regression quay về trong dung sai.

**Ba điều tôi học được:**

1. Một loss mask đúng phải được chứng minh bằng token được giải mã ngược; supervised fraction 0.4149 cho thấy prompt đã thực sự bị loại khỏi loss.
2. So sánh vị trí adapter chỉ công bằng khi khớp cả tham số và số step; dùng cùng rank cho số module khác nhau sẽ làm sai kết luận.
3. Target tăng mạnh không đủ để quyết định deploy; regression gate đã phát hiện mức quên năng lực tổng quát mà train loss không thể hiện.

**Nếu có thêm hai giờ**, tôi sẽ trộn 1–5% replay data tổng quát vào tập train, chạy lại `correct` với cùng 30 step và đo lại target, regression, format và latency. Tôi cũng sẽ kiểm tra riêng các mẫu urgency gián tiếp vì đây là nhóm lỗi còn lại rõ nhất.

---

## Phụ lục — phần thưởng

- [ ] B1 NB6 merge và hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse với hai mask mode
- [ ] B4 quét rank có kiểm soát
- [ ] B5 Hugging Face Hub
