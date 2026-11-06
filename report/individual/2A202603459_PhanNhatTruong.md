# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603459
- **Họ và tên:** Phan Nhật
- **SEED:** 7441

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook có code và output thực nghiệm | Có |
| [`lab1_report.json`](lab1_report.json) — do ô báo cáo cuối sinh ra | Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay | Có |

Số liệu dưới đây lấy từ notebook và JSON ở cùng thư mục, được tải lên tại commit `d36e405`. Dữ liệu sau khi bỏ trùng có 5.169 tin: 4.516 ham và 653 spam. Train gồm 3.619 tin, validation 775 tin, test 775 tin; SUB lấy từ test, gồm 20 spam và 20 ham.

**Tình trạng cần đồng bộ:** bản notebook nộp vẫn để 5 giá trị `HAND_CALC` là `None`; JSON ghi 8/13 kiểm tra đạt và 5 kiểm tra tính tay chưa đạt. Các đáp án đã được bổ sung, kiểm tra khớp 5/5 hash trong notebook ở `src/` tại commit `6693ae5`, nhưng bản nộp chưa cập nhật. Vì vậy, đã có ảnh tính tay nhưng chưa thể ghi toàn bộ bài đã qua kiểm tra; cần cập nhật notebook nộp và xuất lại JSON.

## 2. Kết quả chính

| Phương pháp | F1 test 775 tin | F1 SUB 40 tin | p95 ms/tin | USD ước tính/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + Logistic Regression | 0,924 | 0,788 | 0,122 | 0 |
| BoW + MLP | 0,949 | 0,919 | 2,405 | 0 |
| MiniLM + kNN | 0,886 | 1,000 | 68,178 | 0 |
| MiniLM + Logistic Regression | 0,936 | 1,000 | 52,502 | 0 |
| MiniLM + MLP head | 0,917 | 1,000 | 39,068 | 0 |
| Router9 — Luna zero-shot | Chưa đo toàn test | 0,947 | 2.233,421 | 0,008197 |
| Router9 — Terra zero-shot | Chưa đo toàn test | 0,974 | 3.352,806 | Chưa xác nhận |
| Router9 — Luna few-shot | Chưa đo toàn test | 0,947 | 3.244,941 | 0,021397 |

LLM chỉ chạy trên SUB, nên không coi F1 đó là kết quả trên toàn bộ 775 tin. Terra hơn Luna khoảng 0,027 F1, nhưng chỉ tương ứng giảm từ 2 xuống 1 tin spam bị bỏ sót; precision đều bằng 1. SUB ít mẫu nên chưa đủ để kết luận model nào luôn tốt hơn ngoài thực tế.

Chi phí Terra trong JSON đang là 0 do bảng giá thiếu khóa `cx/gpt-6.1-terra`, không phải bằng chứng sử dụng miễn phí. Chi phí Luna là ước lượng token theo độ dài văn bản và giá tham khảo, chưa đối soát hóa đơn Router9. Giá trị 0 ở mô hình chạy cục bộ chỉ thể hiện không có phí API theo tin nhắn, chưa tính hạ tầng.

Output LLM ghi provider `router9`, endpoint `https://router9--llmgate--q8cj86cbs7t7.code.run/v1`, model `cx/gpt-6-luna` và `cx/gpt-6.1-terra`. Chưa có log máy chủ hoặc request ID để đối soát từng request độc lập.

## 3. Dự đoán và thực tế

| Thí nghiệm | Dự đoán đã ghi | Kết quả thực tế | Giải thích |
| --- | --- | --- | --- |
| `dl_curve` — BoW + MLP | MLP bằng hoặc kém LogReg vì ít dữ liệu, dễ overfit | F1 MLP 0,949, cao hơn LogReg 0,924 khoảng 0,025 | MLP học được tổ hợp đặc trưng phi tuyến. Tuy nhiên, val loss vẫn có thể tăng khi train loss giảm; điểm test cao hơn không đồng nghĩa không overfit. |
| `llm_fewshot` — thêm 4 ví dụ | F1 khoảng 0,95–0,97; chi phí tăng 1,5–2 lần | F1 giữ nguyên 0,947 so với zero-shot; chi phí tăng khoảng 2,61 lần | Ví dụ lấy từ TRAIN là hợp lệ nhưng chưa chắc giải quyết các trường hợp model đang sai. Ví dụ dài làm tăng input token mà không bảo đảm tăng F1. |
| `hard_eval` — 50 case khó | BoW + LogReg có accuracy 55–65%; LLM 75–85% | BoW + LogReg đạt 50%; hai phương án LLM đều đạt 100% | BoW thiếu ngữ cảnh nên bỏ sót spam có giọng tự nhiên. LLM làm tốt trên bộ thử này, nhưng 50 case viết tay chưa đại diện cho toàn bộ spam thực tế. |

Một kết quả khác không khớp dự đoán là khi bỏ ReLU, F1 vẫn bằng 0,949. Hai lớp Linear liên tiếp biểu diễn một phép biến đổi affine, nhưng cách tối ưu và số bước train khác LogReg tự code nên không nhất thiết cho kết quả giống thí nghiệm Phần 1.

## 4. Phân tích 50 case khó

| Phương pháp | F1 SUB thông thường | F1 50 case khó | Accuracy 50 case khó |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0,788 | 0,286 | 50% |
| BoW + MLP | 0,919 | 0,421 | 56% |
| MiniLM + kNN | 1,000 | 0,737 | 70% |
| MiniLM + LogReg | 1,000 | 0,640 | 64% |
| MiniLM + MLP head | 1,000 | 0,653 | 66% |
| Luna zero-shot | 0,947 | 1,000 | 100% |
| Luna + prompt phòng thủ | Chưa đo riêng trên SUB | 1,000 | 100% |

BoW + LogReg tụt F1 nhiều nhất, từ 0,788 xuống 0,286, giảm khoảng 0,502; BoW + MLP giảm khoảng 0,498. Test thông thường cùng phân phối với train và vẫn còn các mẫu gần giống nhau dù đã bỏ trùng tuyệt đối. Bộ khó cố ý thay đổi cách diễn đạt nên bộc lộ hạn chế này.

Nhóm khó nhất với BoW + LogReg là **spam giọng tự nhiên** (đúng 0/10), tiếp theo là **thao túng/lừa đảo** (1/8) và **né từ khóa** (1/6). BoW + MLP đúng 0/6 ở nhóm né từ khóa. Các mô hình embedding cũng chỉ đạt 20–40% ở nhóm spam giọng tự nhiên; riêng kNN chỉ đúng 25% ở nhóm nói về spam, cho thấy giống nội dung chưa chắc giống ý định.

Hai case cụ thể mà BoW + LogReg đoán sai:

1. **Case #11 — quảng cáo giả giọng người quen:** “Hey it's Jess from the gym … 3 months for the price of 1 …”. Nhãn thật là spam. Cả 10 tin trong nhóm spam giọng tự nhiên đều bị mô hình đoán sai, nên case này bị đoán thành ham. Cách mở đầu thân mật che khuất nội dung khuyến mãi; BoW chỉ cộng trọng số token, không biểu diễn đầy đủ mục đích chào mời.
2. **Case #14 — lời mời việc làm đáng ngờ:** “Hi, I'm working from home and making about £300 a day … no experience needed …”. Nhãn thật là spam nhưng cũng bị đoán thành ham. Dạng kể chuyện cá nhân khác các mẫu “WIN”, “FREE” quen thuộc; dấu hiệu hứa thu nhập cao cần được hiểu trong ngữ cảnh toàn câu.

Output lưu cho thấy hai phương án LLM đều đúng 50/50, bao gồm nhóm prompt injection. Vì điểm bằng nhau nên chưa thấy lợi ích tăng điểm của prompt phòng thủ trong lần chạy này. Vẫn nên tách SMS khỏi chỉ dẫn, kiểm tra định dạng nhãn và kết hợp một lớp phân loại độc lập. Bộ 50 case giúp phát hiện điểm yếu, không nên dùng để suy trực tiếp tỷ lệ lỗi ngoài đời hoặc vừa điều chỉnh mô hình vừa coi nó là test cuối.

## 5. Thử thách cá nhân

`7441 % 6 = 1`, nên thử thách là **đề #2: vocab 100 và 5.000 từ**, train lại và so F1 test.

Bản notebook nộp xây vocab chỉ từ TRAIN, vector hóa TRAIN/TEST bằng cùng vocab, rồi chạy `sklearn.linear_model.LogisticRegression(max_iter=1000)`. Output ghi:

| Vocab | Precision test | Recall test | F1 test |
| --- | ---: | ---: | ---: |
| 100 từ | 0,911 | 0,891 | 0,901 |
| 5.000 từ | 1,000 | 0,891 | 0,943 |

Vocab lớn tăng F1 khoảng 0,042. Recall giữ nguyên nhưng precision tăng, tức mô hình chặn nhầm ít tin ham hơn trong lần thử này. Có thêm từ giúp phân biệt tin nhắn tốt hơn, nhưng cũng tăng số chiều, bộ nhớ và nguy cơ học từ hiếm không ổn định.

**Giới hạn:** đề yêu cầu dùng mô hình Phần 1, còn code thử thách hiện dùng scikit-learn thay vì `train_logreg` tự viết. Các số trên mô tả đúng output đang có nhưng chưa phải thí nghiệm chỉ thay vocab trên cùng quy trình train của Phần 1. Cần chạy lại bằng `train_logreg`, giữ learning rate và số epoch giống nhau để hoàn thiện đúng yêu cầu; chưa coi kết quả scikit-learn là bằng chứng phần tự code đã đạt.

## 6. Phương án cho sản phẩm thật

10 triệu SMS/ngày tương đương `10.000.000 / 86.400 ≈ 115,74 tin/giây`. Theo lần chạy nộp, BoW + LogReg (p95 0,122 ms), BoW + MLP (2,405 ms) và MiniLM + MLP head (39,068 ms) đạt SLA. MiniLM + LogReg có p95 **52,502 ms** nên chưa đạt, dù p50 chỉ 30,322 ms. kNN và các LLM đều vượt 50 ms.

Nếu chọn một mô hình làm baseline vận hành, em chọn **BoW + MLP**: F1 toàn test 0,949, p95 2,405 ms và throughput batch khoảng 14.836 tin/giây. Mô hình có dư địa độ trễ lớn và không có phí API theo request. Đánh đổi là accuracy trên bộ khó chỉ 56%, nên chưa thể triển khai chỉ dựa vào F1 test thường; cần dữ liệu gần thực tế, kiểm tra false positive trên tin giao dịch và theo dõi lỗi.

MiniLM + MLP head là phương án bổ sung: đạt SLA và accuracy stress test 66%, nhưng F1 toàn test 0,917 thấp hơn BoW + MLP. Throughput batch khoảng 45,55 tin/giây chưa đủ lưu lượng trung bình nếu chỉ có một worker tương đương, nên cần thử tải, tối ưu và dự phòng lưu lượng đỉnh.

Phương án lai có thể dùng BoW xử lý tin chắc chắn, chuyển tin khó sang embedding, và dùng LLM để rà soát bất đồng bộ hoặc cho luồng được phép chậm hơn. Nếu gọi LLM đồng bộ thì SMS đó có thể vượt SLA; không thể mặc nhiên tuyên bố cả hệ lai đạt 50 ms. Ngưỡng chuyển tiếp cần chọn trên validation và kiểm tra hiệu chuẩn xác suất.

Với **300 triệu SMS/tháng** (30 ngày), độ dài tin và prompt giống lần đo:

- Luna zero-shot: `300.000.000 / 1.000 × 0,0081966667 ≈ 2.459 USD/tháng`.
- Luna few-shot: `300.000.000 / 1.000 × 0,0213966667 ≈ 6.419 USD/tháng`.

Đây là ngoại suy theo giá tham khảo, chưa tính retry, khác biệt độ dài tin, giá thực Router9 và hạ tầng. Chưa ước lượng Terra vì số chi phí đang sai.

## 7. AI log và điều đã học

Các lần dùng AI có bằng chứng trong cuộc trao đổi này:

| Phần | Yêu cầu đã gửi AI | AI hỗ trợ gì | Kiểm chứng và tình trạng |
| --- | --- | --- | --- |
| Kiểm tra toàn bài | Kiểm tra các phần và LLM có đi qua Router9 | Đọc code/output; chỉ ra lỗi giá Terra, hàm trùng, ghi log và phần thiếu | AI chạy lại 8 kiểm tra cốt lõi, tái tạo F1 ML/DL khớp notebook và kiểm tra API bằng mock. Chưa đối soát log Router9. |
| Router9 | Chuyển hẳn LLM sang Router9, sửa đơn giản | Chỉnh endpoint/key, bỏ hàm trùng, sửa parser và khôi phục kiểm tra | Commit `aa5fec4` có 8 test repo và 2 kiểm tra LLM giả lập đạt. Bản notebook nộp vẫn là bản cũ, cần đồng bộ. |
| Few-shot | Kiểm tra 4 ví dụ từ TRAIN | Đối chiếu nguồn, nhãn và prompt | Xác nhận 2 spam + 2 ham từ TRAIN, không trùng chính xác VAL/TEST; prompt dựng lại khớp output. |
| Tính tay | Làm 5 câu, viết công thức rồi thay số | Trình bày phép tính và điền HAND_CALC trong `src/` | Commit `6693ae5` khớp 5/5 hash; ảnh đã tải lên. Bản notebook nộp và JSON chưa cập nhật phần này. |
| Báo cáo | Điền báo cáo từ bài làm | Tổng hợp số liệu, chênh lệch, chi phí tháng và giới hạn | Đối chiếu bảng chính với JSON, stress test/thử thách với output notebook. Báo cáo có AI hỗ trợ soạn; người nộp cần đọc, xác nhận và tự giải thích. |

Qua kết quả, em thấy F1 cao trên test thường chưa chứng minh mô hình xử lý tốt tin nhắn đổi cách diễn đạt: BoW + LogReg đạt 0,924 trên test nhưng chỉ 0,286 ở bộ khó. Few-shot không tự động tăng chất lượng; lần chạy này F1 giữ nguyên trong khi chi phí tăng khoảng 2,61 lần. Khi chọn phương án thực tế, cần dùng đúng p95 của lần đo và kiểm tra dữ liệu, chi phí, thay vì chỉ nhìn tên hoặc kích thước model.
