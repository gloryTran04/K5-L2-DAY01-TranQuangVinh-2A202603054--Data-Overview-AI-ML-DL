# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python: 3.13.16 / PyTorch: 2.11.0+cu130 / Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về cả bức ảnh. Record không có `bbox_xyxy` hay `polygon_xy`, tức là không chỉ vào vị trí nào trong ảnh. Mỗi `sample_id` chỉ nhận một danh sách 5 lớp xếp hạng (cab, minibus, police_van, recreational_vehicle, streetcar). Ảnh `traffic` có hàng chục xe buýt và ô tô, nhưng model chỉ trả về một nhãn "cab" cho cả khung hình, không gắn với chiếc xe cụ thể nào.

3. Class list đi kèm taxonomy/checkpoint, không phải do model tự nghĩ ra. `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K nên chỉ chọn được trong 1000 lớp cố định đó. Bằng chứng: ImageNet-1K không có lớp kiểu "cảnh đường phố/giao thông", nên model phải chọn lớp gần nhất là "cab". Tương tự, ảnh `kitchen` không có lớp "bếp" nên top-1 thành "gong" (0.420191).

4. Vì cùng một `class_id` hoặc `class_name` có thể mang nghĩa khác nhau ở các taxonomy khác nhau. Ngay trong output bài lab: lớp "bàn ăn" là `class_id=532`, `class_name=dining_table` trong ImageNet-1K (record `kitchen` rank 2), nhưng lại là `class_id=60`, `class_name=dining table` trong COCO-80 (file detection, sample `dining`). `class_id=0` là "person" trong COCO-80 nhưng là một lớp hoàn toàn khác trong ImageNet-1K. Ghi đủ ba trường thì mới tra ngược và ghép dữ liệu không nhầm.

5. Guideline cần nói rõ:
   - Chọn lớp theo chủ thể chính (vật chiếm diện tích lớn nhất / nổi bật nhất) hay theo bối cảnh chung của ảnh.
   - Thứ tự ưu tiên khi nhiều lớp cùng hợp lệ. Ví dụ ảnh `traffic` có cả xe buýt, ô tô, người đi bộ: chọn "bus" vì chiếm nhiều nhất, hay một lớp cảnh.
   - Làm gì khi không lớp nào trong taxonomy phù hợp: dùng lớp "khác" nếu có, hoặc escalation, không tự chọn lớp "gần giống".
   - Dự án là single-label (một lớp) hay multi-label.

6. `score` chỉ là mức độ "tin" của model vào dự đoán của chính nó, chưa được con người kiểm chứng. Score cao vẫn có thể sai: ảnh `kitchen` bị đoán là "gong" với score 0.420191, cao gấp gần 6 lần lớp thứ hai. Ảnh `traffic` có top-1 chỉ 0.510915, cho thấy model khá phân vân. Nếu lấy prediction làm ground truth, model sẽ học lại chính lỗi của nó. Ground truth phải do người gán theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Ảnh `kitchen` rộng 640 px, cao 427 px, gốc (0, 0) ở góc trên-trái.
   - Góc trên-trái của hộp ở điểm (385.33, 69.24): cách mép trái ảnh khoảng 385 px, cách mép trên khoảng 69 px, ngay trên đỉnh đầu người đầu bếp.
   - Góc dưới-phải ở điểm (498.92, 348.92): cách mép trái khoảng 499 px, cách mép trên khoảng 349 px, ngang chân người.
   - Hộp rộng 113.58 px (= 498.92 − 385.33) và cao 279.68 px (= 348.92 − 69.24). Hộp cao gấp khoảng 2,5 lần chiều rộng, đúng dáng một người đang đứng quay lưng.

3. Ở ngưỡng 0.20: 17 prediction (person ×2, bowl ×5, oven ×2, cup ×2, spoon ×3, potted plant ×1, dining table ×1, bottle ×1).
   Ở ngưỡng 0.60: 6 prediction (person ×2, bowl ×2, oven ×2).

4. Khi hạ ngưỡng, số hộp tăng và nhiều hộp có score thấp, nên reviewer phải:
   - kiểm tra từng hộp mới có đúng là vật thật không (loại false positive);
   - kiểm tra hộp trùng lặp trên cùng một vật;
   - kiểm tra lớp có đúng không. Ví dụ hộp bowl 0.38 và cup 0.38 trên ảnh `kitchen` nằm chồng lên vùng cánh tay và các vật khác trên bàn, rất dễ nhầm.

   Ngưỡng thấp cho nhiều gợi ý hơn nhưng tăng khối lượng kiểm tra.

5. Quy tắc box: Vẽ một hộp chữ nhật sát nhất bao trọn toàn bộ phần nhìn thấy của mỗi vật thể thuộc taxonomy. Không chừa nền thừa, không cắt mất phần nhìn thấy, không đoán phần bị che. Mỗi vật một hộp riêng, kể cả khi nhiều vật cùng lớp đứng sát nhau.

6. Vật bị cắt mép: hộp person 0.61 (`bbox_xyxy`=[0.12, 263.18, 61.55, 310.78]) ở mép trái ảnh `kitchen`. Trên PNG chỉ thấy một cánh tay thò vào khung hình, phần còn lại của người nằm ngoài ảnh.
   **Mình đưa vào guideline**, vì đây là tình huống lặp lại ở rất nhiều ảnh và quyết định được bằng một quy tắc chung: "Vật bị cắt mép ảnh vẫn gán nhãn nếu xác định được lớp; hộp chỉ bao phần nhìn thấy và chạm mép ảnh."
   Chỉ escalation khi phần nhìn thấy quá nhỏ đến mức không xác định được là vật gì.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]

2. Box của người này ([385.45, 66.44, 498.02, 348.58]) chỉ là hình chữ nhật, nên bao luôn cả nền tường, nồi treo phía sau, khoảng trống giữa hai chân và hai bên khuỷu tay. Polygon 348 điểm bám theo đường viền cơ thể: đầu, vai, hai cánh tay chống hông, tạp dề và khe hở giữa hai chân. Năm điểm đầu đi lần từng pixel quanh đỉnh đầu (y = 70 → 72). Nhờ vậy polygon cho biết **hình dạng thật** và pixel nào thuộc vật, pixel nào là nền.

3. `instance_id` dùng để phân biệt **từng đối tượng riêng lẻ** trong bộ output của bài lab. Ví dụ `kitchen-001` và `kitchen-009` đều là person (`class_id`=0), nhưng là hai người/vật khác nhau với hai polygon riêng.
   - **Không phải class ID:** class ID cho biết "là loại gì" (0 = person); nhiều instance chung một class ID.
   - **Không phải tracking ID:** bài lab chỉ có ảnh tĩnh, `instance_id` chỉ là số thứ tự `<sample_id>-NNN` theo thứ tự output. Nó không theo dõi cùng một vật qua nhiều khung hình, và chạy lại có thể đánh số khác.

4. **Quy tắc vẽ biên mask:** Polygon bám sát đường biên **phần nhìn thấy** của từng instance, không tô phần bị vật khác che. Khi hai vật chạm hoặc chồng nhau, mỗi pixel chỉ thuộc **một** instance (vật nằm phía trước). Không gộp nhiều vật cùng lớp vào một mask.

5. Trên PNG, mask `dining table` (`kitchen-005`, score 0.578416) phủ gần hết mặt bàn gỗ, có vẻ tràn lên các vật đặt trên bàn (cán lăn bột, đĩa bánh) và cả vùng xô kim loại dưới gầm bàn. Đồng thời mask này chồng lên các mask `bowl` nằm trên bàn (`kitchen-002`, `kitchen-003`, `kitchen-006`).
   **Mình dùng guideline**, vì trường hợp "vật đặt trên bàn" xảy ra ở rất nhiều ảnh và có thể quy định chung: "Vật đặt trên/dưới bàn là instance riêng; pixel của vật đó không thuộc mask bàn."
   Riêng các vật không có lớp trong COCO-80 (ví dụ xô kim loại) mà guideline chưa nói có tách khỏi bàn hay không thì escalation cho Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho **cả ảnh**: `class_id` + `class_name` + `taxonomy_name` | `kitchen` top-1 = "gong" (0.420191), sai rõ với ảnh nhà bếp. `traffic` top-1 = "cab" (0.510915) trong khi chủ thể nổi bật là xe buýt và ô tô; ImageNet-1K không có lớp "cảnh giao thông" | Xem toàn ảnh, chọn một lớp theo quy tắc chủ thể chính; escalation nếu không lớp nào phù hợp | Lớp có nằm trong taxonomy và đúng quy tắc chọn chủ thể không; các annotator có chọn thống nhất trên ảnh tương tự không |
| Phát hiện vật thể | Mỗi vật **một hộp** `bbox_xyxy` (pixel, gốc trên-trái) + lớp | Hàng loạt nồi, chảo treo trên tường không có hộp nào (COCO-80 không có lớp nồi/chảo). Hộp person 0.61 chỉ là cánh tay bị cắt mép trái. Bàn gỗ lớn không được detect ở 0.35. Hộp oven 0.63 bên phải bao cả bồn rửa, khó quyết định | Vẽ hộp sát cho mọi vật thuộc taxonomy, kể cả vật model bỏ sót; sửa hoặc xóa hộp sai lớp, sai vị trí | Thiếu hộp, hộp thừa, sai lớp, hộp lỏng/chặt, hộp trùng; soát kỹ các hộp score thấp |
| Instance segmentation | Mỗi instance **một polygon** `polygon_xy` + lớp + `instance_id` riêng | Mask `dining table` (`kitchen-005`) tràn sang vật trên bàn và vùng dưới gầm bàn. `spoon` 0.49 (`kitchen-008`) thực ra trông như muôi/vá treo, là lớp gần nhất trong COCO-80. Seg chỉ ra 1 oven (0.49) trong khi detection ra 2 oven | Vẽ polygon bám biên phần nhìn thấy; tách riêng các vật chạm nhau; không tô phần bị che | Phóng to kiểm tra biên; vùng chồng lấn giữa các mask; mỗi vật đúng một instance và đúng lớp |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai (CC BY 2.0) do notebook tải và kiểm tra checksum. Không tải lên ảnh cá nhân, khuôn mặt, biển số xe ngoài phạm vi, dữ liệu khách hàng hay dữ liệu nội bộ. Không ghi họ tên, MSSV hoặc thông tin cá nhân vào báo cáo và output.

2. Mình sẽ dừng ngay, không upload hoặc commit dữ liệu đó, và báo cho Lab Coach (hoặc giảng viên chưa xử lý được) trước khi làm tiếp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
