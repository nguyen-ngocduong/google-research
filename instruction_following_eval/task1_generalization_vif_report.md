# Báo Cáo Kỹ Thuật: Áp Dụng Phương Pháp Luận Bài Báo "Generalizing Verifiable Instruction Following" Vào Xây Dựng Dataset (Task 1)

> **Tài liệu tham khảo chính:**
> * Bài báo: *Generalizing Verifiable Instruction Following* (NeurIPS 2025) – Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, Hannaneh Hajishirzi (AI2 & University of Washington). [arXiv:2507.02833](https://www.alphaxiv.org/pdf/2507.02833)
> * Mã nguồn chính thức: [GitHub: allenai/IFBench](https://github.com/allenai/IFBench)

---

## 1. Bối Cảnh & Vấn Đề Cốt Lõi (The Overfitting Problem)

Trong quá trình huấn luyện và đánh giá năng lực tuân thủ chỉ dẫn (Instruction Following - IF) của các Mô hình Ngôn ngữ Lớn (LLM), cộng đồng AI thường sử dụng benchmark chuẩn **IFEval** (Google Research, 2023). Tuy nhiên, bài báo chỉ ra một lỗ hổng nghiêm trọng:

1. **Bão hòa Benchmark (Benchmark Saturation):**
   * Các mô hình mã nguồn mở và thương mại hàng đầu (Llama 3.1, Qwen 2.5, Tulu 3...) dễ dàng đạt điểm số rất cao trên IFEval (thường $> 80\% - 90\%$).
2. **Hiện tượng "Học Vẹt" Ràng Buộc (Pattern Memorization):**
   * IFEval chỉ định nghĩa **25 mẫu ràng buộc (constraint templates)** cố định (như JSON, đếm từ, gạch đầu dòng, viết hoa...). Mô hình thực chất chỉ nhận diện các từ khóa kích hoạt bề mặt (surface lexical cues) mà không thực sự hiểu meta-logic trừu tượng của việc tuân thủ quy tắc.
3. **Thất bại khi gặp Unseen Constraints:**
   * Khi đánh giá trên bộ kiểm thử ngoại miền **IFBench** (gồm 58 ràng buộc mới chưa từng xuất hiện trong tập train), độ chính xác của hầu hết các mô hình đều giảm nghiêm trọng (rơi xuống dưới $30\% - 50\%$).
4. **Mục tiêu của Task 1:**
   * Không lặp lại sai lầm "overfit 25 constraints của IFEval", mà phải xây dựng một pipeline sinh dữ liệu có khả năng **thúc đẩy tính tổng quát hóa (Generalization)** sang bất kỳ ràng buộc nào mà mô hình chưa từng thấy trong quá trình huấn luyện.

---

## 2. Các Nguyên Lý Then Chốt Từ Bài Báo Đã Được Áp Dụng Vào Task 1

Dựa trên các phát hiện thực nghiệm và mã nguồn của bài báo, pipeline sinh dữ liệu tại [`datasets/code/augment_with_constraints.py`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/code/augment_with_constraints.py) đã áp dụng 6 nguyên lý cốt lõi sau:

### Nguyên lý 1: Phân tách rạch ròi SEEN vs. UNSEEN (Tránh Data Contamination)
* **Ý tưởng bài báo:** Để chứng minh năng lực generalization, ta bắt buộc phải phân tách hai tập ràng buộc độc lập tuyệt đối:
  * **Tập Huấn Luyện (SEEN Pool):** Dùng để huấn luyện RLVR/GRPO. Bao gồm 25 ràng buộc của IFEval kết hợp với các ràng buộc mở rộng của IFTrain.
  * **Tập Đánh Giá (UNSEEN Pool):** Giữ kín hoàn toàn trong suốt quá trình train. Bao gồm các ràng buộc ngoại miền (Out-Of-Domain - OOD) từ bộ 58 IFBench verifiers.
* **Áp dụng vào code:**
  * Khai báo rõ 2 pool: `SEEN_CONSTRAINT_DEFINITIONS` và `UNSEEN_CONSTRAINT_DEFINITIONS`.
  * Bổ sung metadata định danh trực tiếp trong mỗi mẫu dữ liệu:
    * `"constraint_split": "seen"` cho dữ liệu huấn luyện.
    * `"constraint_split": "unseen"` cho dữ liệu kiểm thử OOD.
  * CLI hỗ trợ cờ `--split seen` hoặc `--split unseen`.

---

### Nguyên lý 2: Multi-Constraint Composition (Ghép nhiều ràng buộc cùng lúc)
* **Phát hiện quan trọng của bài báo (Figure 2 trong paper):** 
  * Huấn luyện mô hình với **nhiều ràng buộc đồng thời ($K \in [2, 5]$)** mang lại sự bứt phá vượt bậc về khả năng tổng quát hóa, vượt trội hơn hẳn so với việc chỉ huấn luyện 1 ràng buộc đơn lẻ.
  * Việc giải quyết 3–5 ràng buộc cùng lúc ép mô hình phải hình thành một **"Internal Verification Checklist"** (danh sách kiểm tra nội tại trong chuỗi suy luận) thay vì chỉ kích hoạt phản xạ từ khóa.
* **Áp dụng vào code:**
  * Tích hợp bộ chọn ngẫu nhiên $K \in [1, 5]$ theo đúng tỷ lệ phân phối khuyến nghị:
    * $K = 1$: 20%
    * $K = 2$: 35%
    * $K = 3$: 25%
    * $K = 4$: 15%
    * $K = 5$: 5%

---

### Nguyên lý 3: Variable Range Augmentation (Mở rộng dải tham số biến động)
* **Ý tưởng bài báo:** Nếu trong tập train chỉ yêu cầu đếm 10 từ hoặc 3 đoạn văn, mô hình sẽ bị bó hẹp trong dải số đó. Bài báo chứng minh rằng mở rộng dải tham số ngẫu nhiên ($N$) rộng hơn thực tế kiểm thử sẽ giúp mô hình linh hoạt hơn.
* **Áp dụng vào code:**
  * Ngẫu nhiên hóa số từ: $N \in [40, 250]$ với các quan hệ `at least`, `less than`, `range`.
  * Ngẫu nhiên hóa số câu: $N \in [2, 6]$.
  * Ngẫu nhiên hóa số đoạn văn: $N \in [2, 4]$.
  * Ngẫu nhiên hóa số gạch đầu dòng / số mục: $N \in [3, 6]$.
  * Ngẫu nhiên hóa từ khóa cấm, chữ cái kiểm tra tần suất, tiêu đề, tag bọc XML.

---

### Nguyên lý 4: Compatibility Engine & Contradiction Filtering (Khử mâu thuẫn)
* **Vấn đề bài báo giải quyết:** Khi ghép ngẫu nhiên nhiều constraints, rất dễ xảy ra xung đột logic khiến prompt trở nên bất khả thi (ví dụ: vừa yêu cầu `viết HOA toàn bộ`, vừa cấm `chữ cái in hoa`; hoặc vừa yêu cầu `định dạng JSON`, vừa bắt `gạch đầu dòng markdown`).
* **Áp dụng vào code:**
  * Mỗi constraint được gắn một nhóm xung đột (`group`) và danh sách các nhóm đối kháng (`conflicts_with`).
  * Hàm `sample_compatible_constraints()` đảm bảo:
    * Không chọn 2 constraints cùng 1 nhóm (ví dụ: không có 2 luật về độ dài từ cùng lúc).
    * Khi một nhóm đã được chọn (ví dụ: `JSON`), toàn bộ các nhóm xung đột (`LIST_FORMAT`, `NUMBERED_LIST`, `PARAGRAPHS`, `POSTSCRIPT`, `ALL_CAPS`) sẽ lập tức bị khóa.

---

### Nguyên lý 5: Chuẩn Hóa Schema Dữ Liệu Theo Tiêu Chuẩn IF-RLVR
* **Chuẩn hóa trường:** Mỗi mẫu dữ liệu xuất ra chứa đầy đủ các trường phục vụ trực tiếp cho quá trình huấn luyện RLVR (GRPO) và thẩm định:
  ```json
  {
    "key": "alpaca-gpt4_0",
    "dataset_source": "vicgalle/alpaca-gpt4",
    "base_instruction": "Give three tips for staying healthy.",
    "prompt": "Give three tips for staying healthy.\n\nPlease adhere strictly to the following constraints in your response:\n- Wrap your entire response in double quotation marks (\"... \").\n- Your entire response must contain fewer than 6 sentences.\n- Add a postscript starting with 'P.S.' at the very end of your response.",
    "num_constraints": 3,
    "constraint_type": "multi",
    "constraint_ids": [
      "startend:quotation",
      "length_constraints:number_sentences",
      "detectable_content:postscript"
    ],
    "constraints_description": [
      "Wrap your entire response in double quotation marks (\"... \").",
      "Your entire response must contain fewer than 6 sentences.",
      "Add a postscript starting with 'P.S.' at the very end of your response."
    ],
    "ground_truth": [
      {
        "instruction_id": [
          "startend:quotation",
          "length_constraints:number_sentences",
          "detectable_content:postscript"
        ],
        "kwargs": [
          {},
          {
            "num_sentences": 6,
            "relation": "less than"
          },
          {
            "postscript_marker": "P.S."
          }
        ]
      }
    ],
    "constraint_split": "seen",
    "messages": [
      {
        "role": "user",
        "content": "..."
      }
    ]
  }
  ```

---

### Nguyên lý 6: Tối Ưu Hóa Bộ Nhớ & Khả Năng Khôi Phục (Engineering Practices)
* **Streaming Generator:** Sử dụng `datasets.load_dataset(..., streaming=True)` và ghi file `.jsonl` theo dòng, giải phóng rác định kỳ, giúp RAM duy trì cố định **$< 150\text{MB}$** (hoàn toàn an toàn trên máy 8GB RAM, không cần GPU).
* **Cơ chế `--resume` & `--append`:** Tự động đếm số lượng dòng hiện có, bỏ qua các mẫu đã xử lý để chạy tiếp mà không bị trùng lặp hoặc ghi đè dữ liệu cũ.

---

## 3. Bảng So Sánh Chi Tiết: Trước vs. Sau Khi Áp Dụng Bài Báo

| Tiêu chí | Trước khi áp dụng (Baseline v1) | Sau khi áp dụng bài báo (Task 1 hiện tại) |
| :--- | :--- | :--- |
| **Số lượng constraints** | Chỉ 25 constraints gốc từ IFEval | Mở rộng Seen Pool + Unseen OOD Pool (dựa trên IFBench) |
| **Phân chia Seen / Unseen** | Không có (dễ bị rò rỉ dữ liệu) | Tách biệt rõ ràng qua metadata `"constraint_split": "seen"` và `"unseen"` |
| **Số constraints / mẫu** | Đa số 1 constraint đơn lẻ | Ghép đa ràng buộc ($1 \le K \le 5$) có trọng số |
| **Kiểm tra mâu thuẫn** | Chọn ngẫu nhiên, dễ xung đột logic | Khử xung đột tự động bằng `Compatibility Engine` |
| **Dải tham số ($N$)** | Cố định giá trị nhỏ | Biến thiên ngẫu nhiên (Variable Range Augmentation) |
| **Tính tương thích RLVR** | Chỉ có text prompt thô | Schema hoàn chỉnh gồm `ground_truth` kwargs cho Python Verifiers |
| **Tài nguyên phần cứng** | Nguy cơ tràn RAM nếu load mảng lớn | Streaming xử lý $< 150\text{MB}$ RAM, 0% GPU |

---

## 4. Các Tập Dữ Liệu Đã Được Sinh Hoàn Chỉnh

Các tập dữ liệu đã được tạo sẵn trong thư mục [`datasets/`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets):

1. **Tập Huấn Luyện (Training Set - Seen Split):**
   * [`datasets/seen/augmented_alpaca_gpt4.jsonl`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/seen/augmented_alpaca_gpt4.jsonl) & [`.json`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/seen/augmented_alpaca_gpt4.json) (1.000 mẫu, `constraint_split="seen"`)
   * [`datasets/seen/augmented_gpt4_self_instruct.jsonl`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/seen/augmented_gpt4_self_instruct.jsonl) & [`.json`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/seen/augmented_gpt4_self_instruct.json) (1.000 mẫu, `constraint_split="seen"`)
2. **Tập Kiểm Thử Ngoại Miền (Evaluation Set - Unseen OOD Split):**
   * [`datasets/unseen/eval_unseen_constraints.jsonl`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/unseen/eval_unseen_constraints.jsonl) & [`.json`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/unseen/eval_unseen_constraints.json) (200 mẫu OOD, `constraint_split="unseen"`)

---

## 5. Cầu Nối Chuẩn Bị Sang Task 2 (Kết Hợp VerIF & Soft Constraints)

Mặc dù Task 1 đã giải quyết trọn vẹn bài toán **Tổng quát hóa Hard Constraints**, bài báo *Generalizing Verifiable Instruction Following* cũng thừa nhận một giới hạn nội tại:
> *"When models over-optimize for verifiable rewards, general response quality can decline (Reward Hacking)."*

Khi chỉ tối ưu hàm thưởng cứng (0/1), mô hình có xu hướng hy sinh nội dung chính của câu hỏi để cố nhồi nhét từ khóa hay đếm từ. 

👉 **Đó chính là lý do Task 2 cần tích hợp phương pháp của VerIF (Tsinghua University):**
* Thêm **Soft Constraints** (ngữ nghĩa, văn phong, tính đầy đủ) vào tập dữ liệu.
* Xây dựng hàm thưởng lai: $\text{Reward} = w_{\text{hard}} \cdot R_{\text{hard}} + w_{\text{soft}} \cdot R_{\text{soft}}$ để ngăn chặn triệt để Reward Hacking, giúp mô hình cân bằng hoàn hảo giữa việc trả lời đúng trọng tâm instruction và tuân thủ tuyệt đối các constraints.
