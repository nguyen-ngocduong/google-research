# BẢNG ĐẶC TẢ VÀ DANH MỤC HARD CONSTRAINTS (IF-RLVR ENGINE)

Tài liệu này tổng hợp toàn bộ **26 quy tắc ràng buộc cứng (Hard Constraints)** được định nghĩa trong engine [`augment_with_constraints.py`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/augment_with_constraints.py). 

Bộ quy tắc này được phân tách nghiêm ngặt thành **Seen Pool** (tập huấn luyện) và **Unseen Pool** (tập đánh giá Out-of-Distribution - OOD) phục vụ cho bài toán huấn luyện **SFT → RLVR (GRPO)** theo tiêu chuẩn của **IFEval** và **IFBench**.

---

## 1. TẬP SEEN CONSTRAINTS (20 Quy tắc - Chuẩn IFEval / IFTrain)
> **Mục đích:** Dùng cho tập dữ liệu huấn luyện RLVR. Giúp mô hình rèn luyện khả năng tuân thủ các quy tắc định dạng và hình thức cơ bản.

### 1.1. Nhóm Từ Khóa & Chữ Cái (Keywords & Letters)

| ID | Nhóm Xung Đột (Group) | Mẫu Ràng Buộc Sinh Ra (Prompt Template) | Barem Kiểm Tra (Kwargs cho Verifier) | Xung Đột Với (Conflicts With) |
|---|---|---|---|---|
| `keywords:existence` | `KEYWORD_EXIST` | *"Include the keyword '{word}' in your response."* (hoặc 2 từ) | `{"keywords": ["..."]}` | `ALL_CAPS`, `ALL_LOWER` |
| `keywords:frequency` | `KEYWORD_FREQ` | *"The word '{word}' must appear {at least / at most / exactly} {n} times in your response."* | `{"keyword": "...", "frequency": n, "relation": "..."}` | `ALL_CAPS`, `ALL_LOWER` |
| `keywords:forbidden_words` | `FORBIDDEN_WORDS` | *"Do not include any of the following words in your response: {w1}, {w2}, {w3}."* | `{"forbidden_words": ["...", ...]}` | - |
| `keywords:letter_frequency` | `LETTER_FREQ` | *"The letter '{char}' should appear {at least / at most} {n} times in your entire response."* | `{"letter": "...", "let_frequency": n, "let_relation": "..."}` | - |

---

### 1.2. Nhóm Giới Hạn Độ Dài (Length Constraints)

| ID | Nhóm Xung Đột (Group) | Mẫu Ràng Buộc Sinh Ra (Prompt Template) | Barem Kiểm Tra (Kwargs cho Verifier) | Xung Đột Với (Conflicts With) |
|---|---|---|---|---|
| `length_constraints:number_words` | `LENGTH_WORDS` | *"Your response should contain {at least / fewer than / between} {n} words."* | `{"num_words": n, "relation": "..."}` | - |
| `length_constraints:number_sentences` | `LENGTH_SENTENCES` | *"Your entire response must contain {exactly / at least / fewer than} {n} sentences."* | `{"num_sentences": n, "relation": "..."}` | `JSON` |
| `length_constraints:number_paragraphs` | `LENGTH_PARAGRAPHS` | *"Your response must contain exactly {n} paragraphs, separated by double newlines."* | `{"num_paragraphs": n}` | `JSON`, `ONE_LINE` |

---

### 1.3. Nhóm Định Dạng Danh Sách (Lists & Bullets)

| ID | Nhóm Xung Đột (Group) | Mẫu Ràng Buộc Sinh Ra (Prompt Template) | Barem Kiểm Tra (Kwargs cho Verifier) | Xung Đột Với (Conflicts With) |
|---|---|---|---|---|
| `detectable_format:number_bullet_lists` | `BULLETS` | *"Format your response with exactly {n} bullet points (using * or -)."* | `{"num_bullets": n}` | `JSON`, `NUMBERED_LIST` |
| `format:numbered_list` | `NUMBERED_LIST` | *"Format your entire response as a numbered list with exactly {n} items."* | `{"num_items": n}` | `JSON`, `BULLETS` |

---

### 1.4. Nhóm Cấu Trúc Khối Văn Bản & Bao Đóng (Structure & Wrapper)

| ID | Nhóm Xung Đột (Group) | Mẫu Ràng Buộc Sinh Ra (Prompt Template) | Barem Kiểm Tra (Kwargs cho Verifier) | Xung Đột Với (Conflicts With) |
|---|---|---|---|---|
| `detectable_format:json_format` | `JSON` | *"Your entire response must be a single valid JSON object."* | `{}` | `BULLETS`, `NUMBERED_LIST`, `ALL_CAPS`, `ALL_LOWER`, `XML_TAG`, `PUNCTUATION`, `LENGTH_SENTENCES`, `LENGTH_PARAGRAPHS` |
| `detectable_format:title` | `TITLE` | *"Your response must start with a title enclosed in brackets (e.g., [Title])."* | `{}` | `JSON` |
| `detectable_format:multiple_sections` | `SECTIONS` | *"Organize your response into exactly {n} sections, each starting with 'Section 1', 'Section 2', etc."* | `{"num_sections": n}` | `JSON`, `LENGTH_SENTENCES` |
| `detectable_format:number_highlighted_sections` | `HIGHLIGHTS` | *"Highlight exactly {n} key terms or phrases using markdown bold (**word**)."* | `{"num_highlights": n}` | `JSON` |
| `detectable_content:postscript` | `POSTSCRIPT` | *"Add a postscript starting with 'P.S.' at the very end of your response."* | `{"postscript_marker": "P.S."}` | `JSON` |
| `startend:end_checker` | `END_CHECKER` | *"End your entire response with the exact phrase: '{end_phrase}'."* | `{"end_phrase": "..."}` | `JSON` |
| `startend:quotation` | `QUOTATION` | *"Wrap your entire response in double quotation marks (\"... \")."* | `{}` | `JSON` |
| `punctuation:no_comma` | `PUNCTUATION` | *"Do not use any commas (',') anywhere in your entire response."* | `{}` | `JSON` |
| `format:xml_wrapper` | `XML_TAG` | *"Wrap your entire answer inside `<response>` and `</response>` tags."* | `{}` | `JSON` |

---

### 1.5. Nhóm Chữ Hoa / Chữ Thường (Case Modifications)

| ID | Nhóm Xung Đột (Group) | Mẫu Ràng Buộc Sinh Ra (Prompt Template) | Barem Kiểm Tra (Kwargs cho Verifier) | Xung Đột Với (Conflicts With) |
|---|---|---|---|---|
| `change_case:english_capital` | `ALL_CAPS` | *"Your entire response must be written in ALL CAPS."* | `{}` | `ALL_LOWER`, `JSON`, `KEYWORD_EXIST`, `KEYWORD_FREQ` |
| `change_case:english_lowercase` | `ALL_LOWER` | *"Your entire response must be written in all lowercase letters."* | `{}` | `ALL_CAPS`, `JSON`, `TITLE`, `KEYWORD_EXIST`, `KEYWORD_FREQ` |

---

## 2. TẬP UNSEEN CONSTRAINTS (6 Quy tắc - Dựa trên IFBench OOD)
> **Mục đích:** Tuyệt đối không xuất hiện trong tập Train. Dùng riêng cho tập Test OOD (`eval_unseen_constraints.json`) nhằm đánh giá năng lực **Generalization** của mô hình sau khi train RLVR (Task 1).

| ID | Nhóm Xung Đột (Group) | Mẫu Ràng Buộc Sinh Ra (Prompt Template) | Barem Kiểm Tra (Kwargs cho Verifier) | Ý Nghĩa Kiểm Tra & Thách Thức Kỹ Thuật |
|---|---|---|---|---|
| `count:unique_word_count` | `UNIQUE_WORDS` | *"Your response must contain at least {n} unique words."* ($n \in [25, 60]$) | `{"num_unique_words": n}` | Ép mô hình đa dạng hóa từ vựng, cấm lặp lại từ nhiều lần. |
| `format:parentheses` | `PARENTHESES` | *"Include at least {n} instances of nested parentheses ((like this)) in your response."* ($n \in [1, 3]$) | `{"num_nested": n}` | Kiểm tra cú pháp phân tầng dấu ngoặc kép lồng nhau. |
| `words:palindrome` | `PALINDROME` | *"Include the palindromic words '{w1}' and '{w2}' in your response."* (Ví dụ: `radar`, `kayak`, `level`) | `{"palindromes": ["...", "..."]}` | Ép mô hình tích hợp từ ngữ đối xứng theo đúng ngữ cảnh câu. |
| `sentence:increment` | `SENTENCE_INCREMENT` | *"Each subsequent sentence in your response must contain more words than the sentence before it."* | `{}` | Ràng buộc cấu trúc lũy tiến: câu sau bắt buộc phải dài hơn câu trước. |
| `words:start_verb` | `START_VERB` | *"Every paragraph in your response must start with a strong action verb."* | `{}` | Kiểm tra ngữ pháp phân loại từ (Part-of-Speech: Action Verb ở đầu đoạn). |
| `count:punctuation` | `PUNCTUATION_DIVERSITY` | *"Use at least {n} different types of punctuation marks across your entire response."* ($n \in [4, 7]$) | `{"num_punctuation_types": n}` | Kiểm tra độ phong phú dấu câu (chấm, phẩy, hỏi, than, chấm phẩy, gạch ngang, hai chấm...). |

---

## 3. CƠ CHẾ KIỂM SOÁT XUNG ĐỘT (COMPATIBILITY ENGINE)

Trong `augment_with_constraints.py`, thuật toán lấy mẫu ràng buộc hoạt động theo cơ chế **Constrained Randomized Graph**:
1. **Lấy ngẫu nhiên $k$ luật:** Mỗi prompt được gán từ 1 đến 5 ràng buộc.
2. **Kiểm tra ma trận xung đột:** Trước khi thêm một luật vào prompt:
   * Kiểm tra xem nhóm của luật đó (`group`) đã có mặt trong danh sách cấm chưa.
   * Cập nhật toàn bộ các nhóm bị luật đó khắc chế (`conflicts_with`) vào danh sách cấm (`forbidden_groups`).
3. **Ý nghĩa khoa học:** Đảm bảo **Không gian nghiệm (Solution Space)** luôn luôn tồn tại. Mô hình không bao giờ bị rơi vào tình trạng "nhiệm vụ bất khả thi" (như vừa bắt JSON vừa bắt chữ in hoa toàn bộ).

---

## 4. TỔNG HỢP VAI TRÒ TRONG PIPELINE SEN-32B V2

```text
       [Base SFT Instruction (Alpaca / Self-Instruct)]
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  TASK 1: Ghép Hard Constraints (Kho 26 quy tắc này)     │
  │  - Train Set: Dùng SEEN POOL (20 quy tắc)               │
  │  - OOD Test Set: Dùng UNSEEN POOL (6 quy tắc IFBench)   │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  TASK 2: Ghép Soft Constraints & VerIF Reward Spec       │
  │  - LLM sinh Semantic & Tone constraints                 │
  │  - Áp dụng Gated Hybrid Reward để chống Reward Hacking  │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  TRIỂN KHAI SERVER GPU: RLVR / GRPO TRAINING            │
  │  - Hard Verifier chấm Code (nhanh, chính xác 100%)      │
  │  - LLM Judge chấm Soft Verifier                         │
  └─────────────────────────────────────────────────────────┘
```
