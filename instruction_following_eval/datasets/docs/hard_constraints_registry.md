# BẢNG ĐẶC TẢ VÀ DANH MỤC HARD CONSTRAINTS (IF-RLVR ENGINE)

Tài liệu này tổng hợp toàn bộ **29 quy tắc ràng buộc cứng Seen** được định nghĩa trong engine [`augment_with_constraints.py`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/code/augment_with_constraints.py) và liên kết tới **58 quy tắc Unseen IFBench** tại [`ifbench_unseen_constraints.md`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/docs/ifbench_unseen_constraints.md). 

Bộ quy tắc này được phân tách nghiêm ngặt thành **Seen Pool** (tập huấn luyện - 29 quy tắc IFTrain) và **Unseen Pool** (tập đánh giá Out-of-Distribution - 58 quy tắc IFBench) phục vụ cho bài toán huấn luyện **SFT → RLVR (GRPO)** theo tiêu chuẩn cao nhất của **IFEval** và **IFBench**.

---

## 1. TẬP SEEN CONSTRAINTS (29 Quy tắc - Chuẩn IFTrain / IFEval)
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
| `change_case:capital_word_frequency` | `CAPITAL_WORDS` | *"Your response must contain at least {n} capitalized words / words written in ALL CAPS."* | `{"capital_words": n}` | `ALL_CAPS`, `ALL_LOWER` |

---

### 1.6. Nhóm Ngôn Ngữ, Bảng Biểu & Tổ Hợp Cú Pháp (Advanced IF-RLVR)

| ID | Nhóm Xung Đột (Group) | Mẫu Ràng Buộc Sinh Ra (Prompt Template) | Barem Kiểm Tra (Kwargs cho Verifier) | Xung Đột Với (Conflicts With) |
|---|---|---|---|---|
| `language:response_language` | `LANGUAGE` | *"Your entire response must be written in {language}."* | `{"language": "..."}` | `ALL_CAPS`, `ALL_LOWER`, `END_SENTENCE` |
| `length_constraints:nth_paragraph_first_word` | `PARAGRAPHS_WORD` | *"Paragraph {nth} must start with the word '{word}'."* | `{"nth_paragraph": n, "first_word": "..."}` | `JSON`, `ONE_LINE` |
| `detectable_content:number_placeholders` | `PLACEHOLDERS` | *"Include at least {n} placeholders in brackets (e.g. [name], [address], [date]) in your response."* | `{"num_placeholders": n}` | `JSON` |
| `detectable_format:constrained_response` | `CONSTRAINED_RESP` | *"Your response must strictly be one of the following choices: {options}."* | `{"constrained_responses": [...]}` | `LENGTH_WORD`, `PARAGRAPHS`, `SECTIONS`, `LIST_FORMAT`, `NUMBERED_LIST`, `TABLE`, `TWO_RESPONSES` |
| `combination:two_responses` | `TWO_RESPONSES` | *"Provide two distinct responses to the prompt. Separate them with six asterisks: ******."* | `{"separator": "******"}` | `JSON`, `CONSTRAINED_RESP` |
| `combination:repeat_prompt` | `REPEAT_PROMPT` | *"First repeat the request or prompt verbatim, and then provide your response."* | `{}` | `JSON`, `CONSTRAINED_RESP` |
| `detectable_format:table_format` | `TABLE` | *"Organize your response or include a section formatted as a Markdown table (using \| Column 1 \| Column 2 \|)."* | `{}` | `JSON`, `LIST_FORMAT`, `NUMBERED_LIST`, `CONSTRAINED_RESP` |
| `punctuation:no_period` | `PUNCTUATION_PERIOD` | *"Do not use any period ('.') anywhere in your entire response."* | `{}` | `JSON` |

---

## 2. TẬP UNSEEN CONSTRAINTS (24 Quy Tắc Tuyển Chọn từ IFBench OOD)
> **Chi tiết toàn bộ 58 quy tắc OOD:** Xem tài liệu đặc tả đầy đủ tại [`ifbench_unseen_constraints.md`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/docs/ifbench_unseen_constraints.md).

Tập Unseen được tích hợp trong [`augment_with_constraints.py`](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/code/augment_with_constraints.py) gồm **24 quy tắc OOD tuyển chọn** với verifier 100% tự động bằng Python:

| ID | Nhóm Xung Đột (Group) | Mẫu Ràng Buộc Sinh Ra (Prompt Template) | Barem Kiểm Tra (Kwargs cho Verifier) | Ý Nghĩa Kiểm Tra & Thách Thức Kỹ Thuật |
|---|---|---|---|---|
| `count:unique_word_count` | `UNIQUE_WORDS` | *"Your response must contain at least {n} unique words."* ($n \in [25, 60]$) | `{"num_unique_words": n}` | Ép đa dạng hóa từ vựng, cấm lặp lại từ nhiều lần. |
| `format:parentheses` | `PARENTHESES` | *"Include at least {n} instances of nested parentheses ((like this)) in your response."* ($n \in [1, 3]$) | `{"num_nested": n}` | Phân tầng cấu trúc dấu ngoặc kép lồng nhau. |
| `words:palindrome` | `PALINDROME` | *"Include the palindromic words '{w1}' and '{w2}' in your response."* (vd: `radar`, `level`) | `{"palindromes": [...]}` | Chèn từ vựng đối xứng tự nhiên vào ngữ cảnh. |
| `sentence:increment` | `SENTENCE_INCREMENT` | *"Each subsequent sentence must contain more words than the sentence before it."* | `{}` | Ràng buộc độ dài lũy tiến: câu sau dài hơn câu trước. |
| `words:start_verb` | `START_VERB` | *"Every paragraph must start with a strong action verb."* | `{}` | Nhận thức ngữ pháp và loại từ (Action Verb đầu đoạn). |
| `count:punctuation` | `PUNCTUATION_DIVERSITY` | *"Use at least {n} different types of punctuation marks across your entire response."* | `{"num_punctuation_types": n}` | Độ phong phú dấu câu (chấm, phẩy, hỏi, than, chấm phẩy...). |
| `words:alphabet_loop` | `ALPHABET_LOOP` | *"Each word must start with the next letter of the alphabet in sequential order, looping back to 'a'."* | `{}` | Dây chuyền chữ cái đầu theo thứ tự A $\rightarrow$ Z liên tục. |
| `words:prime_lengths` | `PRIME_LENGTHS` | *"Every word must have a length (character count) that is a prime number (2, 3, 5, 7, 11...)."* | `{}` | Ép độ dài từ vựng là số nguyên tố khắt khe. |
| `format:nested_quotes` | `NESTED_QUOTES` | *"Include quotes within quotes within quotes, at least 3 levels deep."* | `{}` | Xử lý trích dẫn đa tầng xen kẽ dấu ngoặc đơn và ngoặc kép. |
| `count:numbers_count` | `NUMBERS_COUNT` | *"Include exactly {n} numbers (written as digits) in your entire response."* | `{"num_numbers": n}` | Kiểm soát chính xác số lượng thực thể số học dạng chữ số. |
| `format:title_case` | `TITLE_CASE` | *"Write your entire response in Title Case (capitalize the first letter of every major word)."* | `{}` | Viết hoa tiêu đề chuẩn mực cho toàn bài viết. |
| `sentence:last_word_first_next` | `LAST_FIRST_CHAIN` | *"The last word of each sentence must become the first word of the very next sentence."* | `{}` | Nối từ dây chuyền giữa các câu liên tiếp (Anadiplosis). |
| `words:no_consecutive_first_letter` | `NO_CONSECUTIVE_LETTER` | *"No two consecutive words can share the same first letter."* | `{}` | Cấm điệp âm giữa 2 từ đứng cạnh nhau. |
| `words:limited_repeat` | `LIMITED_REPEAT` | *"Do not repeat any word more than {n} times across your entire response."* | `{"max_repeats": n}` | Triệt tiêu hiện tượng lặp từ và suy thoái văn phong. |
| `sentence:word_count_step` | `SENTENCE_STEP` | *"Each subsequent sentence must contain exactly {step} more words than the sentence before it."* | `{"step": step}` | Cấp số cộng độ dài câu chính xác. |
| `format:date_format_list` | `DATE_FORMAT` | *"Include at least {n} dates formatted strictly as YYYY-MM-DD separated by commas."* | `{"min_dates": n}` | Xuất dữ liệu ngày tháng theo chuẩn ISO-8601. |
| `format:csv_format` | `CSV_FORMAT` | *"Format your entire response as CSV data with exactly {rows} data rows and columns: ID, Name, Category, Value."* | `{"num_rows": rows}` | Sinh bảng dữ liệu CSV đúng cú pháp dấu phẩy. |
| `format:output_template` | `OUTPUT_TEMPLATE` | *"Use this exact template: My Answer: [answer] My Conclusion: [conclusion] Future Outlook: [outlook]"* | `{}` | Tuân thủ biểu mẫu điền thông tin cố định. |
| `words:paragraph_last_first_match` | `PARA_MATCH` | *"Each paragraph in your response must end with the exact same word it started with."* | `{}` | Cấu trúc vòng tròn khép kín cho từng đoạn văn bản. |
| `words:conjunction_count` | `CONJUNCTION_COUNT` | *"Use at least {n} different coordinating conjunctions (from: and, but, for, nor, or, so, yet)."* | `{"min_conjunctions": n}` | Đa dạng hóa liên từ đẳng lập trong lập luận. |
| `format:indent_stairs` | `INDENT_STAIRS` | *"Create a staircase effect by incrementally indenting each new line with 2 additional spaces."* | `{}` | Thụt đầu dòng dạng bậc thang lũy tiến. |
| `format:special_bullet` | `SPECIAL_BULLET` | *"Format your response as a list using '-> ' instead of standard bullet points for every item."* | `{}` | Thay thế định dạng danh sách mặc định bằng ký tự chỉ định. |
| `manipulation:no_whitespace` | `NO_WHITESPACE` | *"Your response must not contain any whitespace characters (no spaces, no tabs, no newlines)."* | `{}` | Loại bỏ toàn bộ khoảng trắng, nối liền văn bản. |
| `sentence:symbol_end` | `SYMBOL_END` | *"Every sentence in your response must end with the symbol combination '!*'."* | `{}` | Đính kèm tổ hợp ký tự đặc biệt ở cuối mỗi câu. |

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
  │  TASK 1: Ghép Hard Constraints                          │
  │  - Train Set: Dùng SEEN POOL (29 quy tắc chuẩn IFTrain) │
  │  - OOD Test Set: Dùng UNSEEN POOL (Kho 58 quy tắc       │
  │    chuẩn IFBench OOD)                                   │
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

