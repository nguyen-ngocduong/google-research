# Danh Sách 25 Ràng Buộc (Verifiable Instructions) trong IFEval

Tài liệu tổng hợp và phân tích 25 loại ràng buộc khách quan (verifiable constraints) được định nghĩa trong mã nguồn `google-research/instruction_following_eval` (file `instructions_registry.py`).

---

## I. Tổng Quan 9 Nhóm Ràng Buộc (25 Constraints)

| STT | Nhóm | Mã Ràng Buộc (`instruction_id`) | Lớp Kiểm Tra (Checker Class) | Ý Nghĩa / Mục Tiêu |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **Keywords** | `keywords:existence` | `KeywordChecker` | Chứa một hoặc nhiều từ khóa chỉ định |
| 2 | **Keywords** | `keywords:frequency` | `KeywordFrequencyChecker` | Số lần xuất hiện của từ khóa ($>, <, =$ $N$ lần) |
| 3 | **Keywords** | `keywords:forbidden_words` | `ForbiddenWords` | Không được chứa danh sách từ khóa cấm |
| 4 | **Keywords** | `keywords:letter_frequency` | `LetterFrequencyChecker` | Số lần xuất hiện của một chữ cái cụ thể |
| 5 | **Language** | `language:response_language` | `ResponseLanguageChecker` | Phản hồi hoàn toàn bằng 1 ngôn ngữ chỉ định |
| 6 | **Length** | `length_constraints:number_words` | `NumberOfWords` | Giới hạn số lượng từ (tối thiểu, tối đa) |
| 7 | **Length** | `length_constraints:number_sentences` | `NumberOfSentences` | Giới hạn số lượng câu hoàn chỉnh |
| 8 | **Length** | `length_constraints:number_paragraphs` | `ParagraphChecker` | Giới hạn số lượng đoạn văn (tách bởi `\n\n`) |
| 9 | **Length** | `length_constraints:nth_paragraph_first_word` | `ParagraphFirstWordCheck` | Từ đầu tiên của đoạn thứ $N$ phải là từ cụ thể |
| 10 | **Content** | `detectable_content:number_placeholders` | `PlaceholderChecker` | Có ít nhất $N$ placeholder dạng `[...]` |
| 11 | **Content** | `detectable_content:postscript` | `PostscriptChecker` | Có phần tái bút bắt đầu bằng `P.S.` |
| 12 | **Format** | `detectable_format:number_bullet_lists` | `BulletListChecker` | Số lượng gạch đầu dòng (`* ` hoặc `- `) |
| 13 | **Format** | `detectable_format:number_highlighted_sections` | `HighlightSectionChecker` | Số phần được làm nổi bật bằng markdown `*...*` |
| 14 | **Format** | `detectable_format:multiple_sections` | `SectionChecker` | Chia bài thành các phần: `Section X`, `PARAGRAPH X` |
| 15 | **Format** | `detectable_format:json_format` | `JsonFormat` | Toàn bộ phản hồi nằm trong định dạng JSON hợp lệ |
| 16 | **Format** | `detectable_format:title` | `TitleChecker` | Có tiêu đề nằm trong ngoặc nhọn kép `<<Title>>` |
| 17 | **Format** | `detectable_format:constrained_response` | `ConstrainedResponseChecker` | Chọn câu trả lời từ danh sách cố định |
| 18 | **Combination** | `combination:two_responses` | `TwoResponsesChecker` | Đưa 2 câu trả lời khác nhau, ngăn bởi `******` |
| 19 | **Combination** | `combination:repeat_prompt` | `RepeatPromptThenAnswer` | Nhắc lại nguyên văn prompt trước khi trả lời |
| 20 | **Start / End** | `startend:end_checker` | `EndChecker` | Câu kết thúc phải là một câu chỉ định chính xác |
| 21 | **Start / End** | `startend:quotation` | `QuotationChecker` | Bọc toàn bộ phản hồi bằng dấu ngoặc kép `"..."` |
| 22 | **Case** | `change_case:english_capital` | `CapitalLettersEnglishChecker` | Toàn bộ phản hồi viết HOA (không có chữ thường) |
| 23 | **Case** | `change_case:english_lowercase` | `LowercaseLettersEnglishChecker` | Toàn bộ phản hồi viết THƯỜNG (không có chữ hoa) |
| 24 | **Case** | `change_case:capital_word_frequency` | `CapitalWordFrequencyChecker` | Số lượng từ viết hoa toàn bộ trong bài |
| 25 | **Punctuation** | `punctuation:no_comma` | `CommaChecker` | Cấm tuyệt đối không dùng dấu phẩy `,` |

---

## II. Chi Tiết Từng Ràng Buộc & Ví Dụ Thực Tế

### 1. Nhóm Keywords (4 ràng buộc)
* **`keywords:existence`**:
  * *Mục tiêu:* Kiểm tra sự hiện diện của một hoặc nhiều từ khóa.
  * *Ví dụ:* `"The email must include the keywords 'correlated' and 'experiencing'."`
* **`keywords:frequency`**:
  * *Mục tiêu:* Kiểm tra số lần xuất hiện của từ (quan hệ: `at least`, `less than`, `equal to`).
  * *Ví dụ:* `"Make sure to use the word 'clearly' at least 2 times."`
* **`keywords:forbidden_words`**:
  * *Mục tiêu:* Kiểm tra danh sách từ cấm, không từ nào được xuất hiện.
  * *Ví dụ:* `"Do not include the following keywords: field, thanks, issue, collaborator."`
* **`keywords:letter_frequency`**:
  * *Mục tiêu:* Kiểm tra số lần xuất hiện của một ký tự đơn trong toàn văn bản.
  * *Ví dụ:* `"In your entire response, the letter t should appear at most once."`

### 2. Nhóm Language (1 ràng buộc)
* **`language:response_language`**:
  * *Mục tiêu:* Sử dụng thư viện `langdetect` để xác minh văn bản được viết đúng ngôn ngữ yêu cầu.
  * *Ví dụ:* `"Write a rubric ... only using the Punjabi language, no other language is allowed."`

### 3. Nhóm Length Constraints (4 ràng buộc)
* **`length_constraints:number_words`**:
  * *Mục tiêu:* Đếm tổng số từ (word count) theo ràng buộc $\ge N$, $\le N$,...
  * *Ví dụ:* `"Write a 300+ word summary of the wikipedia page..."`
* **`length_constraints:number_sentences`**:
  * *Mục tiêu:* Tách câu bằng `nltk.sent_tokenize` và đếm số câu.
  * *Ví dụ:* `"Your response should contain fewer than 6 sentences."`
* **`length_constraints:number_paragraphs`**:
  * *Mục tiêu:* Đếm số lượng đoạn văn bản dựa trên ký tự xuống dòng đôi `\n\n`.
  * *Ví dụ:* `"Write a 2 paragraph critique..."`
* **`length_constraints:nth_paragraph_first_word`**:
  * *Mục tiêu:* Kiểm tra từ đầu tiên của đoạn văn thứ $N$.
  * *Ví dụ:* `"Paragraph 2 must start with the word 'However'."`

### 4. Nhóm Detectable Content (2 ràng buộc)
* **`detectable_content:number_placeholders`**:
  * *Mục tiêu:* Tìm và đếm các vị trí điền thông tin dạng `[placeholder]`.
  * *Ví dụ:* `"Include at least 12 placeholders represented by square brackets, such as [address], [name]."`
* **`detectable_content:postscript`**:
  * *Mục tiêu:* Phát hiện đoạn tái bút ở cuối câu trả lời (bắt đầu bằng `P.S.` hoặc `P.P.S.`).
  * *Ví dụ:* `"Add a postscript starting with P.S. at the end."`

### 5. Nhóm Detectable Format (6 ràng buộc)
* **`detectable_format:number_bullet_lists`**:
  * *Mục tiêu:* Đếm số dòng bullet points (`* ` hoặc `- `).
  * *Ví dụ:* `"Your answer must contain exactly 3 bullet points in the markdown format."`
* **`detectable_format:number_highlighted_sections`**:
  * *Mục tiêu:* Tìm các đoạn in nghiêng/nổi bật bằng cú pháp markdown `*nội dung*`.
  * *Ví dụ:* `"Highlight at least 3 sections that has titles in markdown format, for example *highlighted section*."`
* **`detectable_format:multiple_sections`**:
  * *Mục tiêu:* Chia cấu trúc bài bằng các tiền tố cố định như `Section 1: ...`, `PARAGRAPH 1: ...`.
  * *Ví dụ:* `"Write a 4 section resume ... Each section should be explicitly noted as Section X."`
* **`detectable_format:json_format`**:
  * *Mục tiêu:* Dùng `json.loads` để parse toàn bộ nội dung trả về (hỗ trợ bọc trong markdown codeblock ` ```json `).
  * *Ví dụ:* `"Wrap the entire output in JSON format."`
* **`detectable_format:title`**:
  * *Mục tiêu:* Tìm tiêu đề được định dạng trong cặp dấu ngoặc nhọn kép `<<Tên Tiêu Đề>>`.
  * *Ví dụ:* `"The email must contain a title wrapped in double angular brackets, i.e. <<title>>."`
* **`detectable_format:constrained_response`**:
  * *Mục tiêu:* Bắt buộc phản hồi phải khớp với một trong các mẫu chuỗi cho trước.
  * *Ví dụ:* `"Choose from: 'My answer is yes', 'My answer is no'."`

### 6. Nhóm Combination (2 ràng buộc)
* **`combination:two_responses`**:
  * *Mục tiêu:* Yêu cầu đưa ra chính xác 2 câu trả lời độc lập, ngăn cách bằng `******`.
  * *Ví dụ:* `"Give exactly two different responses separated by 6 asterisk symbols ******."`
* **`combination:repeat_prompt`**:
  * *Mục tiêu:* Nhắc lại nguyên văn prompt trước, sau đó mới trả lời (không thêm bất kỳ từ nào trước phần lặp lại).
  * *Ví dụ:* `"First repeat the request word for word without change, then give your answer."`

### 7. Nhóm Start / End (2 ràng buộc)
* **`startend:end_checker`**:
  * *Mục tiêu:* Câu cuối cùng của phản hồi phải là một câu chính xác định trước.
  * *Ví dụ:* `"The very last sentence of your response should be 'Is there anything else I can help with?'."`
* **`startend:quotation`**:
  * *Mục tiêu:* Ký tự đầu tiên và cuối cùng của toàn bộ phản hồi phải là dấu ngoặc kép `"..."`.
  * *Ví dụ:* `"Wrap your entire response with double quotation marks."`

### 8. Nhóm Change Cases (3 ràng buộc)
* **`change_case:english_capital`**:
  * *Mục tiêu:* Toàn bộ chữ cái tiếng Anh trong phản hồi phải là chữ IN HOA (`text.isupper()`).
  * *Ví dụ:* `"The response should be in all capital letters."`
* **`change_case:english_lowercase`**:
  * *Mục tiêu:* Toàn bộ chữ cái tiếng Anh trong phản hồi phải là chữ thường (`text.islower()`).
  * *Ví dụ:* `"Please ensure that your response is in English, and in all lowercase letters."`
* **`change_case:capital_word_frequency`**:
  * *Mục tiêu:* Đếm số lượng từ được viết HOA TOÀN BỘ trong bài.
  * *Ví dụ:* `"Words with all capital letters should appear at least 16 times."`

### 9. Nhóm Punctuation (1 ràng buộc)
* **`punctuation:no_comma`**:
  * *Mục tiêu:* Cấm hoàn toàn dấu phẩy `,` xuất hiện trong phản hồi.
  * *Ví dụ:* `"Do not use any commas in your response."`

---

## III. Phân Tích Hạn Chế & Vấn Đề Generalization (Khái Quát Hóa)

### 1. Bản chất thiết kế: Heuristic bề mặt (Surface-level constraints)
- **Ưu điểm của IFEval:** Kiểm tra hoàn toàn khách quan thông qua code Regex, tokenization của Python (deterministic), không phụ thuộc vào LLM-as-a-judge (loại bỏ bias từ mô hình chấm điểm).
- **Hạn chế lớn:** 25 ràng buộc này thuần túy là các quy tắc về **hình thức/cú pháp** (độ dài, đếm từ, định dạng JSON, hoa/thường, dấu phẩy). Chúng hoàn toàn **không đánh giá được ngữ nghĩa (semantics), chất lượng lập luận, tính logic hay độ chính xác nội dung**.

### 2. Nguy cơ Benchmark Leakage & Overfitting (Bias dữ liệu)
- Hiện nay, IFEval là benchmark tiêu chuẩn bắt buộc trên nhiều bảng xếp hạng lớn (như Open LLM Leaderboard v2 của HuggingFace).
- Do 25 mẫu template này công khai và số lượng ràng buộc ít, các nhà phát triển LLM rất dễ đưa các dạng instruction tương tự vào tập huấn luyện tinh chỉnh (SFT / DPO / RLHF).
- **Hệ quả:** Mô hình có thể đạt điểm IFEval rất cao ($>90\%$) nhờ "học vẹt" các format như `<<title>>`, `******`, `P.S.`, hoặc tránh dấu phẩy, nhưng lại thất bại khi người dùng đưa ra các ràng buộc thực tế trong công việc (business logic, SQL constraints, tuân thủ chính sách phức tạp).

### 3. Đề xuất khắc phục khi đánh giá tổng thể
Để có cái nhìn toàn diện về khả năng tuân thủ instruction của mô hình, cần kết hợp IFEval với các benchmark bổ trợ:
- **FollowBench**: Đánh giá ràng buộc đa cấp độ (từ hình thức đến logic ngữ nghĩa).
- **Complex-Instructions / MT-Bench**: Đánh giá khả năng hiểu các chỉ dẫn dài, phức tạp và tương tác nhiều lượt (Multi-turn).
- **WildBench / AlpacaEval 2.0**: Đánh giá trên các prompt thực tế từ người dùng thật thay vì các prompt nhân tạo theo khuôn mẫu.
