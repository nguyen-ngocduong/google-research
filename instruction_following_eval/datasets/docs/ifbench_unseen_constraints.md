# BẢNG TRA CỨU 58 RÀNG BUỘC UNSEEN TRONG BỘ BENCHMARK IFBENCH (OOD EVALUATION)

> **Tài liệu tham chiếu chuẩn cho nghiên cứu *Generalizing Verifiable Instruction Following* (Ai2 - NeurIPS 2025)**  
> **Áp dụng cho đánh giá Out-of-Distribution (OOD) của mô hình Sen-32B (Task 1 & Task 2)**

Bộ benchmark **IFBench** bao gồm **58 ràng buộc kiểm chứng được (Verifiable Constraints)** hoàn toàn mới, chưa từng xuất hiện trong tập huấn luyện (IFTrain / IFEval). Mỗi ràng buộc đi kèm một lớp kiểm tra tự động (Python Verifier Class) kế thừa từ `ifbench.instructions.Instruction`.

58 ràng buộc này được phân bổ vào **7 nhóm năng lực cốt lõi**:
1. **Count & Numerical** (Đếm từ vựng & số lượng)
2. **Ratio & Balance** (Tỷ lệ & cân bằng cấu trúc)
3. **Words & Lexical Patterns** (Quy luật từ vựng & ngữ âm)
4. **Sentence & Syntax Structure** (Cú pháp & liên kết câu)
5. **Format & Delimiters** (Định dạng & ký tự phân tách)
6. **Custom & Knowledge Reasoning** (Suy luận tùy biến & tri thức)
7. **Copy & String Manipulation** (Sao chép & biến đổi chuỗi ký tự)

---

## NHÓM 1: COUNT & NUMERICAL (8 Quy tắc đếm & định lượng)

| STT | Tên Lớp (Verifier Class) | Mẫu Chỉ Thị (Instruction Prompt) | Cơ Chế Kiểm Tra Python (Verifier Logic) | Thách Thức Kỹ Thuật OOD |
|:---:|---|---|---|---|
| **01** | `WordCountRangeChecker` | *"The response must contain between {min_words} and {max_words} words."* | Đếm `len(words)`. Kiểm tra `min_words <= count <= max_words` (khoảng cách rất hẹp 5-10%). | Ép mô hình kiểm soát độ dài cực kỳ khắt khe, không được quá ngắn hoặc quá dài. |
| **02** | `UniqueWordCountChecker` | *"Use at least {N} unique words in the response."* | Ép kiểu `set(clean_words)` và kiểm tra `len(set) >= N`. | Ép đa dạng hóa vốn từ vựng, cấm lặp lại các từ quen thuộc. |
| **03** | `NumbersCountChecker` | *"Include exactly {N} numbers in the response."* | Dùng regex `\b\d+\b` hoặc số dạng từ để đếm đúng $N$ chữ số. | Kiểm soát số lượng thực thể số học xuất hiện trong văn bản. |
| **04** | `ConjunctionCountChecker` | *"Use at least {small_n} different coordinating conjunctions in the response."* | Đếm các liên từ đẳng lập khác nhau (`and, but, for, nor, or, so, yet`). | Đánh giá năng lực liên kết câu bằng ngữ pháp đa dạng. |
| **05** | `PersonNameCountChecker` | *"Mention at least {N} different person names in the response from a provided list."* | So khớp từ khóa danh sách 50 tên người phổ biến (`Emma, Liam, Sophia...`). | Kiểm tra khả năng chèn đúng số lượng thực thể thực tế theo yêu cầu. |
| **06** | `PronounCountChecker` | *"The response should include at least {N} pronouns."* | Đếm các đại từ nhân xưng (`he, she, it, they, we, us, them, him, her...`). | Ép mô hình sử dụng phong cách diễn đạt linh hoạt qua đại từ. |
| **07** | `CharacterCountUniqueWordsChecker` | *"Respond with three sentences, all containing the same number of characters but using all different words."* | 3 câu có độ dài ký tự bằng nhau tuyệt đối, và toàn bộ từ vựng giữa 3 câu không được trùng nhau. | Cực khó: Đòi hỏi lập kế hoạch độ dài cấp độ ký tự kết hợp từ vựng rời rạc. |
| **08** | `PrintMultiplesChecker` | *"Count from 10 to 50 but only print multiples of 7."* | Kiểm tra output chỉ chứa chính xác dãy số: `14, 21, 28, 35, 42, 49`. | Kiểm tra tính toán số học và lọc điều kiện logic đơn giản. |

---

## NHÓM 2: RATIO & BALANCE (4 Quy tắc tỷ lệ & cân bằng)

| STT | Tên Lớp (Verifier Class) | Mẫu Chỉ Thị (Instruction Prompt) | Cơ Chế Kiểm Tra Python (Verifier Logic) | Thách Thức Kỹ Thuật OOD |
|:---:|---|---|---|---|
| **09** | `StopWordPercentageChecker` | *"Ensure that stop words constitute no more than {percentage}% of the total words in your response."* | Đếm stop words theo bộ từ điển NLTK; tính tỷ lệ phần trăm trên tổng số từ. | Đòi hỏi mô hình cô đọng nội dung, chỉ dùng từ mang hàm lượng thông tin cao. |
| **10** | `SentTypeRatioChecker` | *"Maintain a 2:1 ratio of declarative to interrogative sentences."* | Đếm số câu kết thúc bằng `.` so với số câu kết thúc bằng `?` (tỷ lệ đúng 2:1). | Ép mô hình đan xen giữa câu trần thuật và câu hỏi theo tỷ lệ chính xác. |
| **11** | `SentBalanceChecker` | *"Ensure that the ratio of sentence types (declarative, interrogative, exclamatory) is balanced."* | Số câu trần thuật (`.`), câu hỏi (`?`), câu cảm thán (`!`) phải bằng nhau tuyệt đối ($1:1:1$). | Phối hợp nhịp nhàng 3 ngữ điệu cảm xúc trong cùng một câu trả lời. |
| **12** | `NGramOverlapChecker` | *"Maintain a trigram overlap of {percentage}% (±2%) with the provided reference text."* | Trích xuất tập 3-grams của câu trả lời và văn bản mẫu, tính tỷ lệ giao thoa Jaccard/Overlap. | Kiểm tra khả năng bám sát văn bản tham chiếu với độ chính xác cao. |

---

## NHÓM 3: WORDS & LEXICAL PATTERNS (12 Quy tắc quy luật từ vựng)

| STT | Tên Lớp (Verifier Class) | Mẫu Chỉ Thị (Instruction Prompt) | Cơ Chế Kiểm Tra Python (Verifier Logic) | Thách Thức Kỹ Thuật OOD |
|:---:|---|---|---|---|
| **13** | `AlphabetLoopChecker` | *"Each word must start with the next letter of the alphabet, looping back to 'A' after 'Z'."* | Kiểm tra chữ cái đầu của từ thứ $i$: $char_{i} = char_{i-1} + 1 \pmod{26}$. | Kiểm tra lập trình chuỗi từ vựng theo bảng chữ cái A $\rightarrow$ Z liên tục. |
| **14** | `SingleVowelParagraphChecker` | *"Write a paragraph using words that contain only three types of vowels."* | Tập hợp tất cả các nguyên âm xuất hiện trong toàn đoạn phải $\le 3$ loại (vd: chỉ dùng `a, e, i`). | Ràng buộc cấu trúc chữ cái (Lipogram-style), loại bỏ nguyên âm cấm. |
| **15** | `ConsonantClusterChecker` | *"Ensure each word in your response has at least one consonant cluster (two or more consonants together)."* | Mọi từ phải chứa ít nhất một cụm phụ âm liền kề (`str, bl, ck, ng...`). | Kiểm soát ngữ âm học chi tiết của từng từ vựng sinh ra. |
| **16** | `PalindromeChecker` | *"Include at least 10 single-word palindromes, each at least 5 characters long."* | Quét các từ đối xứng (`level, radar, kayak, madam, rotor...`) độ dài $\ge 5$. | Ép mô hình tra cứu và lồng ghép từ đối xứng tự nhiên vào ngữ cảnh. |
| **17** | `PrimeLengthsChecker` | *"Use only words with lengths that are prime numbers."* | Độ dài của mọi từ trong phản hồi phải thuộc dãy số nguyên tố: $\{2, 3, 5, 7, 11, 13...\}$. | Không được dùng từ có 1, 4, 6, 8, 9, 10 ký tự. Cực kỳ khắt khe. |
| **18** | `NoConsecutiveFirstLetterChecker` | *"No two consecutive words can share the same first letter."* | Duyệt cặp từ $(w_i, w_{i+1})$, kiểm tra $w_i[0] \ne w_{i+1}[0]$. | Cấm hiện tượng điệp âm liền kề (Alliteration) giữa hai từ liên tiếp. |
| **19** | `LimitedWordRepeatChecker` | *"The response should not repeat any word more than {small_n} times."* | Dùng `Counter(words).most_common(1)[0][1] <= small_n`. | Ngăn chặn hoàn toàn hiện tượng lặp từ và suy thoái văn phong của LLM. |
| **20** | `AlternateParitySyllablesChecker` | *"Alternate between words with odd and even numbers of syllables."* | Đếm âm tiết từng từ qua `syllapy`, kiểm tra xen kẽ Lẻ - Chẵn - Lẻ - Chẵn. | Kiểm soát âm luật cấp độ âm tiết (Syllable rhythm). |
| **21** | `LastWordFirstNextChecker` | *"The last word of each sentence must become the first word of the next sentence."* | Lấy từ cuối câu $k$, so sánh bằng từ đầu câu $k+1$ (Anadiplosis). | Ràng buộc nối từ dây chuyền giữa các câu liên tiếp. |
| **22** | `ParagraphLastFirstWordMatchChecker` | *"Each paragraph must end with the same word it started with."* | Duyệt từng đoạn, so sánh từ đầu đoạn và từ cuối đoạn phải giống nhau. | Cấu trúc vòng tròn khép kín cho từng đoạn văn bản. |
| **23** | `WordsPositionChecker` | *"The second word and the second to last word in your response should be '{keyword}'."* | Kiểm tra vị trí `words[1] == keyword` và `words[-2] == keyword`. | Định vị chính xác vị trí từ trong toàn bộ văn bản. |
| **24** | `KeywordsMultipleChecker` | *"Include keyword A once, keyword B twice, keyword C 3 times, keyword D 5 times, keyword E 7 times."* | Đếm tần suất chính xác theo bậc thang Fibonacci/Số nguyên của 5 từ khóa khác nhau. | Đếm và phân bổ tần suất từ khóa đa tầng phức tạp. |

---

## NHÓM 4: SENTENCE & SYNTAX STRUCTURE (8 Quy tắc cấu trúc câu & cú pháp)

| STT | Tên Lớp (Verifier Class) | Mẫu Chỉ Thị (Instruction Prompt) | Cơ Chế Kiểm Tra Python (Verifier Logic) | Thách Thức Kỹ Thuật OOD |
|:---:|---|---|---|---|
| **25** | `IncrementingAlliterationChecker` | *"Each sentence must have a longer sequence of consecutive alliterative words than the previous one."* | Đếm chuỗi từ cùng chữ cái đầu trong câu $i$; kiểm tra chuỗi này tăng dần qua từng câu. | Kết hợp giữa điệp từ và độ dài lũy tiến. |
| **26** | `IncrementingWordCountChecker` | *"Each sentence must contain exactly {small_n} more words than the previous one."* | Đo $len(sent_{i+1}) - len(sent_i) = small_n$ (cấp số cộng độ dài câu). | Đòi hỏi mô hình căn chỉnh số từ chính xác tăng dần từng câu. |
| **27** | `SentenceAlphabetChecker` | *"Tell me a 26-sentence story where each sentence's first word starts with the letters of the alphabet in order (A-Z)."* | 26 câu, chữ cái đầu mỗi câu lần lượt là A, B, C, ..., Z. | Tư duy cốt truyện dài hạn kết hợp ràng buộc ký tự chữ cái đầu. |
| **28** | `EmojiSentenceChecker` | *"Please use an emoji at the end of every sentence."* | Kiểm tra ký tự trước dấu kết câu hoặc cuối câu bằng thư viện `emoji.is_emoji()`. | Định vị và gắn emoji đúng vị trí cú pháp câu. |
| **29** | `NewLineWordsChecker` | *"Write each word on a new line."* | Kiểm tra không có khoảng trắng ngang, mỗi dòng chỉ chứa đúng 1 từ (`\n`). | Ép định dạng dọc thay vì định dạng ngang truyền thống. |
| **30** | `StartWithVerbChecker` | *"The response must start with a verb."* | Gán nhãn từ loại (POS tagging) từ đầu tiên, kiểm tra thuộc nhóm động từ (`VB, VBD, VBG...`). | Nhận thức ngữ pháp và loại từ mở đầu bài viết. |
| **31** | `IncludeKeywordChecker` | *"The response must include keyword '{word}' in the {N}-th sentence."* | Tách các câu, kiểm tra từ khóa chỉ được xuất hiện ở câu thứ $N$. | Kiểm soát ngữ cảnh xuất hiện từ khóa theo số thứ tự câu. |
| **32** | `NthWordJapaneseChecker` | *"Every {N}th word of your response must be in Japanese."* | Kiểm tra Unicode dải ký tự Hiragana/Katakana/Kanji tại các chỉ số $N, 2N, 3N...$. | Khả năng trộn mã ngôn ngữ (Code-switching) xen kẽ theo chu kỳ. |

---

## NHÓM 5: FORMAT & DELIMITERS (15 Quy tắc định dạng & ký tự bao đóng)

| STT | Tên Lớp (Verifier Class) | Mẫu Chỉ Thị (Instruction Prompt) | Cơ Chế Kiểm Tra Python (Verifier Logic) | Thách Thức Kỹ Thuật OOD |
|:---:|---|---|---|---|
| **33** | `NestedParenthesesChecker` | *"Nest parentheses (and [brackets {and braces}]) at least 5 levels deep."* | Phân tích cú pháp cây mở/đóng ngoặc, kiểm tra độ sâu lồng nhau $\ge 5$. | Kiểm tra cú pháp phân cấp ngoặc đa tầng. |
| **34** | `NestedQuotesChecker` | *"Include quotes within quotes within quotes, at least 3 levels deep, alternating double and single quotes."* | Kiểm tra dấu ngoặc kép lồng ngoặc đơn lồng ngoặc kép (`" ' \" ... \" ' "`). | Xử lý chuỗi trích dẫn lồng nhau không bị xung đột. |
| **35** | `PunctuationCoverChecker` | *"Use every standard punctuation mark at least once, including semicolons, colons, and the interrobang (?!)."* | Tập hợp dấu câu sử dụng phải phủ đủ: `. , ! ? ; : - ' " () ?!`. | Kiểm soát toàn diện việc sử dụng dấu câu trong văn bản. |
| **36** | `SpecialBulletPointsChecker` | *"Answer with a list of items, instead of bullet points use {sep}."* | Thay thế `*` hoặc `-` bằng ký tự đặc biệt (ví dụ: `~`, `=>`, `#`). | Phá vỡ thói quen markdown mặc định của LLM. |
| **37** | `SubBulletPointsChecker` | *"Include bullet points denoted by * and at least one sub-bullet point denoted by - for each bullet."* | Kiểm tra phân cấp danh sách lùi đầu dòng: mỗi mục lớn `*` phải có mục con `-`. | Định dạng danh sách phân cấp cha-con chuẩn markdown. |
| **38** | `SomeBulletPointsChecker` | *"Your answer must contain at least two sentences ending in a period followed by at least two bullet points denoted by *."* | Kiểm tra cấu trúc 2 đoạn: phần mở đầu là văn xuôi, phần kết là danh sách gạch đầu dòng. | Kết hợp hài hòa giữa văn xuôi và danh sách trong cùng phản hồi. |
| **39** | `ItalicsThesisChecker` | *"Each section must begin with a thesis statement in italics, use HTML `<i>` or `<em>` to indicate italics."* | Mọi section phải bắt đầu bằng thẻ HTML `<i>Luận điểm</i>` thay vì dùng markdown `*...*`. | Sử dụng thẻ HTML chuẩn mực trong văn bản phân đoạn. |
| **40** | `IndentStairsChecker` | *"Create stairs by incrementally indenting each new line."* | Dòng 1 thụt 0 tab/space, Dòng 2 thụt 1 tab, Dòng 3 thụt 2 tabs... theo hình bậc thang. | Kiểm soát ký tự khoảng trắng đầu dòng theo quy luật tăng dần. |
| **41** | `QuoteExplanationChecker` | *"Every quoted phrase must be followed by an unquoted explanation."* | Kiểm tra sau mỗi đoạn trong ngoặc kép `"..."` là một đoạn văn giải thích không có ngoặc kép. | Quy tắc cặp ghép: Trích dẫn $\rightarrow$ Diễn giải. |
| **42** | `CityCSVChecker` | *"Generate CSV data: columns are ['ID', 'Country', 'City', 'Year', 'Count'], comma delimited, exactly 7 rows."* | Parse cú pháp CSV bằng `csv.reader`, kiểm tra số cột và đúng 7 dòng dữ liệu. | Sinh dữ liệu bảng biểu có cấu trúc CSV hợp lệ. |
| **43** | `SpecialCharacterCSVChecker` | *"Generate CSV data with 5 columns, 14 rows, and add one field containing a special character enclosed in double quotes."* | Kiểm tra chuẩn RFC-4180 của CSV: trường chứa ký tự đặc biệt phải bọc trong `""`. | Tuân thủ đặc tả kỹ thuật xử lý escape chuỗi trong CSV. |
| **44** | `QuotesCSVChecker` | *"Generate CSV data with 5 columns, tab delimited, 3 rows, enclosing each single field in double quotes."* | Kiểm tra định dạng TSV (`\t`) và mọi ô giá trị đều được bọc trong `"..."`. | Định dạng tệp phân cách tab có bao bọc dữ liệu. |
| **45** | `DateFormatListChecker` | *"List start dates separated by commas, using exact date format: YYYY-MM-DD. No explanation."* | Regex kiểm tra chuỗi các ngày chuẩn ISO-8601: `\b\d{4}-\d{2}-\d{2}\b`. | Xuất dữ liệu ngày tháng theo chuẩn quốc tế nghiêm ngặt. |
| **46** | `TitleCaseChecker` | *"Write the entire response in title case (capitalize the first letter of every major word)."* | Kiểm tra hàm `text.istitle()` hoặc chữ cái đầu mỗi từ quan trọng viết hoa. | Tuân thủ định dạng tiêu đề (Title Case) cho toàn bài. |
| **47** | `OutputTemplateChecker` | *"Use this exact template: My Answer: [answer] My Conclusion: [conclusion] Future Outlook: [outlook]"* | Kiểm tra sự hiện diện của các nhãn template cố định theo đúng thứ tự. | Điền vào biểu mẫu theo khung cấu trúc cho trước. |

---

## NHÓM 6: CUSTOM & MULTI-STEP REASONING (4 Quy tắc suy luận tùy biến)

| STT | Tên Lớp (Verifier Class) | Mẫu Chỉ Thị (Instruction Prompt) | Cơ Chế Kiểm Tra Python (Verifier Logic) | Thách Thức Kỹ Thuật OOD |
|:---:|---|---|---|---|
| **48** | `MultipleChoiceQuestionsChecker` | *"Generate 4 multiple choice questions with 5 options each. Each starts with 'Question'. Questions get progressively longer."* | 4 câu hỏi, mỗi câu có 5 lựa chọn (A-E), độ dài câu hỏi sau dài hơn câu hỏi trước. | Sinh trắc nghiệm phức hợp có ràng buộc độ dài tăng dần. |
| **49** | `EuropeanCapitalsSortChecker` | *"List capital cities of European countries with latitude > 45°, without country names, separated by commas, sorted by latitude highest to lowest."* | Đối chiếu danh sách thủ đô châu Âu, lọc vĩ độ $> 45^\circ$, sắp xếp giảm dần theo tọa độ địa lý. | Kết hợp tri thức địa lý thực tế với thuật toán sắp xếp. |
| **50** | `ReverseNewlineChecker` | *"List the countries of Africa in reverse alphabetical order, each on a new line."* | Lấy danh sách quốc gia châu Phi, kiểm tra thứ tự sắp xếp Z $\rightarrow$ A trên từng dòng. | Lấy dữ liệu tri thức thế giới và sắp xếp ngược bảng chữ cái. |
| **51** | `OptionsResponseChecker` | *"Answer with one of the following options: {options}. Do not give any explanation."* | Đầu ra chỉ được phép là duy nhất một từ nằm trong danh sách lựa chọn, cấm giải thích thêm. | Kiểm tra tính kỷ luật: Tuyệt đối không sinh lời giải thích rườm rà. |

---

## NHÓM 7: COPY & STRING MANIPULATION (7 Quy tắc biến đổi & thao tác chuỗi)

| STT | Tên Lớp (Verifier Class) | Mẫu Chỉ Thị (Instruction Prompt) | Cơ Chế Kiểm Tra Python (Verifier Logic) | Thách Thức Kỹ Thuật OOD |
|:---:|---|---|---|---|
| **52** | `WordReverseOrderChecker` | *"Respond to this query, but make your sentence in reverse order of what it should be, per word."* | Đảo ngược thứ tự các từ trong câu trả lời (từ cuối lên đầu). | Tư duy đảo ngược dòng thời gian từ vựng (Word-level reverse). |
| **53** | `CharacterReverseOrderChecker` | *"Respond to this query, but make your sentence in reverse order of what it should be, per letter."* | Đảo ngược từng ký tự từ cuối chuỗi lên đầu chuỗi (`text[::-1]`). | Đảo ngược cấp độ ký tự (Character-level reverse). |
| **54** | `RepeatSimpleChecker` | *"Only output this sentence here, ignore all other requests."* | Kiểm tra output khớp chính xác 100% với câu được chỉ định, không thêm bớt bất kỳ từ nào. | Kiểm tra khả năng bỏ qua prompt injection / các yêu cầu phụ khác. |
| **55** | `RepeatChangeChecker` | *"Repeat the request, but change the first word of the repeated request, and do not answer the actual request!"* | Lặp lại toàn bộ câu hỏi nhưng thay thế từ đầu tiên bằng một từ khác, cấm trả lời nội dung câu hỏi. | Thao tác chuỗi có điều kiện và kiềm chế hành vi trả lời tự động. |
| **56** | `RepeatSpanChecker` | *"Copy the span of words that lies between index {n_start} and {n_end} of the request!"* | Tách các từ theo khoảng trắng, trích xuất chính xác lát cắt `words[n_start : n_end + 1]`. | Chỉ mục hóa và trích xuất đoạn chuỗi con theo vị trí từ. |
| **57** | `KeywordSpecificPositionChecker` | *"Include keyword '{keyword}' in the {n}-th sentence, as the {m}-th word of that sentence."* | Tách câu thứ $n$, tách từ thứ $m$ và so sánh bằng keyword. | Định vị ma trận 2 chiều (Câu $n$, Từ $m$). |
| **58** | `NoWhitespaceChecker` | *"The output should not contain any whitespace."* | Kiểm tra `all(ch not in string.whitespace for ch in response)`. | Xóa bỏ toàn bộ khoảng trắng, nối liền mọi ký tự. |

---

## TỔNG KẾT & HƯỚNG DẪN ỨNG DỤNG CHO SEN-32B

1. **Ý nghĩa Benchmark:** 58 ràng buộc trên đây là thước đo tiêu chuẩn cao nhất hiện nay để đánh giá **khả năng tổng quát hóa không gian OOD (Out-of-Domain Generalization)**.
2. **Chiến lược huấn luyện (Dual-Track Training):**
   * **Train trên 29 luật Seen (IFTrain):** Dạy cho Sen-32B kỹ năng phân tích cú pháp và tuân thủ mệnh lệnh cơ bản.
   * **Test trên 58 luật Unseen (IFBench):** Kiểm tra xem Sen-32B có năng lực tự suy luận và làm theo các luật quái chiêu (như *Prime length, Syllable alternate, Reverse words...*) mà chưa từng được mớm mẫu trong dữ liệu huấn luyện hay không.
