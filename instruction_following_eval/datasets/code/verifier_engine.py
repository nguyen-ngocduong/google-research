#!/usr/bin/env python3
# coding=utf-8
"""
Verifier Engine for Instruction Following Evaluation (IFEval / IFBench / CRPL)
Version: v2 (Structured Items with Detailed Verification Notes)
"""
import re
import json
import string
from collections import Counter
from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0

def split_sentences(text):
    clean = text.strip()
    parts = re.split(r'(?<!\b\d)(?<=[.!?])\s+', clean)
    return [p.strip() for p in parts if p.strip()]

def verify_hard_constraint_detailed(inst_id, kwargs, response_text):
    """
    Xác thực ràng buộc cứng và trả về tuple: (passed: bool, note: str)
    """
    text = response_text.strip()
    
    # 1. Quotation
    if inst_id == "startend:quotation":
        passed = (text.startswith('"') and text.endswith('"')) or (text.startswith('“') and text.endswith('”'))
        return passed, "Được bao bọc trong ngoặc kép" if passed else "Không có ngoặc kép ở đầu và cuối"
    
    # 2. Number of sentences
    if inst_id == "length_constraints:number_sentences":
        num = kwargs.get("num_sentences", 0)
        rel = kwargs.get("relation", "exactly")
        sentences = split_sentences(text)
        count = len(sentences)
        if rel == "less than": passed = count < num
        elif rel == "at least": passed = count >= num
        else: passed = (count == num)
        return passed, f"Số câu: {count} (yêu cầu {rel} {num})"
        
    # 3. Postscript
    if inst_id == "detectable_content:postscript":
        marker = kwargs.get("postscript_marker", "P.S.")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines: return False, "Phản hồi rỗng"
        passed = lines[-1].startswith(marker) or lines[-1].startswith("Tái bút:")
        return passed, "Có tái bút ở dòng cuối" if passed else "Dòng cuối không bắt đầu bằng P.S. hoặc Tái bút:"
        
    # 4. Words count
    if inst_id == "length_constraints:number_words":
        words = len(text.split())
        num = kwargs.get("num_words", 0)
        rel = kwargs.get("relation", "less than")
        if rel == "less than": passed = words < num
        elif rel == "at least": passed = words >= num
        elif rel == "range": passed = kwargs.get("min_words", 0) <= words <= kwargs.get("max_words", 9999)
        else: passed = (words == num)
        return passed, f"Số từ: {words} (yêu cầu {rel} {num})"
        
    # 5. Paragraphs count
    if inst_id == "length_constraints:number_paragraphs":
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        req = kwargs.get("num_paragraphs", 0)
        passed = len(paras) == req
        return passed, f"Số đoạn văn: {len(paras)} (yêu cầu {req})"
        
    # 6. Bullet lists
    if inst_id == "detectable_format:number_bullet_lists":
        bullets = re.findall(r'(?m)^\s*[*-]\s+', text)
        req = kwargs.get("num_bullets", 0)
        passed = len(bullets) == req
        return passed, f"Số gạch đầu dòng: {len(bullets)} (yêu cầu {req})"
        
    # 7. Numbered list
    if inst_id == "format:numbered_list":
        items = re.findall(r'(?m)^\s*\d+\.\s+', text)
        req = kwargs.get("num_items", 0)
        passed = len(items) == req
        return passed, f"Số mục đánh số: {len(items)} (yêu cầu {req})"
        
    # 8. JSON Format
    if inst_id == "detectable_format:json_format":
        try:
            clean_j = text.strip().strip('`')
            if clean_j.startswith('json'): clean_j = clean_j[4:].strip()
            json.loads(clean_j)
            return True, "JSON hợp lệ"
        except Exception as e:
            return False, f"JSON không hợp lệ: {str(e)[:50]}"
        
    # 9. Title (<<Title>> or # Title)
    if inst_id == "detectable_format:title":
        has_chevrons = bool(re.search(r'<<[^>]+>>', text))
        has_header = bool(re.search(r'(?m)^#\s+.+', text))
        passed = has_chevrons or has_header
        return passed, "Có tiêu đề định dạng hợp lệ" if passed else "Không tìm thấy tiêu đề dạng <<...>> hoặc # ..."
        
    # 10. Keyword existence
    if inst_id == "keywords:existence":
        kws = kwargs.get("keywords", [])
        lower = text.lower()
        missing = [w for w in kws if not re.search(rf'\b{re.escape(w.lower())}\b', lower)]
        passed = len(missing) == 0
        return passed, "Đủ từ khóa" if passed else f"Thiếu từ khóa: {missing}"
        
    # 11. Keyword frequency
    if inst_id == "keywords:frequency":
        word = kwargs.get("keyword", "").lower()
        freq = kwargs.get("frequency", 0)
        rel = kwargs.get("relation", "exactly")
        actual = len(re.findall(rf'\b{re.escape(word)}\b', text.lower()))
        if rel == "at least": passed = actual >= freq
        elif rel == "at most": passed = actual <= freq
        else: passed = (actual == freq)
        return passed, f"Từ '{word}' xuất hiện {actual} lần (yêu cầu {rel} {freq})"
        
    # 12. Letter frequency
    if inst_id == "keywords:letter_frequency":
        letter = kwargs.get("letter", "").lower()
        freq = kwargs.get("let_frequency", 0)
        rel = kwargs.get("let_relation", "at least")
        actual = text.lower().count(letter)
        if rel == "at least": passed = actual >= freq
        elif rel == "at most": passed = actual <= freq
        else: passed = (actual == freq)
        return passed, f"Chữ '{letter}' xuất hiện {actual} lần (yêu cầu {rel} {freq})"
        
    # 13. Forbidden words
    if inst_id == "keywords:forbidden_words":
        forb = kwargs.get("forbidden_words", [])
        lower = text.lower()
        found = [w for w in forb if re.search(rf'\b{re.escape(w.lower())}\b', lower)]
        passed = len(found) == 0
        return passed, "Không chứa từ cấm" if passed else f"Chứa từ cấm: {found}"
        
    # 14. End checker
    if inst_id == "startend:end_checker":
        phrase = kwargs.get("end_phrase", "").strip()
        passed = text.strip().endswith(phrase) or text.strip().rstrip(".!?").endswith(phrase)
        return passed, "Kết thúc đúng cụm từ" if passed else f"Không kết thúc bằng '{phrase}'"
        
    # 15. Multiple sections
    if inst_id == "detectable_format:multiple_sections":
        n = kwargs.get("num_sections", 0)
        spliter = kwargs.get("section_spliter", "Section")
        sections = re.findall(rf'(?i){re.escape(spliter)}\s*\d+', text)
        passed = len(sections) >= n
        return passed, f"Tìm thấy {len(sections)} phần (yêu cầu ít nhất {n})"
        
    # 16. Highlighted sections
    if inst_id == "detectable_format:number_highlighted_sections":
        n = kwargs.get("num_highlights", 0)
        highlights = re.findall(r'\*{1,2}[^*\n]+\*{1,2}', text)
        passed = len(highlights) >= n
        return passed, f"Tìm thấy {len(highlights)} đoạn highlight (yêu cầu {n})"
        
    # 17. No comma
    if inst_id == "punctuation:no_comma":
        comma_count = text.count(',') + text.count('，')
        passed = comma_count == 0
        return passed, "Không có dấu phẩy" if passed else f"Chứa {comma_count} dấu phẩy"
        
    # 18. XML wrapper
    if inst_id == "format:xml_wrapper":
        passed = text.startswith("<response>") and text.endswith("</response>")
        return passed, "Bao bọc thẻ <response> đúng" if passed else "Thiếu hoặc sai thẻ <response>"
        
    # 19. All caps / Lower
    if inst_id == "change_case:english_capital":
        passed = text.isupper()
        return passed, "Toàn bộ là chữ hoa" if passed else "Có chứa chữ thường"
    if inst_id == "change_case:english_lowercase":
        passed = text.islower()
        return passed, "Toàn bộ là chữ thường" if passed else "Có chứa chữ hoa"

    # 20. Capital words frequency
    if inst_id == "change_case:capital_word_frequency":
        caps = re.findall(r'\b[A-Z]{2,}\b', text)
        rel = kwargs.get("relation", "at least")
        tgt = kwargs.get("capital_words", 2)
        if rel == "at least": passed = len(caps) >= tgt
        elif rel == "at most": passed = len(caps) <= tgt
        elif rel == "less than": passed = len(caps) < tgt
        else: passed = (len(caps) == tgt)
        return passed, f"Có {len(caps)} từ ALL CAPS (yêu cầu {rel} {tgt})"

    # 21. Response language (Chuẩn hóa langdetect, hỗ trợ đa chế độ CRPL)
    if inst_id == "language:response_language":
        LANG_MAP = {"Vietnamese": "vi", "French": "fr", "English": "en", "Spanish": "es", "German": "de"}
        def _lang_ratio(t, code):
            try:
                return next((l.prob for l in detect_langs(t) if l.lang == code), 0.0)
            except Exception:
                return 0.0

        mode = kwargs.get("mode", "all")
        lang_name = kwargs.get("language", "Vietnamese")
        code = LANG_MAP.get(lang_name, "vi")

        if mode == "all":
            prob = _lang_ratio(text, code)
            passed = prob >= 0.80
            return passed, f"Độ khớp ngôn ngữ {lang_name} ({code}): {prob:.1%}"

        if mode == "any_of":
            vi = _lang_ratio(text, "vi")
            en = _lang_ratio(text, "en")
            passed = vi >= 0.80 or (vi >= 0.20 and en >= 0.20)
            return passed, f"Tỷ lệ ngôn ngữ: VN={vi:.1%}, EN={en:.1%}"

        if mode == "first_half_second_half":
            paras = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
            if len(paras) < 2:
                paras = split_sentences(text)
                if len(paras) < 2: return False, "Văn bản quá ngắn để chia nửa ngôn ngữ"
            mid = len(paras) // 2
            first, second = "\n".join(paras[:mid]), "\n".join(paras[mid:])
            r1 = _lang_ratio(first, "vi")
            r2 = _lang_ratio(second, "en")
            passed = r1 >= 0.65 and r2 >= 0.65
            return passed, f"Nửa đầu (VN): {r1:.1%}, Nửa sau (EN): {r2:.1%}"
        return False, f"Chế độ mode '{mode}' không hợp lệ"

    # 22. Paragraph nth first word
    if inst_id == "length_constraints:nth_paragraph_first_word":
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        nth = kwargs.get("nth_paragraph", 1) - 1
        expected = kwargs.get("first_word", "").lower()
        if nth < len(paras):
            w = re.findall(r'\b\w+\b', paras[nth].lower())
            actual = w[0] if w else ""
            passed = (actual == expected)
            return passed, f"Đoạn {nth+1} bắt đầu bằng '{actual}' (yêu cầu '{expected}')"
        return False, f"Không đủ {nth+1} đoạn văn"

    # 23. Number placeholders (Khắc phục n == 0 và hỗ trợ Unicode tiếng Việt)
    if inst_id == "detectable_content:number_placeholders":
        PLACEHOLDER_RE = re.compile(r'\[[^\[\]\n]+\]')
        n = kwargs.get("num_placeholders", 1)
        found = PLACEHOLDER_RE.findall(text)
        if n == 0:
            passed = ('[' not in text and ']' not in text)
            return passed, "Không có ngoặc vuông" if passed else f"Tìm thấy ngoặc vuông: {found}"
        passed = len(found) >= n
        return passed, f"Tìm thấy {len(found)} placeholders (yêu cầu ít nhất {n})"

    # 24. Constrained response
    if inst_id == "detectable_format:constrained_response":
        opts = kwargs.get("constrained_responses", [])
        lower = text.lower()
        matched = [o for o in opts if re.search(rf'\b{re.escape(o.lower())}\b', lower)]
        passed = len(matched) > 0
        return passed, f"Khớp lựa chọn: {matched}" if passed else f"Không chứa lựa chọn nào trong: {opts}"

    # 25. Two responses
    if inst_id == "combination:two_responses":
        passed = "******" in text
        return passed, "Có dấu phân cách ******" if passed else "Thiếu dấu phân cách ******"

    # 26. Repeat prompt
    if inst_id == "combination:repeat_prompt":
        prompt_to_repeat = kwargs.get("prompt_to_repeat", "").strip()
        if prompt_to_repeat:
            first_para = text.split('\n')[0].strip()
            passed = prompt_to_repeat.lower() in first_para.lower()
            return passed, "Đã lặp lại prompt ở dòng đầu" if passed else "Không tìm thấy prompt ở đầu bài"
        passed = len(text.split('\n')) >= 2
        return passed, "Đã lặp lại câu hỏi" if passed else "Không đủ số dòng lặp"

    # 27. Table format
    if inst_id == "detectable_format:table_format":
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        table_lines = [l for l in lines if l.startswith('|') and l.endswith('|')]
        passed = len(table_lines) >= 2
        return passed, f"Tìm thấy {len(table_lines)} dòng bảng Markdown (yêu cầu >= 2)"

    # 28. No period
    if inst_id == "punctuation:no_period":
        period_count = text.count('.') + text.count('。')
        passed = period_count == 0
        return passed, "Không có dấu chấm" if passed else f"Chứa {period_count} dấu chấm"

    # ================== UNSEEN POOL (24 Curated IFBench OOD) ==================
    if inst_id == "count:unique_word_count":
        words = re.findall(r'\b\w+\b', text.lower())
        u_count = len(set(words))
        req = kwargs.get("num_unique_words", 0)
        return u_count >= req, f"Số từ độc nhất: {u_count} (yêu cầu {req})"

    if inst_id == "format:parentheses":
        nested = re.findall(r'\(\([^\(\)]+\)\)', text)
        req = kwargs.get("num_nested", 1)
        return len(nested) >= req, f"Số ngoặc đôi (()): {len(nested)} (yêu cầu {req})"

    if inst_id == "words:palindrome":
        palindromes = kwargs.get("palindromes", [])
        lower = text.lower()
        found = [p for p in palindromes if re.search(rf'\b{re.escape(p.lower())}\b', lower)]
        return len(found) == len(palindromes), f"Từ đối xứng tìm thấy: {found}/{palindromes}"

    if inst_id == "sentence:increment":
        sentences = split_sentences(text)
        if len(sentences) < 2: return False, "Cần ít nhất 2 câu"
        lens = [len(s.split()) for s in sentences]
        passed = all(lens[i] < lens[i+1] for i in range(len(lens)-1))
        return passed, f"Độ dài từng câu: {lens}"

    if inst_id == "words:start_verb":
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not paras: return False, "Phản hồi rỗng"
        COMMON_VERBS = {'eat', 'exercise', 'get', 'drink', 'sleep', 'maintain', 'prioritize', 'practice', 'focus', 'ensure', 'avoid', 'include', 'engage', 'build', 'create', 'develop', 'run', 'walk', 'read', 'write', 'start', 'begin', 'take', 'make', 'do', 'use', 'stay', 'keep', 'choose', 'aim', 'limit', 'reduce', 'boost', 'protect', 'adopt', 'schedule', 'cultivate', 'foster', 'incorporate', 'strive'}
        for para in paras:
            first_word = re.findall(r'\b\w+\b', para.lower())
            if not first_word or (first_word[0] not in COMMON_VERBS and not (first_word[0].endswith('ing') or first_word[0].endswith('ed'))):
                return False, f"Đoạn văn không bắt đầu bằng động từ hợp lệ: '{first_word[0] if first_word else 'Rỗng'}'"
        return True, "Mọi đoạn đều bắt đầu bằng động từ"

    if inst_id == "count:punctuation":
        found = set(c for c in text if c in string.punctuation)
        req = kwargs.get("num_punctuation_types", 4)
        return len(found) >= req, f"Số loại dấu câu: {len(found)} (yêu cầu {req})"

    if inst_id == "words:alphabet_loop":
        words = re.findall(r'\b[a-zA-Z]', text.lower())
        if len(words) < 3: return False, "Không đủ số từ"
        passed = all((ord(words[i]) - ord('a')) == (ord(words[i-1]) - ord('a') + 1) % 26 for i in range(1, len(words)))
        return passed, "Vòng lặp bảng chữ cái" if passed else "Thứ tự bảng chữ cái không liên tục"

    if inst_id == "words:prime_lengths":
        primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words: return False, "Không có từ"
        passed = all(len(w) in primes for w in words)
        return passed, "Mọi từ đều có độ dài là số nguyên tố" if passed else "Có từ có độ dài không phải số nguyên tố"

    if inst_id == "format:nested_quotes":
        max_d, stack = 0, []
        for ch in text:
            if ch in ('"', "'"):
                if stack and stack[-1] == ch: stack.pop()
                else: stack.append(ch); max_d = max(max_d, len(stack))
        return max_d >= 3, f"Độ sâu ngoặc kép lồng nhau: {max_d} (yêu cầu >= 3)"

    if inst_id == "count:numbers_count":
        nums = re.findall(r'\b\d+\b', text)
        req = kwargs.get("num_numbers", 3)
        return len(nums) == req, f"Số lượng số tìm thấy: {len(nums)} (yêu cầu {req})"

    if inst_id == "format:title_case":
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words: return False, "Không có từ"
        cap_count = sum(1 for w in words if w[0].isupper())
        ratio = cap_count / len(words)
        return ratio >= 0.75, f"Tỷ lệ viết hoa đầu từ: {ratio:.1%} (yêu cầu >= 75%)"

    if inst_id == "sentence:last_word_first_next":
        sentences = split_sentences(text)
        if len(sentences) < 2: return False, "Cần ít nhất 2 câu"
        for i in range(len(sentences) - 1):
            w_last = re.findall(r'\b\w+\b', sentences[i].lower())
            w_first = re.findall(r'\b\w+\b', sentences[i+1].lower())
            if not w_last or not w_first or w_last[-1] != w_first[0]:
                return False, f"Câu {i+1} kết thúc bằng '{w_last[-1] if w_last else ''}' nhưng câu {i+2} bắt đầu bằng '{w_first[0] if w_first else ''}'"
        return True, "Từ nối giữa các câu hợp lệ"

    if inst_id == "words:no_consecutive_first_letter":
        words = re.findall(r'\b[a-zA-Z]', text.lower())
        if len(words) < 2: return True, "Đạt"
        passed = all(words[i] != words[i+1] for i in range(len(words) - 1))
        return passed, "Không có 2 từ liên tiếp cùng chữ cái đầu" if passed else "Có từ liên tiếp trùng chữ cái đầu"

    if inst_id == "words:limited_repeat":
        words = re.findall(r'\b\w+\b', text.lower())
        if not words: return False, "Rỗng"
        max_rep = Counter(words).most_common(1)[0][1]
        req = kwargs.get("max_repeats", 4)
        return max_rep <= req, f"Từ lặp nhiều nhất: {max_rep} lần (cho phép tối đa {req})"

    if inst_id == "sentence:word_count_step":
        sentences = split_sentences(text)
        if len(sentences) < 2: return False, "Cần ít nhất 2 câu"
        lens = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        step = kwargs.get("step", 2)
        passed = all(lens[i+1] - lens[i] == step for i in range(len(lens)-1))
        return passed, f"Độ dài các câu: {lens} (bước nhảy yêu cầu {step})"

    if inst_id == "format:date_format_list":
        dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', text)
        req = kwargs.get("min_dates", 2)
        return len(dates) >= req, f"Tìm thấy {len(dates)} ngày chuẩn YYYY-MM-DD (yêu cầu >= {req})"

    if inst_id == "format:csv_format":
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        req = kwargs.get("num_rows", 3)
        if len(lines) < req: return False, f"Chỉ có {len(lines)} dòng (yêu cầu {req})"
        passed = all(',' in l for l in lines)
        return passed, "Định dạng CSV hợp lệ" if passed else "Có dòng thiếu dấu phẩy phân tách"

    if inst_id == "format:output_template":
        passed = ("My Answer:" in text and "My Conclusion:" in text and "Future Outlook:" in text)
        return passed, "Đủ 3 đề mục template" if passed else "Thiếu một trong 3 đề mục yêu cầu"

    if inst_id == "words:paragraph_last_first_match":
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not paras: return False, "Rỗng"
        for p in paras:
            words = re.findall(r'\b\w+\b', p.lower())
            if len(words) < 2 or words[0] != words[-1]:
                return False, f"Đoạn có từ đầu '{words[0] if words else ''}' khác từ cuối '{words[-1] if words else ''}'"
        return True, "Từ đầu và cuối các đoạn đều trùng khớp"

    if inst_id == "words:conjunction_count":
        conj_list = {'and', 'but', 'for', 'nor', 'or', 'so', 'yet'}
        words = set(re.findall(r'\b\w+\b', text.lower()))
        used = words.intersection(conj_list)
        req = kwargs.get("min_conjunctions", 3)
        return len(used) >= req, f"Số liên từ: {len(used)} ({used}) (yêu cầu {req})"

    if inst_id == "format:indent_stairs":
        lines = [l for l in text.split('\n') if l.strip()]
        if len(lines) < 3: return False, "Cần ít nhất 3 dòng"
        indents = [len(l) - len(l.lstrip(' ')) for l in lines]
        passed = all(indents[i] < indents[i+1] for i in range(len(indents)-1))
        return passed, f"Độ thụt lề: {indents}"

    if inst_id == "format:special_bullet":
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        bullet_lines = [l for l in lines if l.startswith('->')]
        return len(bullet_lines) >= 2, f"Số dòng bắt đầu bằng '->': {len(bullet_lines)} (yêu cầu >= 2)"

    if inst_id == "manipulation:no_whitespace":
        has_ws = any(c in string.whitespace for c in text)
        return not has_ws, "Không có khoảng trắng" if not has_ws else "Chứa khoảng trắng"

    if inst_id == "sentence:symbol_end":
        sentences = split_sentences(text)
        if not sentences: return False, "Rỗng"
        passed = all(s.endswith('!*') or '!*' in s[-5:] for s in sentences)
        return passed, "Mọi câu đều kết thúc bằng !*" if passed else "Có câu không kết thúc bằng !*"

    raise ValueError(f"Unknown constraint id: {inst_id}")

def verify_hard_constraint(inst_id, kwargs, response_text):
    passed, _ = verify_hard_constraint_detailed(inst_id, kwargs, response_text)
    return passed

print("✓ Engine Code Verifier (Phiên bản v2 - Hỗ trợ Chi Tiết Từng Ràng Buộc) đã sẵn sàng!")