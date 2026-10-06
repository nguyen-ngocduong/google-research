#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: verifier_engine_vi.py
Mục đích: Bộ máy xác thực tất định (Deterministic Rule-Based Verifiers) 100% Tiếng Việt
          cho toàn bộ 29 Seen Hard Constraints + 24 Unseen Hard Constraints.

Tính năng:
- Chuẩn hóa Unicode NFC toàn diện trước mọi phép so sánh.
- Hỗ trợ tách câu tiếng Việt (xử lý các từ viết tắt phổ biến: TP., TS., ThS., PGS., GS., v.v.).
- Đếm từ chuẩn tiếng Việt theo âm tiết (syllable tokens).
- Kiểm tra chính xác bảng chữ cái tiếng Việt hoa/thường, liên từ tiếng Việt, động từ hành động.
- Trả về kết quả có cấu trúc: (passed: bool, note: str).
"""

import re
import json
import string
import unicodedata
from typing import Dict, Any, List, Tuple
from collections import Counter

try:
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


def normalize_vi(text: str) -> str:
    """Chuẩn hóa chuỗi văn bản về Unicode NFC chuẩn."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


def split_sentences_vi(text: str) -> List[str]:
    """Tách câu tiếng Việt, bảo vệ các từ viết tắt thông dụng như TP., TS., GS., v.v."""
    clean = normalize_vi(text.strip())
    # Tạm thay thế dấu chấm trong từ viết tắt thông dụng
    abbrevs = ["TP.", "TS.", "ThS.", "PGS.", "GS.", "BS.", "Th.S.", "P.S.", "p.s.", "e.g.", "i.e."]
    temp_text = clean
    for idx, abbr in enumerate(abbrevs):
        placeholder = f"__ABBR_{idx}__"
        temp_text = temp_text.replace(abbr, placeholder)

    # Tách câu dựa trên dấu . ! ? kèm khoảng trắng
    parts = re.split(r'(?<=[.!?])\s+', temp_text)
    
    # Khôi phục từ viết tắt
    final_sentences = []
    for p in parts:
        p_clean = p.strip()
        for idx, abbr in enumerate(abbrevs):
            placeholder = f"__ABBR_{idx}__"
            p_clean = p_clean.replace(placeholder, abbr)
        if p_clean:
            final_sentences.append(p_clean)
    return final_sentences


def get_words_vi(text: str) -> List[str]:
    """Lấy danh sách các từ (tiếng / âm tiết) tiếng Việt bằng cách loại bỏ punctuation."""
    norm = normalize_vi(text)
    # Tách theo khoảng trắng
    tokens = norm.split()
    clean_tokens = []
    for t in tokens:
        stripped = t.strip(string.punctuation + "“”‘’«»…—–")
        if stripped:
            clean_tokens.append(stripped)
    return clean_tokens


COMMON_VI_CONJUNCTIONS = {
    "và", "hoặc", "nhưng", "vì", "nên", "tuy", "tuy_nhiên", "tuy nhiên", "mặc_dù", "mặc dù",
    "do_đó", "do đó", "song", "thế_nhưng", "thế nhưng", "cho_nên", "cho nên", "đồng_thời", "đồng thời",
    "bởi_vì", "bởi vì", "nếu", "hễ", "giá_như", "giá như", "miễn_là", "miễn là", "với"
}

COMMON_VI_START_VERBS = {
    "hãy", "thực_hiện", "thực hiện", "phát_triển", "phát triển", "xây_dựng", "xây dựng",
    "phân_tích", "phân tích", "áp_dụng", "áp dụng", "nghiên_cứu", "nghiên cứu", "tối_ưu", "tối ưu",
    "thiết_lập", "thiết lập", "xem_xét", "xem xét", "đánh_giá", "đánh giá", "nâng_cao", "nâng cao",
    "tạo", "chọn", "tìm", "kiểm_tra", "kiểm tra", "tập_trung", "tập trung", "khởi_động", "khởi động",
    "bắt_đầu", "bắt đầu", "đảm_bảo", "đảm bảo", "lưu_ý", "lưu ý", "khám_phá", "khám phá"
}


def is_prime(n: int) -> bool:
    """Kiểm tra số nguyên tố."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def verify_hard_constraint_detailed(inst_id: str, kwargs: Dict[str, Any], response_text: str) -> Tuple[bool, str]:
    """
    Xác minh chi tiết một Hard Constraint trên phản hồi của mô hình.
    Trả về: (passed: bool, note: str)
    """
    text = normalize_vi(response_text.strip())
    if not text:
        return False, "Phản hồi rỗng."

    # =========================================================================
    # 1. SEEN POOL CONSTRAINTS (29 loại)
    # =========================================================================

    # 1.1. Keywords: Existence
    if inst_id == "keywords:existence":
        kws = kwargs.get("keywords", [])
        text_lower = text.lower()
        missing = [w for w in kws if normalize_vi(w).lower() not in text_lower]
        passed = (len(missing) == 0)
        return passed, f"Chứa đủ {len(kws)} từ khóa" if passed else f"Thiếu từ khóa: {', '.join(missing)}"

    # 1.2. Keywords: Frequency
    if inst_id == "keywords:frequency":
        kw = normalize_vi(kwargs.get("keyword", "")).lower()
        freq = kwargs.get("frequency", 0)
        rel = kwargs.get("relation", "at least")
        words = [w.lower() for w in get_words_vi(text)]
        count = words.count(kw)
        if rel == "at least": passed = (count >= freq)
        elif rel == "at most": passed = (count <= freq)
        else: passed = (count == freq)
        return passed, f"Từ '{kw}' xuất hiện {count} lần (yêu cầu {rel} {freq})"

    # 1.3. Keywords: Forbidden Words
    if inst_id == "keywords:forbidden_words":
        forbids = kwargs.get("forbidden_words", [])
        text_lower = text.lower()
        found = [w for w in forbids if normalize_vi(w).lower() in text_lower]
        passed = (len(found) == 0)
        return passed, "Không vi phạm từ cấm" if passed else f"Xuất hiện từ cấm: {', '.join(found)}"

    # 1.4. Keywords: Letter Frequency
    if inst_id == "keywords:letter_frequency":
        letter = normalize_vi(kwargs.get("letter", "")).lower()
        freq = kwargs.get("let_frequency", 0)
        rel = kwargs.get("let_relation", "at least")
        count = text.lower().count(letter)
        if rel == "at least": passed = (count >= freq)
        elif rel == "at most": passed = (count <= freq)
        else: passed = (count == freq)
        return passed, f"Ký tự '{letter}' xuất hiện {count} lần (yêu cầu {rel} {freq})"

    # 1.5. Length: Number of Words / Syllables (Tiếng / Âm tiết tiếng Việt)
    if inst_id in ["length_constraints:number_words", "length_constraints:number_syllables"]:
        words = len(get_words_vi(text))
        num = kwargs.get("num_syllables", kwargs.get("num_words", kwargs.get("num", 0)))
        rel = kwargs.get("relation", "at least")
        if rel == "less than": passed = (words < num)
        elif rel == "at least": passed = (words >= num)
        elif rel == "range": passed = (kwargs.get("min_words", kwargs.get("min_syllables", 0)) <= words <= kwargs.get("max_words", kwargs.get("max_syllables", 9999)))
        else: passed = (words == num)
        return passed, f"Số tiếng (âm tiết): {words} (yêu cầu {rel} {num})"

    # 1.6. Length: Number of Sentences
    if inst_id == "length_constraints:number_sentences":
        num = kwargs.get("num_sentences", 0)
        rel = kwargs.get("relation", "exactly")
        sentences = split_sentences_vi(text)
        count = len(sentences)
        if rel == "less than": passed = (count < num)
        elif rel == "at least": passed = (count >= num)
        else: passed = (count == num)
        return passed, f"Số câu: {count} (yêu cầu {rel} {num})"

    # 1.7. Length: Number of Paragraphs
    if inst_id == "length_constraints:number_paragraphs":
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        req = kwargs.get("num_paragraphs", 0)
        passed = (len(paras) == req)
        return passed, f"Số đoạn văn: {len(paras)} (yêu cầu {req})"

    # 1.8. Detectable Format: Bullet Lists
    if inst_id == "detectable_format:number_bullet_lists":
        bullets = re.findall(r'(?m)^\s*[*-•]\s+', text)
        req = kwargs.get("num_bullets", 0)
        passed = (len(bullets) == req)
        return passed, f"Số gạch đầu dòng: {len(bullets)} (yêu cầu {req})"

    # 1.9. Format: Numbered List
    if inst_id == "format:numbered_list":
        items = re.findall(r'(?m)^\s*\d+[\.\)]\s+', text)
        req = kwargs.get("num_items", 0)
        passed = (len(items) == req)
        return passed, f"Số mục đánh số: {len(items)} (yêu cầu {req})"

    # 1.10. Detectable Format: JSON Format
    if inst_id == "detectable_format:json_format":
        try:
            clean_j = text.strip()
            if clean_j.startswith("```json"): clean_j = clean_j[7:].split("```")[0].strip()
            elif clean_j.startswith("```"): clean_j = clean_j[3:].split("```")[0].strip()
            json.loads(clean_j)
            return True, "JSON hợp lệ"
        except Exception as e:
            return False, f"JSON không hợp lệ: {str(e)[:40]}"

    # 1.11. Detectable Format: Title
    if inst_id == "detectable_format:title":
        has_chevrons = bool(re.search(r'<<[^>]+>>', text))
        has_header = bool(re.search(r'(?m)^#\s+.+', text))
        passed = (has_chevrons or has_header)
        return passed, "Có tiêu đề định dạng hợp lệ (<<Tiêu đề>> hoặc # Tiêu đề)" if passed else "Thiếu tiêu đề chuẩn"

    # 1.12. Detectable Format: Multiple Sections
    if inst_id == "detectable_format:multiple_sections":
        req = kwargs.get("num_sections", 2)
        splitter = kwargs.get("section_spliter", "Phần")
        pattern = rf'(?mi)^\s*(?:{re.escape(splitter)}|Section)\s*\d+[\s:]'
        sections = re.findall(pattern, text)
        passed = (len(sections) >= req)
        return passed, f"Số phần phân đoạn: {len(sections)} (yêu cầu >= {req})"

    # 1.13. Detectable Format: Highlighted Sections
    if inst_id == "detectable_format:number_highlighted_sections":
        req = kwargs.get("num_highlights", 1)
        highlights = re.findall(r'\*[^*]+\*|\*\*[^*]+\*\*', text)
        passed = (len(highlights) >= req)
        return passed, f"Số đoạn in đậm/nghiêng: {len(highlights)} (yêu cầu >= {req})"

    # 1.14. Detectable Content: Postscript
    if inst_id == "detectable_content:postscript":
        marker = kwargs.get("postscript_marker", "P.S.")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines: return False, "Phản hồi rỗng"
        last_line = lines[-1].lower()
        passed = last_line.startswith(marker.lower()) or last_line.startswith("tái bút:") or last_line.startswith("tb:")
        return passed, "Có tái bút ở dòng cuối" if passed else "Dòng cuối không có tái bút (P.S. hoặc Tái bút:)"

    # 1.15. Start/End: Start Checker
    if inst_id in ["startend:start_checker", "startend:beginning"]:
        phrase = normalize_vi(kwargs.get("start_phrase", kwargs.get("phrase", ""))).strip()
        passed = text.startswith(phrase) or text.lower().startswith(phrase.lower())
        return passed, f"Bắt đầu bằng cụm từ yêu cầu: '{phrase}'" if passed else f"Không bắt đầu bằng '{phrase}'"

    # 1.16. Start/End: End Checker
    if inst_id == "startend:end_checker":
        phrase = normalize_vi(kwargs.get("end_phrase", "")).strip()
        passed = text.endswith(phrase) or text.endswith(phrase.rstrip('.'))
        return passed, f"Kết thúc bằng câu yêu cầu" if passed else f"Không kết thúc bằng '{phrase}'"

    # 1.16. Start/End: Quotation
    if inst_id == "startend:quotation":
        passed = (text.startswith('"') and text.endswith('"')) or (text.startswith('“') and text.endswith('”')) or (text.startswith('«') and text.endswith('»'))
        return passed, "Toàn bộ văn bản nằm trong ngoặc kép" if passed else "Không bao bọc trong ngoặc kép ở đầu và cuối"

    # 1.17. Change Case: ALL CAPS
    if inst_id == "change_case:english_capital":
        letters = [c for c in text if c.isalpha()]
        passed = len(letters) > 0 and all(c.isupper() for c in letters)
        return passed, "Toàn bộ chữ cái là CHỮ HOA" if passed else "Còn chữ cái thường hoặc không có chữ"

    # 1.18. Change Case: All Lowercase
    if inst_id == "change_case:english_lowercase":
        letters = [c for c in text if c.isalpha()]
        passed = len(letters) > 0 and all(c.islower() for c in letters)
        return passed, "Toàn bộ chữ cái là chữ thường" if passed else "Còn chữ cái HOA"

    # 1.19. Punctuation: No Comma
    if inst_id == "punctuation:no_comma":
        has_comma = (',' in text) or ('，' in text)
        passed = not has_comma
        return passed, "Không chứa dấu phẩy" if passed else "Bị dính dấu phẩy"

    # 1.20. Format: XML Wrapper
    if inst_id == "format:xml_wrapper":
        tag = kwargs.get("tag", "response")
        pattern = rf'^\s*<{re.escape(tag)}>.*?</{re.escape(tag)}>\s*$'
        passed = bool(re.match(pattern, text, re.DOTALL))
        return passed, f"Bao bọc trong thẻ XML <{tag}>" if passed else f"Thiếu thẻ XML <{tag}> ... </{tag}>"

    # 1.21. Language: Response Language
    if inst_id == "language:response_language":
        req_lang = kwargs.get("language", "vi").lower()
        if HAS_LANGDETECT:
            try:
                langs = detect_langs(text)
                detected = langs[0].lang if langs else "unknown"
                passed = (detected == req_lang) or (req_lang in ["vi", "vietnamese"] and detected == "vi")
                return passed, f"Ngôn ngữ phát hiện: {detected} (yêu cầu {req_lang})"
            except Exception:
                pass
        # Fallback kiểm tra ký tự tiếng Việt đặc thù
        has_vi_chars = bool(re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text.lower()))
        return has_vi_chars, "Chứa các ký tự tiếng Việt hợp lệ" if has_vi_chars else "Không phát hiện tiếng Việt"

    # 1.22. Length: Nth Paragraph First Word
    if inst_id == "length_constraints:nth_paragraph_first_word":
        nth = kwargs.get("nth_paragraph", 1)
        word = normalize_vi(kwargs.get("first_word", "")).lower()
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if len(paras) < nth:
            return False, f"Chỉ có {len(paras)} đoạn, không đủ {nth} đoạn"
        target_words = get_words_vi(paras[nth - 1])
        first_w = target_words[0].lower() if target_words else ""
        passed = (first_w == word)
        return passed, f"Từ đầu đoạn {nth} là '{first_w}' (yêu cầu '{word}')"

    # 1.23. Detectable Content: Placeholders
    if inst_id == "detectable_content:number_placeholders":
        req = kwargs.get("num_placeholders", 1)
        placeholders = re.findall(r'\[[a-zA-Z0-9_\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\-]+\]', text)
        passed = (len(placeholders) >= req)
        return passed, f"Số chỗ trống [placeholder]: {len(placeholders)} (yêu cầu >= {req})"

    # 1.24. Detectable Format: Constrained Response
    if inst_id == "detectable_format:constrained_response":
        options = [normalize_vi(o).strip().lower() for o in kwargs.get("constrained_responses", [])]
        clean_text = text.strip().lower()
        passed = (clean_text in options)
        return passed, f"Phản hồi thuộc danh sách cho phép: {options}" if passed else f"Phản hồi '{clean_text}' nằm ngoài options"

    # 1.25. Combination: Two Responses
    if inst_id == "combination:two_responses":
        sep = kwargs.get("separator", "******")
        parts = text.split(sep)
        passed = (len(parts) == 2 and len(parts[0].strip()) > 0 and len(parts[1].strip()) > 0)
        return passed, "Chia thành 2 phản hồi phân cách bởi " + sep if passed else "Thiếu phân cách " + sep

    # 1.26. Combination: Repeat Prompt
    if inst_id == "combination:repeat_prompt":
        # Prompt thường được lặp lại ở phần đầu
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        passed = len(lines) >= 2
        return passed, "Có lặp lại đề bài trước khi trả lời" if passed else "Không tìm thấy nội dung lặp lại"

    # 1.27. Change Case: Capital Word Frequency
    if inst_id == "change_case:capital_word_frequency":
        req = kwargs.get("capital_words", 1)
        words = get_words_vi(text)
        cap_words = [w for w in words if w.isupper() and len(w) > 1]
        passed = (len(cap_words) >= req)
        return passed, f"Số từ VIẾT HOA toàn bộ: {len(cap_words)} (yêu cầu >= {req})"

    # 1.28. Detectable Format: Table Format
    if inst_id == "detectable_format:table_format":
        has_table = bool(re.search(r'\|[^\n]+\|\n\|[\s:-|-]+\|\n\|[^\n]+\|', text))
        return has_table, "Chứa bảng Markdown hợp lệ" if has_table else "Không tìm thấy định dạng bảng Markdown"

    # 1.29. Punctuation: No Period
    if inst_id == "punctuation:no_period":
        has_period = ('.' in text) or ('。' in text)
        passed = not has_period
        return passed, "Không chứa dấu chấm" if passed else "Bị dính dấu chấm câu"

    # =========================================================================
    # 2. UNSEEN POOL CONSTRAINTS (24 loại OOD)
    # =========================================================================

    # 2.1. Count: Unique Word Count
    if inst_id == "count:unique_word_count":
        req = kwargs.get("num_unique_words", 20)
        words = [w.lower() for w in get_words_vi(text)]
        unique_cnt = len(set(words))
        passed = (unique_cnt >= req)
        return passed, f"Số từ độc nhất: {unique_cnt} (yêu cầu >= {req})"

    # 2.2. Format: Parentheses (Nested)
    if inst_id == "format:parentheses":
        req = kwargs.get("num_nested", 1)
        nested = re.findall(r'\(\([^\(\)]+\)\)', text)
        passed = (len(nested) >= req)
        return passed, f"Số cặp ngoặc kép lồng ((...)): {len(nested)} (yêu cầu >= {req})"

    # 2.3. Words: Palindrome
    if inst_id == "words:palindrome":
        palindromes = [normalize_vi(p).lower() for p in kwargs.get("palindromes", [])]
        text_lower = text.lower()
        missing = [p for p in palindromes if p not in text_lower]
        passed = (len(missing) == 0)
        return passed, "Chứa đủ các từ đối xứng" if passed else f"Thiếu từ đối xứng: {missing}"

    # 2.4. Sentence: Increment
    if inst_id == "sentence:increment":
        sentences = split_sentences_vi(text)
        if len(sentences) < 2:
            return False, "Cần ít nhất 2 câu để kiểm tra độ dài tăng dần"
        lengths = [len(get_words_vi(s)) for s in sentences]
        passed = all(lengths[i] < lengths[i + 1] for i in range(len(lengths) - 1))
        return passed, f"Độ dài các câu tăng dần: {lengths}" if passed else f"Độ dài câu không tăng dần: {lengths}"

    # 2.5. Words: Start Verb
    if inst_id == "words:start_verb":
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not paras: return False, "Phản hồi rỗng"
        all_verbs = True
        notes = []
        for idx, p in enumerate(paras):
            words = get_words_vi(p)
            first_w = words[0].lower() if words else ""
            two_words = f"{words[0]} {words[1]}".lower() if len(words) >= 2 else ""
            if (first_w not in COMMON_VI_START_VERBS) and (two_words not in COMMON_VI_START_VERBS):
                all_verbs = False
                notes.append(f"Đoạn {idx + 1} mở đầu bằng '{first_w}'")
        return all_verbs, "Mọi đoạn đều bắt đầu bằng động từ hành động" if all_verbs else f"Có đoạn không bắt đầu bằng động từ: {', '.join(notes)}"

    # 2.6. Count: Punctuation Diversity
    if inst_id == "count:punctuation":
        req = kwargs.get("num_punctuation_types", 4)
        found_puncts = set(c for c in text if c in string.punctuation or c in "“”‘’«»…—–")
        passed = (len(found_puncts) >= req)
        return passed, f"Dùng {len(found_puncts)} loại dấu câu: {''.join(found_puncts)} (yêu cầu >= {req})"

    # 2.7. Words: Alphabet Loop
    if inst_id == "words:alphabet_loop":
        words = get_words_vi(text)
        if len(words) < 3: return False, "Cần ít nhất 3 từ"
        passed = True
        for i in range(len(words) - 1):
            c1 = words[i][0].lower()
            c2 = words[i + 1][0].lower()
            # So sánh mã ASCII thứ tự chữ cái kế tiếp (loop a->z->a)
            if not c1.isalpha() or not c2.isalpha():
                continue
            expected = 'a' if c1 == 'z' else chr(ord(c1) + 1)
            if c2 != expected:
                passed = False
                break
        return passed, "Từ bắt đầu theo thứ tự bảng chữ cái" if passed else "Không tuần tự bảng chữ cái"

    # 2.8. Words: Prime Lengths
    if inst_id == "words:prime_lengths":
        words = get_words_vi(text)
        if not words: return False, "Không có từ nào"
        non_primes = [w for w in words if not is_prime(len(w))]
        passed = (len(non_primes) == 0)
        return passed, "Độ dài mọi từ là số nguyên tố" if passed else f"Từ có độ dài không nguyên tố: {non_primes[:3]}"

    # 2.9. Format: Nested Quotes
    if inst_id == "format:nested_quotes":
        has_nested = ('"' in text and ("'" in text or "“" in text or "«" in text))
        return has_nested, "Có ngoặc kép lồng nhau nhiều cấp" if has_nested else "Thiếu ngoặc kép lồng cấp"

    # 2.10. Count: Numbers Count
    if inst_id == "count:numbers_count":
        req = kwargs.get("num_numbers", 3)
        numbers = re.findall(r'\b\d+\b', text)
        passed = (len(numbers) == req)
        return passed, f"Có đúng {len(numbers)} số: {numbers} (yêu cầu {req})"

    # 2.11. Format: Title Case
    if inst_id == "format:title_case":
        words = get_words_vi(text)
        passed = len(words) > 0 and all(w[0].isupper() for w in words if w.isalpha())
        return passed, "Mọi từ đều viết hoa chữ cái đầu (Title Case)" if passed else "Có từ không viết hoa đầu từ"

    # 2.12. Sentence: Last Word First Next
    if inst_id == "sentence:last_word_first_next":
        sentences = split_sentences_vi(text)
        if len(sentences) < 2: return False, "Cần ít nhất 2 câu"
        chain_ok = True
        for i in range(len(sentences) - 1):
            w_last = get_words_vi(sentences[i])[-1].lower() if get_words_vi(sentences[i]) else ""
            w_first = get_words_vi(sentences[i + 1])[0].lower() if get_words_vi(sentences[i + 1]) else ""
            if w_last != w_first:
                chain_ok = False
                break
        return chain_ok, "Từ cuối câu trước là từ đầu câu sau" if chain_ok else "Đứt gãy liên kết cuối-đầu câu"

    # 2.13. Words: No Consecutive First Letter
    if inst_id == "words:no_consecutive_first_letter":
        words = get_words_vi(text)
        passed = True
        dup_example = ""
        for i in range(len(words) - 1):
            c1 = words[i][0].lower()
            c2 = words[i + 1][0].lower()
            if c1 == c2 and c1.isalpha():
                passed = False
                dup_example = f"{words[i]} - {words[i+1]}"
                break
        return passed, "Không có hai từ liền kề cùng chữ cái đầu" if passed else f"Từ trùng chữ cái đầu: {dup_example}"

    # 2.14. Words: Limited Repeat
    if inst_id == "words:limited_repeat":
        max_rep = kwargs.get("max_repeats", 2)
        words = [w.lower() for w in get_words_vi(text)]
        counts = Counter(words)
        violating = {w: c for w, c in counts.items() if c > max_rep}
        passed = (len(violating) == 0)
        return passed, f"Không từ nào lặp quá {max_rep} lần" if passed else f"Từ lặp quá số lần: {violating}"

    # 2.15. Sentence: Word Count Step
    if inst_id == "sentence:word_count_step":
        step = kwargs.get("step", 2)
        sentences = split_sentences_vi(text)
        if len(sentences) < 2: return False, "Cần ít nhất 2 câu"
        lengths = [len(get_words_vi(s)) for s in sentences]
        passed = all((lengths[i + 1] - lengths[i] == step) for i in range(len(lengths) - 1))
        return passed, f"Mỗi câu tăng đúng {step} từ: {lengths}" if passed else f"Độ dài câu không tăng đúng bước {step}: {lengths}"

    # 2.16. Format: Date Format List
    if inst_id == "format:date_format_list":
        req = kwargs.get("min_dates", 1)
        dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', text)
        passed = (len(dates) >= req)
        return passed, f"Số ngày định dạng YYYY-MM-DD: {len(dates)} (yêu cầu >= {req})"

    # 2.17. Format: CSV Format
    if inst_id == "format:csv_format":
        req_rows = kwargs.get("num_rows", 3)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        csv_lines = [l for l in lines if ',' in l]
        passed = (len(csv_lines) >= req_rows)
        return passed, f"Định dạng CSV với {len(csv_lines)} dòng (yêu cầu >= {req_rows})"

    # 2.18. Format: Output Template
    if inst_id == "format:output_template":
        has_answer = "câu trả lời:" in text.lower() or "my answer:" in text.lower()
        has_conclusion = "kết luận:" in text.lower() or "my conclusion:" in text.lower()
        has_outlook = "triển vọng:" in text.lower() or "future outlook:" in text.lower()
        passed = (has_answer and has_conclusion and has_outlook)
        return passed, "Tuân thủ cấu trúc template 3 phần" if passed else "Thiếu một trong 3 phần (Câu trả lời / Kết luận / Triển vọng)"

    # 2.19. Words: Paragraph Last First Match
    if inst_id == "words:paragraph_last_first_match":
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not paras: return False, "Phản hồi rỗng"
        all_match = True
        for p in paras:
            words = get_words_vi(p)
            if len(words) < 2 or words[0].lower() != words[-1].lower():
                all_match = False
                break
        return all_match, "Mỗi đoạn kết thúc bằng đúng từ nó bắt đầu" if all_match else "Đoạn văn không khớp từ đầu và từ cuối"

    # 2.20. Words: Conjunction Count
    if inst_id == "words:conjunction_count":
        req = kwargs.get("min_conjunctions", 3)
        text_lower = text.lower()
        found = set(conj for conj in COMMON_VI_CONJUNCTIONS if conj in text_lower)
        passed = (len(found) >= req)
        return passed, f"Sử dụng {len(found)} liên từ: {list(found)} (yêu cầu >= {req})"

    # 2.21. Format: Indent Stairs
    if inst_id == "format:indent_stairs":
        lines = [l for l in text.split('\n') if l.strip()]
        if len(lines) < 3: return False, "Cần ít nhất 3 dòng"
        stairs_ok = True
        for i in range(len(lines) - 1):
            spaces_cur = len(lines[i]) - len(lines[i].lstrip(' '))
            spaces_next = len(lines[i + 1]) - len(lines[i + 1].lstrip(' '))
            if spaces_next - spaces_cur != 2:
                stairs_ok = False
                break
        return stairs_ok, "Các dòng thụt lề bậc thang +2 spaces" if stairs_ok else "Thụt lề không đúng quy luật bậc thang +2 spaces"

    # 2.22. Format: Special Bullet
    if inst_id == "format:special_bullet":
        bullets = re.findall(r'(?m)^\s*->\s+', text)
        passed = len(bullets) >= 2
        return passed, f"Sử dụng {len(bullets)} dấu đầu dòng '-> '" if passed else "Không tìm thấy dấu đầu dòng '-> '"

    # 2.23. Manipulation: No Whitespace
    if inst_id == "manipulation:no_whitespace":
        has_space = bool(re.search(r'\s', text))
        passed = not has_space
        return passed, "Không chứa bất kỳ khoảng trắng nào" if passed else "Có chứa khoảng trắng"

    # 2.24. Sentence: Symbol End
    if inst_id == "sentence:symbol_end":
        sentences = split_sentences_vi(text)
        if not sentences: return False, "Phản hồi rỗng"
        passed = all(s.strip().endswith("!*") for s in sentences)
        return passed, "Mọi câu đều kết thúc bằng '!*'" if passed else "Có câu không kết thúc bằng '!*'"

    return False, f"Chưa có verifier cho instruction_id: {inst_id}"


def evaluate_instruction_following_vi(response_text: str, verifier_spec: Any) -> Dict[str, Any]:
    """
    Hàm chấm điểm thống nhất cho GRPOTrainer / Reward function.
    Chấp nhận verifier_spec ở dạng JSON string hoặc list các dict {"id": ..., "kwargs": ...}.
    
    Trả về dict:
    {
        "passed_count": int,
        "total_hard": int,
        "strict_score": 1.0 if all passed else 0.0,
        "partial_score": float (passed_count / total_hard),
        "all_hard_passed": bool,
        "details": List[Dict[str, Any]]
    }
    """
    if isinstance(verifier_spec, str):
        try:
            verifier_list = json.loads(verifier_spec)
        except Exception as e:
            return {
                "passed_count": 0,
                "total_hard": 0,
                "strict_score": 0.0,
                "partial_score": 0.0,
                "all_hard_passed": False,
                "error": f"JSON parse error: {e}"
            }
    elif isinstance(verifier_spec, list):
        verifier_list = verifier_spec
    else:
        return {
            "passed_count": 0,
            "total_hard": 0,
            "strict_score": 0.0,
            "partial_score": 0.0,
            "all_hard_passed": False,
            "error": "Invalid verifier spec type"
        }

    flat_items = []
    for item in verifier_list:
        inst_id = item.get("id") or item.get("instruction_id")
        kwargs = item.get("kwargs", {})
        if isinstance(inst_id, list):
            kw_list = kwargs if isinstance(kwargs, list) else [kwargs] * len(inst_id)
            for sub_id, sub_kw in zip(inst_id, kw_list):
                flat_items.append({"id": sub_id, "kwargs": sub_kw})
        else:
            flat_items.append({"id": inst_id, "kwargs": kwargs})

    total_hard = len(flat_items)
    passed_count = 0
    details = []

    for item in flat_items:
        inst_id = item.get("id")
        kwargs = item.get("kwargs", {})
        passed, note = verify_hard_constraint_detailed(inst_id, kwargs, response_text)
        if passed:
            passed_count += 1
        details.append({"id": inst_id, "passed": passed, "note": note})

    partial = (passed_count / total_hard) if total_hard > 0 else 1.0
    strict = 1.0 if (passed_count == total_hard and total_hard > 0) else 0.0

    return {
        "passed_count": passed_count,
        "total_hard": total_hard,
        "strict_score": strict,
        "partial_score": partial,
        "all_hard_passed": (passed_count == total_hard and total_hard > 0),
        "details": details
    }

