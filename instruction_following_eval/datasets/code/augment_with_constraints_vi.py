#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 1: augment_with_constraints_vi.py
Mục đích: Sinh tập Hard Constraints Tiếng Việt (Seen 29 loại hoặc Unseen 24 loại)
          và xử lý theo Batch (mặc định 16 mẫu / batch).

Tính năng:
- Lấy input từ 'datasets/raw/base_vi.jsonl'.
- Chọn ngẫu nhiên k (1 đến 5) ràng buộc không xung đột (Conflict Groups Engine).
- 100% Templates tiếng Việt, chuẩn hóa Unicode NFC.
- Chia nhỏ theo từng Batch: mỗi batch chứa đúng 16 mẫu.
- Lưu kết quả vào 'results/module_hard_constraints/module1_hard_batch_{id}_{timestamp}.json'.
- Hỗ trợ các cờ: --id, --start-idx, --end-idx, --resume, --batch-size 16, --split (seen/unseen).
"""

import os
import sys
import json
import time
import random
import re
import argparse
import datetime
import unicodedata
from typing import List, Dict, Any, Tuple, Optional

# Danh sách từ vựng & cụm từ tiếng Việt mẫu
WORD_BANK_VI = [
    "công nghệ", "chiến lược", "đổi mới", "bền vững", "giải pháp",
    "phát triển", "hài hòa", "sáng tạo", "thách thức", "cân bằng",
    "hiệu quả", "tiềm năng", "xu hướng", "nền tảng", "tối ưu",
    "hội nhập", "chuyển đổi", "trách nhiệm", "khám phá", "đột phá",
    "trí tuệ", "thông tin", "tri thức", "kết nối", "chất lượng"
]

FORBIDDEN_WORDS_POOL_VI = [
    ["không", "chưa", "chẳng"],
    ["rất", "quá", "lắm"],
    ["nhưng", "tuy", "song"],
    ["luôn", "mãi", "suốt"],
    ["tốt", "hay", "tuyệt"]
]

END_PHRASES_VI = [
    "Trên đây là toàn bộ nội dung trình bày.",
    "Hy vọng thông tin này hữu ích cho bạn.",
    "Kết thúc phần trả lời.",
    "Vui lòng cho biết nếu bạn cần thêm chi tiết.",
    "Đó là tổng quan đầy đủ về chủ đề này.",
    "Chúc bạn thực hiện thành công mục tiêu."
]

CONSTRAINED_RESPONSES_POOL_VI = [
    ["Đồng ý", "Không đồng ý", "Phân vân"],
    ["Phương án A", "Phương án B", "Phương án C"],
    ["Đúng", "Sai", "Chưa xác định"],
    ["Tích cực", "Tiêu cực", "Trung lập"]
]


def norm_vi(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# ==============================================================================
# 1. GENERATOR FUNCTIONS CHO SEEN POOL (29 Loại)
# ==============================================================================

def make_seen_keywords_existence() -> Tuple[str, str, Dict[str, Any]]:
    kws = random.sample(WORD_BANK_VI, random.randint(1, 3))
    text = f"Bắt buộc chứa từ khóa '{', '.join(kws)}' trong câu trả lời."
    return "keywords:existence", text, {"keywords": kws}

def make_seen_keywords_frequency() -> Tuple[str, str, Dict[str, Any]]:
    kw = random.choice(WORD_BANK_VI)
    freq = random.randint(2, 4)
    rel = random.choice(["at least", "exactly"])
    rel_vi = "ít nhất" if rel == "at least" else "chính xác"
    text = f"Từ khóa '{kw}' phải xuất hiện {rel_vi} {freq} lần."
    return "keywords:frequency", text, {"keyword": kw, "frequency": freq, "relation": rel}

def make_seen_keywords_forbidden() -> Tuple[str, str, Dict[str, Any]]:
    forbids = random.choice(FORBIDDEN_WORDS_POOL_VI)
    text = f"Tuyệt đối không được chứa bất kỳ từ nào sau đây: {', '.join(forbids)}."
    return "keywords:forbidden_words", text, {"forbidden_words": forbids}

def make_seen_keywords_letter_frequency() -> Tuple[str, str, Dict[str, Any]]:
    letter = random.choice(["a", "e", "i", "o", "u"])
    freq = random.randint(5, 12)
    rel = random.choice(["at least", "at most"])
    rel_vi = "ít nhất" if rel == "at least" else "nhiều nhất"
    text = f"Chữ cái '{letter}' phải xuất hiện {rel_vi} {freq} lần trong toàn bộ bài viết."
    return "keywords:letter_frequency", text, {"letter": letter, "let_frequency": freq, "let_relation": rel}

def make_seen_length_words() -> Tuple[str, str, Dict[str, Any]]:
    rel = random.choice(["at least", "less than"])
    if rel == "at least":
        n = random.randint(100, 250)
        text = f"Câu trả lời của bạn phải có ít nhất {n} từ."
        kwargs = {"num_words": n, "relation": "at least"}
    else:
        n = random.randint(150, 350)
        text = f"Câu trả lời của bạn phải có ít hơn {n} từ."
        kwargs = {"num_words": n, "relation": "less than"}
    return "length_constraints:number_words", text, kwargs

def make_seen_length_sentences() -> Tuple[str, str, Dict[str, Any]]:
    rel = random.choice(["at least", "less than", "exactly"])
    if rel == "exactly":
        n = random.randint(3, 6)
        text = f"Toàn bộ phản hồi phải chứa chính xác {n} câu."
        kwargs = {"num_sentences": n, "relation": "exactly"}
    elif rel == "at least":
        n = random.randint(4, 7)
        text = f"Phản hồi phải chứa ít nhất {n} câu."
        kwargs = {"num_sentences": n, "relation": "at least"}
    else:
        n = random.randint(5, 9)
        text = f"Phản hồi phải chứa ít hơn {n} câu."
        kwargs = {"num_sentences": n, "relation": "less than"}
    return "length_constraints:number_sentences", text, kwargs

def make_seen_length_paragraphs() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 4)
    text = f"Phản hồi phải được cấu trúc thành chính xác {n} đoạn văn, phân cách bằng dòng trống kép."
    return "length_constraints:number_paragraphs", text, {"num_paragraphs": n}

def make_seen_bullet_lists() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(3, 5)
    text = f"Phải chứa chính xác {n} mục gạch đầu dòng định dạng bằng ký tự '*' hoặc '-'."
    return "detectable_format:number_bullet_lists", text, {"num_bullets": n}

def make_seen_numbered_list() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(3, 5)
    text = f"Định dạng nội dung thành danh sách đánh số với chính xác {n} mục (1., 2., ...)."
    return "format:numbered_list", text, {"num_items": n}

def make_seen_json_format() -> Tuple[str, str, Dict[str, Any]]:
    text = "Toàn bộ phản hồi phải ở định dạng JSON hợp lệ, không kèm văn bản giải thích bên ngoài."
    return "detectable_format:json_format", text, {}

def make_seen_title() -> Tuple[str, str, Dict[str, Any]]:
    text = "Đặt một tiêu đề ở dòng đầu tiên được bao bọc trong dấu ngoặc nhọn kép, ví dụ: <<Tiêu đề bài viết>>."
    return "detectable_format:title", text, {}

def make_seen_multiple_sections() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 3)
    text = f"Chia bài viết thành {n} phần, đánh dấu rõ ràng là 'Phần 1:', 'Phần 2:', v.v."
    return "detectable_format:multiple_sections", text, {"num_sections": n, "section_spliter": "Phần"}

def make_seen_highlighted_sections() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 4)
    text = f"Làm nổi bật ít nhất {n} cụm từ hoặc phần quan trọng bằng cách dùng dấu sao Markdown (*chữ nghiêng* hoặc **chữ đậm**)."
    return "detectable_format:number_highlighted_sections", text, {"num_highlights": n}

def make_seen_postscript() -> Tuple[str, str, Dict[str, Any]]:
    text = "Thêm một dòng tái bút bắt đầu bằng 'Tái bút:' hoặc 'P.S.' ở cuối cùng của câu trả lời."
    return "detectable_content:postscript", text, {"postscript_marker": "P.S."}

def make_seen_end_checker() -> Tuple[str, str, Dict[str, Any]]:
    phrase = random.choice(END_PHRASES_VI)
    text = f"Câu trả lời của bạn phải kết thúc bằng chính xác câu sau: '{phrase}'"
    return "startend:end_checker", text, {"end_phrase": phrase}

def make_seen_quotation() -> Tuple[str, str, Dict[str, Any]]:
    text = "Bao bọc toàn bộ câu trả lời của bạn trong dấu ngoặc kép đôi (\"...\")."
    return "startend:quotation", text, {}

def make_seen_capital() -> Tuple[str, str, Dict[str, Any]]:
    text = "Toàn bộ phản hồi phải được viết bằng CHỮ IN HOA (không sử dụng chữ cái thường)."
    return "change_case:english_capital", text, {}

def make_seen_lowercase() -> Tuple[str, str, Dict[str, Any]]:
    text = "Toàn bộ phản hồi phải được viết bằng chữ thường (không sử dụng bất kỳ chữ in hoa nào)."
    return "change_case:english_lowercase", text, {}

def make_seen_no_comma() -> Tuple[str, str, Dict[str, Any]]:
    text = "Tuyệt đối không sử dụng bất kỳ dấu phẩy (',') nào trong toàn bộ bài viết."
    return "punctuation:no_comma", text, {}

def make_seen_xml_wrapper() -> Tuple[str, str, Dict[str, Any]]:
    text = "Bao bọc toàn bộ nội dung câu trả lời bên trong cặp thẻ XML <response> và </response>."
    return "format:xml_wrapper", text, {"tag": "response"}

def make_seen_response_language() -> Tuple[str, str, Dict[str, Any]]:
    text = "Toàn bộ câu trả lời của bạn phải được viết bằng tiếng Việt chuẩn mực."
    return "language:response_language", text, {"language": "vi"}

def make_seen_nth_para_first_word() -> Tuple[str, str, Dict[str, Any]]:
    w = random.choice(["Đầu tiên", "Trước hết", "Ngoài ra", "Cuối cùng"])
    text = f"Đoạn văn thứ 2 phải bắt đầu bằng chính xác từ '{w}'."
    return "length_constraints:nth_paragraph_first_word", text, {"nth_paragraph": 2, "first_word": w}

def make_seen_placeholders() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 4)
    text = f"Bao gồm ít nhất {n} chỗ trống dạng giữ chỗ được đặt trong ngoặc vuông, ví dụ: [họ và tên], [ngày tháng], [địa chỉ]."
    return "detectable_content:number_placeholders", text, {"num_placeholders": n}

def make_seen_constrained_response() -> Tuple[str, str, Dict[str, Any]]:
    opts = random.choice(CONSTRAINED_RESPONSES_POOL_VI)
    text = f"Câu trả lời chỉ được phép là một trong các lựa chọn sau: {', '.join([f'\'{o}\'' for o in opts])}."
    return "detectable_format:constrained_response", text, {"constrained_responses": opts}

def make_seen_two_responses() -> Tuple[str, str, Dict[str, Any]]:
    text = "Cung cấp hai câu trả lời độc lập riêng biệt cho câu hỏi. Phân tách hai câu trả lời bằng ký hiệu '******'."
    return "combination:two_responses", text, {"separator": "******"}

def make_seen_repeat_prompt() -> Tuple[str, str, Dict[str, Any]]:
    text = "Trước tiên hãy lặp lại nguyên văn câu hỏi/yêu cầu của đề bài, sau đó mới đưa ra câu trả lời."
    return "combination:repeat_prompt", text, {}

def make_seen_capital_word_frequency() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(3, 6)
    text = f"Trong bài viết phải chứa ít nhất {n} từ được VIẾT HOA TOÀN BỘ."
    return "change_case:capital_word_frequency", text, {"capital_words": n}

def make_seen_table_format() -> Tuple[str, str, Dict[str, Any]]:
    text = "Trình bày ít nhất một phần nội dung câu trả lời dưới dạng bảng biểu Markdown."
    return "detectable_format:table_format", text, {}

def make_seen_no_period() -> Tuple[str, str, Dict[str, Any]]:
    text = "Tuyệt đối không được sử dụng bất kỳ dấu chấm câu ('.') nào trong toàn bộ phản hồi."
    return "punctuation:no_period", text, {}


# Danh mục Seen Pool (29 loại) kèm Conflict Groups
SEEN_CONSTRAINT_DEFINITIONS = [
    {"id": "keywords:existence", "func": make_seen_keywords_existence, "group": "KEYWORDS", "conflicts_with": ["JSON"]},
    {"id": "keywords:frequency", "func": make_seen_keywords_frequency, "group": "KEYWORDS", "conflicts_with": ["JSON"]},
    {"id": "keywords:forbidden_words", "func": make_seen_keywords_forbidden, "group": "KEYWORDS_FORBIDDEN", "conflicts_with": []},
    {"id": "keywords:letter_frequency", "func": make_seen_keywords_letter_frequency, "group": "LETTER", "conflicts_with": ["ALL_CAPS", "ALL_LOWER"]},
    {"id": "length_constraints:number_words", "func": make_seen_length_words, "group": "LENGTH_WORD", "conflicts_with": ["LENGTH_SENTENCE", "CONSTRAINED_RESP"]},
    {"id": "length_constraints:number_sentences", "func": make_seen_length_sentences, "group": "LENGTH_SENTENCE", "conflicts_with": ["LENGTH_WORD", "PARAGRAPHS"]},
    {"id": "length_constraints:number_paragraphs", "func": make_seen_length_paragraphs, "group": "PARAGRAPHS", "conflicts_with": ["LENGTH_SENTENCE", "JSON", "CONSTRAINED_RESP"]},
    {"id": "detectable_format:number_bullet_lists", "func": make_seen_bullet_lists, "group": "LIST_FORMAT", "conflicts_with": ["JSON", "NUMBERED_LIST", "ALL_CAPS", "CONSTRAINED_RESP"]},
    {"id": "format:numbered_list", "func": make_seen_numbered_list, "group": "NUMBERED_LIST", "conflicts_with": ["JSON", "LIST_FORMAT", "CONSTRAINED_RESP"]},
    {"id": "detectable_format:json_format", "func": make_seen_json_format, "group": "JSON", "conflicts_with": ["LIST_FORMAT", "NUMBERED_LIST", "PARAGRAPHS", "POSTSCRIPT", "SECTIONS", "XML_TAG", "ALL_CAPS", "TABLE"]},
    {"id": "detectable_format:title", "func": make_seen_title, "group": "TITLE", "conflicts_with": ["JSON"]},
    {"id": "detectable_format:multiple_sections", "func": make_seen_multiple_sections, "group": "SECTIONS", "conflicts_with": ["JSON", "PARAGRAPHS", "CONSTRAINED_RESP"]},
    {"id": "detectable_format:number_highlighted_sections", "func": make_seen_highlighted_sections, "group": "HIGHLIGHT", "conflicts_with": ["JSON", "ALL_CAPS"]},
    {"id": "detectable_content:postscript", "func": make_seen_postscript, "group": "POSTSCRIPT", "conflicts_with": ["JSON", "END_SENTENCE"]},
    {"id": "startend:end_checker", "func": make_seen_end_checker, "group": "END_SENTENCE", "conflicts_with": ["JSON", "POSTSCRIPT", "ALL_CAPS", "ALL_LOWER"]},
    {"id": "startend:quotation", "func": make_seen_quotation, "group": "QUOTATION", "conflicts_with": []},
    {"id": "change_case:english_capital", "func": make_seen_capital, "group": "ALL_CAPS", "conflicts_with": ["ALL_LOWER", "LETTER", "HIGHLIGHT", "LIST_FORMAT", "END_SENTENCE", "JSON"]},
    {"id": "change_case:english_lowercase", "func": make_seen_lowercase, "group": "ALL_LOWER", "conflicts_with": ["ALL_CAPS", "LETTER", "END_SENTENCE"]},
    {"id": "punctuation:no_comma", "func": make_seen_no_comma, "group": "NO_COMMA", "conflicts_with": []},
    {"id": "format:xml_wrapper", "func": make_seen_xml_wrapper, "group": "XML_TAG", "conflicts_with": ["JSON"]},
    {"id": "language:response_language", "func": make_seen_response_language, "group": "LANGUAGE", "conflicts_with": []},
    {"id": "length_constraints:nth_paragraph_first_word", "func": make_seen_nth_para_first_word, "group": "PARA_WORD", "conflicts_with": ["JSON"]},
    {"id": "detectable_content:number_placeholders", "func": make_seen_placeholders, "group": "PLACEHOLDERS", "conflicts_with": ["JSON"]},
    {"id": "detectable_format:constrained_response", "func": make_seen_constrained_response, "group": "CONSTRAINED_RESP", "conflicts_with": ["LENGTH_WORD", "PARAGRAPHS", "SECTIONS", "LIST_FORMAT", "NUMBERED_LIST", "TABLE", "TWO_RESPONSES"]},
    {"id": "combination:two_responses", "func": make_seen_two_responses, "group": "TWO_RESPONSES", "conflicts_with": ["JSON", "CONSTRAINED_RESP"]},
    {"id": "combination:repeat_prompt", "func": make_seen_repeat_prompt, "group": "REPEAT_PROMPT", "conflicts_with": ["JSON", "CONSTRAINED_RESP"]},
    {"id": "change_case:capital_word_frequency", "func": make_seen_capital_word_frequency, "group": "CAPITAL_WORDS", "conflicts_with": ["ALL_CAPS", "ALL_LOWER"]},
    {"id": "detectable_format:table_format", "func": make_seen_table_format, "group": "TABLE", "conflicts_with": ["JSON", "LIST_FORMAT", "NUMBERED_LIST", "CONSTRAINED_RESP"]},
    {"id": "punctuation:no_period", "func": make_seen_no_period, "group": "NO_PERIOD", "conflicts_with": ["JSON"]}
]


# ==============================================================================
# 2. GENERATOR FUNCTIONS CHO UNSEEN POOL (24 Loại OOD)
# ==============================================================================

def make_unseen_unique_words() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(25, 50)
    text = f"Câu trả lời của bạn phải chứa ít nhất {n} từ vựng độc nhất (không tính từ lặp lại)."
    return "count:unique_word_count", text, {"num_unique_words": n}

def make_unseen_parentheses() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(1, 3)
    text = f"Sử dụng ít nhất {n} trường hợp chứa dấu ngoặc đơn lồng nhau ((dạng như thế này))."
    return "format:parentheses", text, {"num_nested": n}

def make_unseen_palindrome() -> Tuple[str, str, Dict[str, Any]]:
    pal_pool = ["radar", "level", "civic", "rotor", "madam", "kayak"]
    chosen = random.sample(pal_pool, 2)
    text = f"Bắt buộc chứa các từ đối xứng sau trong phản hồi: '{chosen[0]}' và '{chosen[1]}'."
    return "words:palindrome", text, {"palindromes": chosen}

def make_unseen_sentence_increment() -> Tuple[str, str, Dict[str, Any]]:
    text = "Mỗi câu tiếp theo trong bài viết phải chứa số lượng từ nhiều hơn câu liền trước nó."
    return "sentence:increment", text, {}

def make_unseen_start_verb() -> Tuple[str, str, Dict[str, Any]]:
    text = "Mỗi đoạn văn trong bài viết bắt buộc phải bắt đầu bằng một động từ hành động mạnh mẽ."
    return "words:start_verb", text, {}

def make_unseen_count_punctuation() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(4, 6)
    text = f"Sử dụng ít nhất {n} loại dấu câu khác nhau trong toàn bộ bài viết (ví dụ: . , ! ? : ; -)."
    return "count:punctuation", text, {"num_punctuation_types": n}

def make_unseen_alphabet_loop() -> Tuple[str, str, Dict[str, Any]]:
    text = "Các từ trong một câu văn phải lần lượt bắt đầu bằng các chữ cái liên tiếp theo thứ tự bảng chữ cái."
    return "words:alphabet_loop", text, {}

def make_unseen_prime_lengths() -> Tuple[str, str, Dict[str, Any]]:
    text = "Độ dài ký tự của mỗi từ trong câu trả lời bắt buộc phải là một số nguyên tố (2, 3, 5, 7, 11...)."
    return "words:prime_lengths", text, {}

def make_unseen_nested_quotes() -> Tuple[str, str, Dict[str, Any]]:
    text = "Bao gồm trích dẫn lồng trong trích dẫn nhiều cấp (ít nhất 2 cấp ngoặc kép khác nhau)."
    return "format:nested_quotes", text, {}

def make_unseen_numbers_count() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(3, 5)
    text = f"Câu trả lời phải chứa chính xác {n} con số được viết bằng chữ số (ví dụ: 1, 2, 3)."
    return "count:numbers_count", text, {"num_numbers": n}

def make_unseen_title_case() -> Tuple[str, str, Dict[str, Any]]:
    text = "Viết toàn bộ câu trả lời theo kiểu Title Case (viết hoa chữ cái đầu tiên của từng từ)."
    return "format:title_case", text, {}

def make_unseen_last_first_chain() -> Tuple[str, str, Dict[str, Any]]:
    text = "Từ cuối cùng của mỗi câu phải trở thành từ đầu tiên của câu tiếp theo."
    return "sentence:last_word_first_next", text, {}

def make_unseen_no_consecutive_letter() -> Tuple[str, str, Dict[str, Any]]:
    text = "Tuyệt đối không có hai từ liền kề nhau có cùng chữ cái bắt đầu."
    return "words:no_consecutive_first_letter", text, {}

def make_unseen_limited_repeat() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 4)
    text = f"Không được phép lặp lại bất kỳ từ nào quá {n} lần trong toàn bộ câu trả lời."
    return "words:limited_repeat", text, {"max_repeats": n}

def make_unseen_word_count_step() -> Tuple[str, str, Dict[str, Any]]:
    step = random.randint(2, 3)
    text = f"Mỗi câu tiếp theo phải chứa nhiều hơn câu liền trước đúng {step} từ."
    return "sentence:word_count_step", text, {"step": step}

def make_unseen_date_format_list() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 3)
    text = f"Bao gồm ít nhất {n} mốc thời gian được định dạng theo chuẩn YYYY-MM-DD (ví dụ: 2026-10-05)."
    return "format:date_format_list", text, {"min_dates": n}

def make_unseen_csv_format() -> Tuple[str, str, Dict[str, Any]]:
    rows = random.randint(3, 5)
    text = f"Định dạng toàn bộ câu trả lời dưới dạng dữ liệu CSV với chính xác {rows} dòng dữ liệu."
    return "format:csv_format", text, {"num_rows": rows}

def make_unseen_output_template() -> Tuple[str, str, Dict[str, Any]]:
    text = "Sử dụng chính xác cấu trúc mẫu sau:\nCâu trả lời: [nội dung]\nKết luận: [nội dung]\nTriển vọng: [nội dung]"
    return "format:output_template", text, {}

def make_unseen_paragraph_match() -> Tuple[str, str, Dict[str, Any]]:
    text = "Mỗi đoạn văn trong bài viết phải kết thúc bằng chính xác từ mà nó đã bắt đầu."
    return "words:paragraph_last_first_match", text, {}

def make_unseen_conjunction_count() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(3, 5)
    text = f"Sử dụng ít nhất {n} liên từ tiếng Việt khác nhau trong bài viết (ví dụ: và, hoặc, nhưng, vì, nên, tuy nhiên)."
    return "words:conjunction_count", text, {"min_conjunctions": n}

def make_unseen_indent_stairs() -> Tuple[str, str, Dict[str, Any]]:
    text = "Tạo hiệu ứng bậc thang bằng cách thụt lề đầu dòng tăng dần +2 dấu cách cho mỗi dòng tiếp theo."
    return "format:indent_stairs", text, {}

def make_unseen_special_bullet() -> Tuple[str, str, Dict[str, Any]]:
    text = "Sử dụng ký hiệu '-> ' thay thế cho dấu đầu dòng thông thường cho tất cả các mục."
    return "format:special_bullet", text, {}

def make_unseen_no_whitespace() -> Tuple[str, str, Dict[str, Any]]:
    text = "Toàn bộ phản hồi tuyệt đối không được chứa bất kỳ ký tự khoảng trắng hay xuống dòng nào."
    return "manipulation:no_whitespace", text, {}

def make_unseen_symbol_end() -> Tuple[str, str, Dict[str, Any]]:
    text = "Mọi câu trong phản hồi bắt buộc phải kết thúc bằng cặp ký hiệu đặc biệt '!*'."
    return "sentence:symbol_end", text, {}


UNSEEN_CONSTRAINT_DEFINITIONS = [
    {"id": "count:unique_word_count", "func": make_unseen_unique_words, "group": "UNIQUE_WORDS", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "format:parentheses", "func": make_unseen_parentheses, "group": "PARENTHESES", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "words:palindrome", "func": make_unseen_palindrome, "group": "PALINDROME", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "sentence:increment", "func": make_unseen_sentence_increment, "group": "SENTENCE_INCREMENT", "conflicts_with": ["SENTENCE_STEP", "NO_WHITESPACE"]},
    {"id": "words:start_verb", "func": make_unseen_start_verb, "group": "START_VERB", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "count:punctuation", "func": make_unseen_count_punctuation, "group": "PUNCT_DIVERSITY", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "words:alphabet_loop", "func": make_unseen_alphabet_loop, "group": "ALPHABET_LOOP", "conflicts_with": ["NO_CONSEC_LETTER", "PRIME_LENGTHS", "NO_WHITESPACE"]},
    {"id": "words:prime_lengths", "func": make_unseen_prime_lengths, "group": "PRIME_LENGTHS", "conflicts_with": ["ALPHABET_LOOP", "NO_WHITESPACE"]},
    {"id": "format:nested_quotes", "func": make_unseen_nested_quotes, "group": "NESTED_QUOTES", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "count:numbers_count", "func": make_unseen_numbers_count, "group": "NUMBERS_COUNT", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "format:title_case", "func": make_unseen_title_case, "group": "TITLE_CASE", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "sentence:last_word_first_next", "func": make_unseen_last_first_chain, "group": "CHAIN_WORDS", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "words:no_consecutive_first_letter", "func": make_unseen_no_consecutive_letter, "group": "NO_CONSEC_LETTER", "conflicts_with": ["ALPHABET_LOOP", "NO_WHITESPACE"]},
    {"id": "words:limited_repeat", "func": make_unseen_limited_repeat, "group": "LIMITED_REPEAT", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "sentence:word_count_step", "func": make_unseen_word_count_step, "group": "SENTENCE_STEP", "conflicts_with": ["SENTENCE_INCREMENT", "NO_WHITESPACE"]},
    {"id": "format:date_format_list", "func": make_unseen_date_format_list, "group": "DATE_FORMAT", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "format:csv_format", "func": make_unseen_csv_format, "group": "CSV_FORMAT", "conflicts_with": ["OUTPUT_TEMPLATE", "INDENT_STAIRS", "NO_WHITESPACE"]},
    {"id": "format:output_template", "func": make_unseen_output_template, "group": "OUTPUT_TEMPLATE", "conflicts_with": ["CSV_FORMAT", "NO_WHITESPACE"]},
    {"id": "words:paragraph_last_first_match", "func": make_unseen_paragraph_match, "group": "PARA_MATCH", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "words:conjunction_count", "func": make_unseen_conjunction_count, "group": "CONJUNCTION_COUNT", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "format:indent_stairs", "func": make_unseen_indent_stairs, "group": "INDENT_STAIRS", "conflicts_with": ["CSV_FORMAT", "NO_WHITESPACE"]},
    {"id": "format:special_bullet", "func": make_unseen_special_bullet, "group": "SPECIAL_BULLET", "conflicts_with": ["NO_WHITESPACE"]},
    {"id": "manipulation:no_whitespace", "func": make_unseen_no_whitespace, "group": "NO_WHITESPACE", "conflicts_with": [
        "UNIQUE_WORDS", "PARENTHESES", "PALINDROME", "SENTENCE_INCREMENT", "START_VERB", "PUNCT_DIVERSITY",
        "ALPHABET_LOOP", "PRIME_LENGTHS", "NESTED_QUOTES", "NUMBERS_COUNT", "TITLE_CASE", "CHAIN_WORDS",
        "NO_CONSEC_LETTER", "LIMITED_REPEAT", "SENTENCE_STEP", "DATE_FORMAT", "CSV_FORMAT", "OUTPUT_TEMPLATE",
        "PARA_MATCH", "CONJUNCTION_COUNT", "INDENT_STAIRS", "SPECIAL_BULLET", "SYMBOL_END"
    ]},
    {"id": "sentence:symbol_end", "func": make_unseen_symbol_end, "group": "SYMBOL_END", "conflicts_with": ["NO_WHITESPACE"]}
]


def sample_compatible_constraints(k: int, pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lấy mẫu k ràng buộc không xung đột từ tập pool chỉ định."""
    available = list(pool)
    selected = []
    forbidden_groups = set()

    random.shuffle(available)

    for item in available:
        if len(selected) >= k:
            break
        if item["group"] in forbidden_groups:
            continue
        if any(g in forbidden_groups for g in item["conflicts_with"]):
            continue

        selected.append(item)
        forbidden_groups.add(item["group"])
        forbidden_groups.update(item["conflicts_with"])

    return selected


def augment_instruction_vi(base_text: str, k: int, split: str = "seen") -> Dict[str, Any]:
    """Ghép base instruction tiếng Việt với k hard constraints tiếng Việt."""
    pool = SEEN_CONSTRAINT_DEFINITIONS if split == "seen" else UNSEEN_CONSTRAINT_DEFINITIONS
    chosen_defs = sample_compatible_constraints(k, pool=pool)

    constraint_texts = []
    instruction_ids = []
    kwargs_list = []

    for c_def in chosen_defs:
        inst_id, desc_text, kwargs = c_def["func"]()
        constraint_texts.append(norm_vi(desc_text))
        instruction_ids.append(inst_id)
        kwargs_list.append(kwargs)

    # Xây dựng prompt tiếng Việt hoàn chỉnh
    if len(constraint_texts) == 1:
        augmented_prompt = f"{base_text}\n\nRàng buộc:\n- {constraint_texts[0]}"
    else:
        bullets = "\n".join([f"- {ct}" for ct in constraint_texts])
        augmented_prompt = (
            f"{base_text}\n\n"
            f"Vui lòng tuân thủ nghiêm ngặt các ràng buộc sau trong câu trả lời:\n"
            f"{bullets}"
        )

    ground_truth = [{
        "instruction_id": instruction_ids,
        "kwargs": kwargs_list
    }]

    return {
        "prompt": norm_vi(augmented_prompt),
        "num_constraints": len(chosen_defs),
        "constraint_type": "single" if len(chosen_defs) == 1 else "multi",
        "constraint_ids": instruction_ids,
        "constraints_description": constraint_texts,
        "ground_truth": ground_truth,
        "constraint_split": split
    }


def main():
    parser = argparse.ArgumentParser(description="Module 1: Sinh Hard Constraints tiếng Việt theo Batch 16 mẫu.")
    parser.add_argument("--input", type=str, default="datasets/raw/base_vi.jsonl",
                        help="Đường dẫn file dataset gốc tiếng Việt (base_vi.jsonl)")
    parser.add_argument("--output-dir", type=str, default="results/module_hard_constraints",
                        help="Thư mục lưu các file batch JSON kết quả")
    parser.add_argument("--split", type=str, default="seen", choices=["seen", "unseen"],
                        help="Tập ràng buộc: 'seen' (29 loại) hoặc 'unseen' (24 loại OOD)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Số lượng mẫu trong mỗi batch (mặc định: 16)")
    parser.add_argument("--id", type=str, default=None,
                        help="Chạy duy nhất 1 mẫu theo ID số nguyên hoặc key")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Chỉ mục bắt đầu duyệt trong danh sách mẫu")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Chỉ mục kết thúc duyệt trong danh sách mẫu")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Giới hạn số mẫu tối đa cần xử lý")
    parser.add_argument("--resume", action="store_true",
                        help="Tiếp tục từ batch chưa hoàn thành")
    parser.add_argument("--save-unified", action="store_true",
                        help="Đồng thời lưu file gộp vào datasets/seen hoặc datasets/unseen")
    parser.add_argument("--translate", action="store_true",
                        help="Tự động dịch sang tiếng Việt nếu câu lệnh chưa được dịch (hoặc còn ở dạng tiếng Anh)")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash-lite",
                        help="Mô hình LLM dịch: 'gemini-3.5-flash-lite' hoặc 'gemma4-31B-it' (mặc định: gemini-3.5-flash-lite)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="OpenAI API Base URL tùy chỉnh (dùng khi chạy vLLM cho Gemma)")
    parser.add_argument("--delay", type=float, default=3.5,
                        help="Thời gian nghỉ (giây) giữa các request API dịch (mặc định 3.5s phù hợp hạn mức 15 RPM của Gemini free)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Lỗi: File input '{args.input}' không tồn tại. Vui lòng chạy fetch_and_translate.py trước.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Đọc dữ liệu từ base_vi.jsonl
    raw_samples = []
    with open(args.input, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                raw_samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    print(f"📖 Đã đọc {len(raw_samples)} mẫu tiếng Việt từ '{args.input}'.")

    # Lọc theo --id
    if args.id:
        target_id = str(args.id).strip()
        raw_samples = [
            s for s in raw_samples 
            if str(s.get("id")) == target_id or s.get("key") == target_id
        ]
        print(f"🎯 Lọc theo --id '{args.id}': tìm thấy {len(raw_samples)} mẫu.")

    # Lọc theo dải start-idx, end-idx
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(raw_samples)
    if args.num_samples is not None:
        end_idx = min(end_idx, start_idx + args.num_samples)

    selected_samples = raw_samples[start_idx:end_idx]
    print(f"⚙️ Xử lý {len(selected_samples)} mẫu (từ {start_idx} đến {end_idx}) với Split '{args.split}'.")

    # Xác định các key đã có nếu --resume
    processed_keys = set()
    if args.resume:
        for fname in os.listdir(args.output_dir):
            if fname.endswith(".json") and fname.startswith("module1_hard_batch_"):
                fpath = os.path.join(args.output_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f_b:
                        batch_data = json.load(f_b)
                        if isinstance(batch_data, list):
                            for it in batch_data:
                                if "key" in it:
                                    processed_keys.add(it["key"])
                except Exception:
                    pass
        print(f"🔄 [RESUME] Đã phát hiện {len(processed_keys)} mẫu đã xử lý trước đó.")

    # Phân phối K=1..5
    k_options = [1, 2, 3, 4, 5]
    k_weights = [0.20, 0.35, 0.25, 0.15, 0.05]

    # Chia thành các batch 16 mẫu
    batch_size = args.batch_size
    current_batch = []
    batch_idx = 1
    total_saved = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # File gộp nếu có --save-unified
    unified_file = f"datasets/{args.split}/augmented_vi.jsonl" if args.split == "seen" else f"datasets/{args.split}/eval_unseen_vi.jsonl"
    if args.save_unified:
        os.makedirs(os.path.dirname(unified_file), exist_ok=True)

    # Khởi tạo client dịch thuật nếu bật cờ --translate
    trans_client = None
    resolved_model = args.model
    if args.translate:
        try:
            from llm_client import get_llm_client_and_model, translate_instruction_vi
            trans_client, resolved_model, resolved_url = get_llm_client_and_model(args.model, args.base_url)
            print("=" * 65)
            print(f"🌐 [Chế độ Dịch Tự Động] Kích hoạt dịch với model: '{resolved_model}'")
            print(f"   • API Endpoint: {resolved_url}")
            print("=" * 65)
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo LLM Client dịch: {e}. Tiếp tục dùng văn bản hiện có.")

    for item in selected_samples:
        key = item.get("key", f"sample_{item.get('id', 0)}")
        if args.resume and key in processed_keys:
            continue

        base_inst = item.get("instruction_vi", "")
        inp_vi = item.get("input_vi", "")

        # Tự động dịch sang tiếng Việt nếu câu lệnh chưa được dịch
        if args.translate and trans_client:
            inst_en = item.get("instruction_en", "")
            is_untranslated = (not base_inst) or (base_inst == inst_en and len(inst_en) > 0) or (item.get("status") == "raw_indexed")
            if is_untranslated:
                in_en = item.get("input_en", "")
                print(f"🔄 Dịch mẫu {key} bằng {resolved_model}...")
                t_inst, t_inp = translate_instruction_vi(trans_client, resolved_model, inst_en, in_en)
                base_inst = t_inst
                inp_vi = t_inp
                item["instruction_vi"] = base_inst
                item["input_vi"] = inp_vi
                if args.delay > 0:
                    time.sleep(args.delay)

        if inp_vi and len(inp_vi.strip()) > 0:
            full_base = f"{base_inst}\n\nNgữ cảnh bổ sung:\n{inp_vi}"
        else:
            full_base = base_inst

        if not full_base or len(full_base.strip()) < 5:
            continue

        k = random.choices(k_options, weights=k_weights, k=1)[0]
        aug = augment_instruction_vi(full_base, k=k, split=args.split)

        record = {
            "id": item.get("id"),
            "key": key,
            "dataset_source": item.get("dataset_source", ""),
            "constraint_split": args.split,
            "base_instruction": full_base,
            "reference_response": item.get("output_vi", ""),
            "prompt": aug["prompt"],
            "num_constraints": aug["num_constraints"],
            "constraint_type": aug["constraint_type"],
            "hard_constraints": {
                "verifier_type": "python_rule_engine",
                "constraint_ids": aug["constraint_ids"],
                "constraints_description": aug["constraints_description"],
                "ground_truth": aug["ground_truth"]
            },
            "messages": [{"role": "user", "content": aug["prompt"]}],
            "batch_id": batch_idx,
            "batch_sample_idx": len(current_batch) + 1,
            "created_at": datetime.datetime.now().isoformat()
        }

        current_batch.append(record)

        # Khi đủ batch_size = 16 mẫu thì xuất file JSON của batch
        if len(current_batch) >= batch_size:
            batch_filename = f"module1_hard_batch_{batch_idx:03d}_{timestamp}.json"
            batch_filepath = os.path.join(args.output_dir, batch_filename)
            with open(batch_filepath, "w", encoding="utf-8") as f_batch:
                json.dump(current_batch, f_batch, ensure_ascii=False, indent=2)

            if args.save_unified:
                with open(unified_file, "a", encoding="utf-8") as f_uni:
                    for rec in current_batch:
                        f_uni.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"📦 [Batch {batch_idx:03d}] Đã lưu {len(current_batch)} mẫu vào '{batch_filepath}'.")
            total_saved += len(current_batch)
            current_batch = []
            batch_idx += 1

    # Lưu nốt batch cuối nếu còn dư mẫu (< 16 mẫu)
    if current_batch:
        batch_filename = f"module1_hard_batch_{batch_idx:03d}_{timestamp}.json"
        batch_filepath = os.path.join(args.output_dir, batch_filename)
        with open(batch_filepath, "w", encoding="utf-8") as f_batch:
            json.dump(current_batch, f_batch, ensure_ascii=False, indent=2)

        if args.save_unified:
            with open(unified_file, "a", encoding="utf-8") as f_uni:
                for rec in current_batch:
                    f_uni.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"📦 [Batch {batch_idx:03d} (Cuối)] Đã lưu {len(current_batch)} mẫu vào '{batch_filepath}'.")
        total_saved += len(current_batch)

    print("\n" + "=" * 60)
    print(f"🎉 HOÀN THÀNH MODULE 1 (Hard Constraints Tiếng Việt):")
    print(f"   • Tổng số mẫu đã lưu : {total_saved} mẫu")
    print(f"   • Số batch hoàn thành: {batch_idx}")
    print(f"   • Thư mục kết quả    : {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
