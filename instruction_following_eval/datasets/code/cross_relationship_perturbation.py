"""
Cross-Relationship Perturbation Module (CRPL) for GRPO
Inspired by: "Cross-Relational Preference Learning for Better LLM Instruction Following"
             (Li et al., arXiv:2608.29352, Aug 2026)

Purpose:
    Given an instruction with K atomic hard constraints, generates perturbed
    variants that exhibit one of three response-space relationships:
        1. Containment   — perturbed space ⊂ or ⊃ original space
        2. Partial Overlap — spaces partially intersect
        3. Disjoint       — spaces are completely separate

Role in GRPO Pipeline:
    Acts as a lightweight, rule-based Environment & Prompt Augmenter.
    By creating fine-grained boundary perturbations, it exposes the policy
    to sensitive parameter transitions during online rollouts, preventing
    heuristic shortcuts (e.g. over-generation) and reward hacking.
    Note: Offline preference pairs (chosen/rejected) are omitted as GRPO 
    evaluates candidates dynamically within groups using verifiable rewards.

Usage:
    python cross_relationship_perturbation.py \
        --input  datasets/seen/augmented_alpaca_gpt4.jsonl \
        --output datasets/seen/crpl_seen_variants.jsonl \
        --mode all \
        --num-samples 21
"""

import json
import random
import re
import copy
import argparse
import os
import sys
from typing import List, Dict, Any, Tuple, Optional


# ==============================================================================
# 1. Perturbation Rules Registry
# ==============================================================================
# Each constraint type has rules for generating containment / overlap / disjoint
# perturbations by adjusting its parameters.

def _perturb_number_words(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb length_constraints:number_words"""
    rel = kwargs.get("relation", "at least")
    n = kwargs.get("num_words", 100)
    
    if mode == "containment":
        # Widen the constraint → original response space ⊂ perturbed space
        if rel == "at least":
            new_n = max(10, n - random.randint(20, 50))
            return f"Your response should contain at least {new_n} words.", {"num_words": new_n, "relation": "at least"}
        elif rel == "less than":
            new_n = n + random.randint(30, 80)
            return f"Your response should contain fewer than {new_n} words.", {"num_words": new_n, "relation": "less than"}
        else:  # range
            low = kwargs.get("min_words", 50)
            high = kwargs.get("max_words", 150)
            new_low = max(10, low - 30)
            new_high = high + 30
            return (
                f"Your response should contain between {new_low} and {new_high} words.",
                {"num_words": new_high, "relation": "range", "min_words": new_low, "max_words": new_high}
            )
    
    elif mode == "partial_overlap":
        # Shift range to partially overlap
        if rel == "at least":
            new_n = n + random.randint(20, 50)
            return (
                f"Your response should contain between {new_n} and {new_n + 100} words.",
                {"num_words": new_n + 100, "relation": "range", "min_words": new_n, "max_words": new_n + 100}
            )
        elif rel == "less than":
            shift = random.randint(30, 60)
            new_low = max(10, n - shift)
            return f"Your response should contain at least {new_low} words.", {"num_words": new_low, "relation": "at least"}
        else:
            low = kwargs.get("min_words", 50)
            high = kwargs.get("max_words", 150)
            shift = (high - low) // 2
            new_low = low + shift
            new_high = high + shift + 30
            return (
                f"Your response should contain between {new_low} and {new_high} words.",
                {"num_words": new_high, "relation": "range", "min_words": new_low, "max_words": new_high}
            )
    
    else:  # disjoint
        if rel == "at least":
            new_n = max(10, n // 3)
            return f"Your response should contain fewer than {new_n} words.", {"num_words": new_n, "relation": "less than"}
        elif rel == "less than":
            new_n = n + random.randint(50, 100)
            return f"Your response should contain at least {new_n} words.", {"num_words": new_n, "relation": "at least"}
        else:
            low = kwargs.get("min_words", 50)
            high = kwargs.get("max_words", 150)
            new_n = max(10, low - random.randint(20, 40))
            return f"Your response should contain fewer than {new_n} words.", {"num_words": new_n, "relation": "less than"}


def _perturb_number_sentences(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb length_constraints:number_sentences"""
    rel = kwargs.get("relation", "at least")
    n = kwargs.get("num_sentences", 3)
    
    if mode == "containment":
        if rel == "exactly":
            return f"Your entire response must contain at least {max(1, n - 1)} sentences.", {"num_sentences": max(1, n - 1), "relation": "at least"}
        elif rel == "at least":
            new_n = max(1, n - 1)
            return f"Your entire response must contain at least {new_n} sentences.", {"num_sentences": new_n, "relation": "at least"}
        else:
            new_n = n + 2
            return f"Your entire response must contain fewer than {new_n} sentences.", {"num_sentences": new_n, "relation": "less than"}
    
    elif mode == "partial_overlap":
        if rel == "exactly":
            new_n = n + random.choice([-1, 1])
            if new_n < 1:
                new_n = n + 1
            return f"Your entire response must contain exactly {new_n} sentences.", {"num_sentences": new_n, "relation": "exactly"}
        elif rel == "at least":
            return f"Your entire response must contain fewer than {n + 2} sentences.", {"num_sentences": n + 2, "relation": "less than"}
        else:
            return f"Your entire response must contain at least {max(1, n - 2)} sentences.", {"num_sentences": max(1, n - 2), "relation": "at least"}
    
    else:  # disjoint
        if rel == "exactly":
            new_n = n + random.randint(3, 5)
            return f"Your entire response must contain exactly {new_n} sentences.", {"num_sentences": new_n, "relation": "exactly"}
        elif rel == "at least":
            new_n = max(1, n - 2) if n > 3 else 1
            return f"Your entire response must contain fewer than {new_n} sentences.", {"num_sentences": new_n, "relation": "less than"}
        else:
            return f"Your entire response must contain at least {n + 3} sentences.", {"num_sentences": n + 3, "relation": "at least"}


def _perturb_keyword_existence(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb keywords:existence"""
    EXTRA_WORDS = [
        "framework", "methodology", "blueprint", "ecosystem", "architecture",
        "landscape", "spectrum", "narrative", "trajectory", "renaissance"
    ]
    kws = kwargs.get("keywords", ["technology"])
    
    if mode == "containment":
        if len(kws) > 1:
            subset = [kws[0]]
            text = f"Include the keyword '{subset[0]}' in your response."
            return text, {"keywords": subset}
        else:
            extra = random.choice(EXTRA_WORDS)
            text = f"Include the keyword '{kws[0]}' or '{extra}' in your response."
            return text, {"keywords": [kws[0]], "alternative": extra}
    
    elif mode == "partial_overlap":
        new_kw = random.choice(EXTRA_WORDS)
        text = f"Include the keywords '{kws[0]}' and '{new_kw}' in your response."
        return text, {"keywords": [kws[0], new_kw]}
    
    else:  # disjoint
        new_kws = random.sample(EXTRA_WORDS, k=min(2, len(kws)))
        if len(new_kws) == 1:
            text = f"Include the keyword '{new_kws[0]}' in your response."
        else:
            text = f"Include the keywords '{new_kws[0]}' and '{new_kws[1]}' in your response."
        return text, {"keywords": new_kws}


def _perturb_keyword_frequency(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb keywords:frequency"""
    word = kwargs.get("keyword", "technology")
    freq = kwargs.get("frequency", 3)
    rel = kwargs.get("relation", "at least")
    
    if mode == "containment":
        if rel == "at least":
            new_freq = max(1, freq - 1)
            return f"The word '{word}' must appear at least {new_freq} times in your response.", {"keyword": word, "frequency": new_freq, "relation": "at least"}
        elif rel == "at most":
            new_freq = freq + 2
            return f"The word '{word}' must appear at most {new_freq} times in your response.", {"keyword": word, "frequency": new_freq, "relation": "at most"}
        else:
            return f"The word '{word}' must appear at least {freq} times in your response.", {"keyword": word, "frequency": freq, "relation": "at least"}
    
    elif mode == "partial_overlap":
        if rel == "at least":
            new_freq = freq + 2
            return f"The word '{word}' must appear at most {new_freq} times in your response.", {"keyword": word, "frequency": new_freq, "relation": "at most"}
        elif rel == "at most":
            new_freq = max(1, freq - 1)
            return f"The word '{word}' must appear at least {new_freq} times in your response.", {"keyword": word, "frequency": new_freq, "relation": "at least"}
        else:
            new_freq = freq + random.choice([-1, 1])
            if new_freq < 1:
                new_freq = freq + 1
            return f"The word '{word}' must appear exactly {new_freq} times in your response.", {"keyword": word, "frequency": new_freq, "relation": "exactly"}
    
    else:  # disjoint
        if rel == "at least":
            new_freq = max(1, freq - 2) if freq > 3 else 1
            return f"The word '{word}' must appear at most {new_freq} times in your response.", {"keyword": word, "frequency": new_freq, "relation": "at most"}
        elif rel == "at most":
            new_freq = freq + random.randint(3, 5)
            return f"The word '{word}' must appear at least {new_freq} times in your response.", {"keyword": word, "frequency": new_freq, "relation": "at least"}
        else:
            new_freq = freq + random.randint(3, 5)
            return f"The word '{word}' must appear exactly {new_freq} times in your response.", {"keyword": word, "frequency": new_freq, "relation": "exactly"}


def _perturb_case(constraint_id: str, kwargs: Dict, mode: str) -> Tuple[str, str, Dict]:
    """Perturb change_case:english_capital / english_lowercase"""
    if constraint_id == "change_case:english_capital":
        if mode == "containment":
            return "change_case:capital_word_frequency", "Your response must contain at least 5 capitalized words / words written in ALL CAPS.", {"capital_words": 5}
        elif mode == "partial_overlap":
            return "change_case:capital_word_frequency", "Your response must contain at least 10 capitalized words / words written in ALL CAPS.", {"capital_words": 10}
        else:
            return "change_case:english_lowercase", "Your entire response must be written in all lowercase letters (no uppercase letters).", {}
    else:
        if mode == "containment":
            return "change_case:english_lowercase", "Your entire response must be written in all lowercase letters (no uppercase letters).", {}
        elif mode == "partial_overlap":
            return "change_case:capital_word_frequency", "Your response must contain at least 3 capitalized words / words written in ALL CAPS.", {"capital_words": 3}
        else:
            return "change_case:english_capital", "Your entire response must be written in ALL CAPITAL LETTERS (no lowercase letters).", {}


def _perturb_number_paragraphs(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb length_constraints:number_paragraphs"""
    n = kwargs.get("num_paragraphs", 3)
    
    if mode == "containment":
        return f"Your response must be structured into at least {max(1, n - 1)} paragraphs, separated by double newlines.", {"num_paragraphs": max(1, n - 1)}
    elif mode == "partial_overlap":
        new_n = n + random.choice([-1, 1])
        if new_n < 1:
            new_n = n + 1
        return f"Your response must be structured into exactly {new_n} paragraphs, separated by double newlines.", {"num_paragraphs": new_n}
    else:
        new_n = n + random.randint(3, 5)
        return f"Your response must be structured into exactly {new_n} paragraphs, separated by double newlines.", {"num_paragraphs": new_n}


def _perturb_bullet_lists(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb detectable_format:number_bullet_lists"""
    n = kwargs.get("num_bullets", 4)
    
    if mode == "containment":
        new_n = max(1, n - 2)
        return f"Your answer must contain at least {new_n} bullet points formatted with '*' or '-'.", {"num_bullets": new_n}
    elif mode == "partial_overlap":
        new_n = n + random.choice([-1, 1])
        if new_n < 1:
            new_n = n + 1
        return f"Your answer must contain exactly {new_n} bullet points formatted with '*' or '-'.", {"num_bullets": new_n}
    else:
        new_n = n + random.randint(5, 8)
        return f"Your answer must contain exactly {new_n} bullet points formatted with '*' or '-'.", {"num_bullets": new_n}


def _perturb_numbers_count(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb count:numbers_count"""
    n = kwargs.get("num_numbers", 4)
    
    if mode == "containment":
        new_n = max(1, n - 1)
        return f"Include at least {new_n} numbers (written as digits, e.g. 1, 2, 3) in your entire response.", {"num_numbers": new_n}
    elif mode == "partial_overlap":
        new_n = n + random.choice([-1, 1])
        if new_n < 1:
            new_n = n + 1
        return f"Include exactly {new_n} numbers (written as digits, e.g. 1, 2, 3) in your entire response.", {"num_numbers": new_n}
    else:
        new_n = n + random.randint(5, 10)
        return f"Include exactly {new_n} numbers (written as digits, e.g. 1, 2, 3) in your entire response.", {"num_numbers": new_n}


def _perturb_unique_words(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb count:unique_word_count"""
    n = kwargs.get("num_unique_words", 40)
    
    if mode == "containment":
        new_n = max(10, n - random.randint(10, 20))
        return f"Your response must contain at least {new_n} unique words.", {"num_unique_words": new_n}
    elif mode == "partial_overlap":
        new_n = n + random.randint(15, 30)
        return f"Your response must contain at least {new_n} unique words.", {"num_unique_words": new_n}
    else:
        new_n = max(5, n // 4)
        return f"Your response must contain fewer than {new_n} unique words.", {"num_unique_words": new_n}


def _perturb_limited_repeat(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb words:limited_repeat"""
    n = kwargs.get("max_repeats", 5)
    
    if mode == "containment":
        new_n = n + random.randint(2, 4)
        return f"Do not repeat any word more than {new_n} times across your entire response.", {"max_repeats": new_n}
    elif mode == "partial_overlap":
        new_n = n + random.choice([-1, 1])
        if new_n < 1:
            new_n = 2
        return f"Do not repeat any word more than {new_n} times across your entire response.", {"max_repeats": new_n}
    else:
        new_n = n + random.randint(5, 10)
        return f"Every content word must appear at least {new_n} times in your response.", {"min_repeats": new_n}


def _perturb_date_format_list(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb format:date_format_list"""
    n = kwargs.get("min_dates", 2)
    
    if mode == "containment":
        new_n = max(1, n - 1)
        return f"Include at least {new_n} dates formatted strictly as YYYY-MM-DD separated by commas in your response.", {"min_dates": new_n}
    elif mode == "partial_overlap":
        new_n = n + 1
        return f"Include at least {new_n} dates formatted strictly as YYYY-MM-DD separated by commas in your response.", {"min_dates": new_n}
    else:
        return "Include at least 2 dates formatted strictly as DD/MM/YYYY in your response.", {"min_dates": 2, "date_format": "DD/MM/YYYY"}


def _perturb_nested_parentheses(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb format:parentheses"""
    n = kwargs.get("num_nested", 1)
    
    if mode == "containment":
        return "Include parentheses in your response.", {"num_nested": 0}
    elif mode == "partial_overlap":
        new_n = n + 1
        return f"Include at least {new_n} instances of nested parentheses ((like this)) in your response.", {"num_nested": new_n}
    else:
        return "Do not use any parentheses in your response.", {"num_nested": 0, "forbidden": True}


def _perturb_capital_word_frequency(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb change_case:capital_word_frequency"""
    n = kwargs.get("capital_words", 5)
    if mode == "containment":
        new_n = max(1, n - 2)
        return f"Your response must contain at least {new_n} capitalized words / words written in ALL CAPS.", {"capital_words": new_n}
    elif mode == "partial_overlap":
        new_n = n + 5
        return f"Your response must contain at least {new_n} capitalized words / words written in ALL CAPS.", {"capital_words": new_n}
    else:
        return "Your entire response must be written in all lowercase letters without any capitalized words.", {"capital_words": 0}


def _perturb_number_placeholders(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb detectable_content:number_placeholders"""
    n = kwargs.get("num_placeholders", 3)
    if mode == "containment":
        new_n = max(1, n - 1)
        return f"Include at least {new_n} placeholders in brackets (e.g. [name], [address], [date]) in your response.", {"num_placeholders": new_n}
    elif mode == "partial_overlap":
        new_n = n + 2
        return f"Include at least {new_n} placeholders in brackets (e.g. [name], [address], [date]) in your response.", {"num_placeholders": new_n}
    else:
        return "Do not include any placeholders, bracketed variables, or square brackets in your response.", {"num_placeholders": 0}


def _perturb_forbidden_words(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb keywords:forbidden_words"""
    words = kwargs.get("forbidden_words", ["however", "furthermore"])
    if mode == "containment":
        subset = words[:max(1, len(words) - 1)]
        return f"Do not include any of the following words in your response: {', '.join(subset)}.", {"forbidden_words": subset}
    elif mode == "partial_overlap":
        new_words = list(words)
        new_words[-1] = "consequently" if "consequently" not in new_words else "therefore"
        return f"Do not include any of the following words in your response: {', '.join(new_words)}.", {"forbidden_words": new_words}
    else:
        return f"Include the following words in your response: {', '.join(words)}.", {"keywords": words}


def _perturb_letter_frequency(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb keywords:letter_frequency"""
    char = kwargs.get("letter", "e")
    freq = kwargs.get("let_frequency", 5)
    rel = kwargs.get("let_relation", "at least")
    if mode == "containment":
        new_freq = max(1, freq - 2) if rel == "at least" else freq + 3
        return f"The letter '{char}' should appear {rel} {new_freq} times in your entire response.", {"letter": char, "let_frequency": new_freq, "let_relation": rel}
    elif mode == "partial_overlap":
        new_freq = freq + 3 if rel == "at least" else max(1, freq - 2)
        return f"The letter '{char}' should appear {rel} {new_freq} times in your entire response.", {"letter": char, "let_frequency": new_freq, "let_relation": rel}
    else:
        opp_rel = "at most" if rel == "at least" else "at least"
        opp_freq = max(1, freq // 2) if rel == "at least" else freq * 2
        return f"The letter '{char}' should appear {opp_rel} {opp_freq} times in your entire response.", {"letter": char, "let_frequency": opp_freq, "let_relation": opp_rel}


def _perturb_response_language(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb language:response_language"""
    lang = kwargs.get("language", "Vietnamese")
    LANG_MAP = {"Vietnamese": "French", "French": "Spanish", "Spanish": "German", "English": "Vietnamese"}
    alt_lang = LANG_MAP.get(lang, "French")
    if mode == "containment":
        return f"Your entire response should preferably be written in {lang}, or bilingually in {lang} and English.", {"language": lang, "mode": "any_of"}
    elif mode == "partial_overlap":
        return f"Write the first half of your response in {lang} and the remainder in English.", {"language": lang, "mode": "first_half_second_half"}
    else:
        return f"Your entire response must be written in {alt_lang}.", {"language": alt_lang, "mode": "all"}


def _perturb_multiple_sections(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb detectable_format:multiple_sections"""
    n = kwargs.get("num_sections", 3)
    if mode == "containment":
        new_n = max(2, n - 1)
        return f"Organize your response into at least {new_n} sections, each starting with 'Section 1', 'Section 2', etc.", {"num_sections": new_n}
    elif mode == "partial_overlap":
        new_n = n + 1
        return f"Organize your response into exactly {new_n} sections, each starting with 'Section 1', 'Section 2', etc.", {"num_sections": new_n}
    else:
        return "Write your response as a continuous narrative prose without any numbered section headings.", {"num_sections": 0}


def _perturb_highlighted_sections(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb detectable_format:number_highlighted_sections"""
    n = kwargs.get("num_highlights", 3)
    if mode == "containment":
        new_n = max(1, n - 1)
        return f"Highlight at least {new_n} key terms or phrases using markdown bold (**word**).", {"num_highlights": new_n}
    elif mode == "partial_overlap":
        new_n = n + 2
        return f"Highlight exactly {new_n} key terms or phrases using markdown bold (**word**).", {"num_highlights": new_n}
    else:
        return "Do not use any bold formatting (**text**) anywhere in your response.", {"num_highlights": 0}


def _perturb_nth_paragraph_first_word(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb length_constraints:nth_paragraph_first_word"""
    nth = kwargs.get("nth_paragraph", 1)
    word = kwargs.get("first_word", "First")
    if mode == "containment":
        return f"Paragraph {nth} should start with '{word}' or a similar introductory word.", {"nth_paragraph": nth, "first_word": word}
    elif mode == "partial_overlap":
        return f"Paragraph {nth + 1} must start with the word '{word}'.", {"nth_paragraph": nth + 1, "first_word": word}
    else:
        return f"Paragraph {nth} must NOT start with the word '{word}'.", {"nth_paragraph": nth, "first_word": f"NOT_{word}"}


def _perturb_end_checker(kwargs: Dict, mode: str) -> Tuple[str, Dict]:
    """Perturb startend:end_checker"""
    phrase = kwargs.get("end_phrase", "Thank you.")
    if mode == "containment":
        return f"End your response with a concluding remark such as '{phrase}'.", {"end_phrase": phrase}
    elif mode == "partial_overlap":
        alt_phrase = f"{phrase.rstrip('.')}!"
        return f"End your entire response with the exact phrase: '{alt_phrase}'.", {"end_phrase": alt_phrase}
    else:
        return f"Do not end your response with '{phrase}'. Use a different concluding statement.", {"end_phrase": f"NOT_{phrase}"}


# Binary / toggle constraints
BINARY_PERTURBATIONS = {
    "detectable_format:table_format": {
        "containment": ("Organize your response or include a section formatted as a Markdown table or structured bullet list.", {}),
        "partial_overlap": ("Organize your entire response strictly as a Markdown table with at least 3 columns.", {}),
        "disjoint": ("Do not use any tables or grid formatting in your response. Write purely in paragraphs.", {}),
    },
    "combination:two_responses": {
        "containment": ("Provide one or two distinct perspectives on the prompt.", {}),
        "partial_overlap": ("Provide three distinct responses to the prompt. Separate each with six asterisks: ******.", {"separator": "******"}),
        "disjoint": ("Provide exactly one unified, cohesive response without any dividing asterisks or separators.", {}),
    },
    "detectable_format:constrained_response": {
        "containment": ("Provide a brief response choosing among the options or a related alternative.", {}),
        "partial_overlap": ("Your response must strictly be one of the following choices, followed by a one-sentence rationale.", {}),
        "disjoint": ("Provide an open-ended explanatory response rather than selecting from pre-defined choices.", {}),
    },
    "punctuation:no_comma": {
        "containment": ("Do not use commas (',') in the first paragraph of your response.", {}),
        "partial_overlap": ("You may use at most 2 commas (',') in your entire response.", {"max_commas": 2}),
        "disjoint": ("You must use at least 5 commas (',') in your response.", {"min_commas": 5}),
    },
    "punctuation:no_period": {
        "containment": ("Avoid using periods ('.') at the end of sentences where possible.", {}),
        "partial_overlap": ("You may use at most 2 periods ('.') in your entire response.", {"max_periods": 2}),
        "disjoint": ("Every sentence must end with a period ('.').", {}),
    },
    "detectable_format:json_format": {
        "containment": ("Your response should be structured data (JSON or YAML format).", {}),
        "partial_overlap": ("Include at least one valid JSON object within your response.", {}),
        "disjoint": ("Your response must be written in plain prose paragraphs without any structured data formats.", {}),
    },
    "detectable_format:title": {
        "containment": ("Your response should include a title or heading.", {}),
        "partial_overlap": ("Your response must include exactly 2 titles wrapped in double angular brackets, e.g. <<Title>>.", {}),
        "disjoint": ("Do not include any titles or headings in your response.", {}),
    },
    "detectable_content:postscript": {
        "containment": ("You may optionally add a closing remark at the end of your response.", {}),
        "partial_overlap": ("Add a postscript starting with 'N.B.' at the very end of your response.", {"postscript_marker": "N.B."}),
        "disjoint": ("Do not include any postscripts, appendices, or closing remarks after your main response.", {}),
    },
    "startend:quotation": {
        "containment": ("You may optionally wrap your response in quotation marks.", {}),
        "partial_overlap": ("Wrap your entire response in single quotation marks ('...').", {}),
        "disjoint": ("Do not use any quotation marks anywhere in your response.", {}),
    },
    "format:xml_wrapper": {
        "containment": ("You may optionally wrap your answer inside XML tags.", {}),
        "partial_overlap": ("Wrap your entire answer inside <answer> and </answer> XML tags.", {"tag": "answer"}),
        "disjoint": ("Do not use any XML or HTML tags in your response.", {}),
    },
    "combination:repeat_prompt": {
        "containment": ("You may optionally restate the question before answering.", {}),
        "partial_overlap": ("Paraphrase the request in your own words before providing your response.", {}),
        "disjoint": ("Do not restate, repeat, or reference the original question in your response.", {}),
    },
    "format:indent_stairs": {
        "containment": ("Use indentation to structure your response.", {}),
        "partial_overlap": ("Create a staircase effect by incrementally indenting each new line with 4 additional spaces.", {}),
        "disjoint": ("Do not use any indentation in your response. All lines must start at column 0.", {}),
    },
    "words:paragraph_last_first_match": {
        "containment": ("Try to create thematic coherence within each paragraph.", {}),
        "partial_overlap": ("The first sentence and last sentence of each paragraph must share a common keyword.", {}),
        "disjoint": ("Each paragraph must start and end with completely different topics.", {}),
    },
    "sentence:last_word_first_next": {
        "containment": ("Maintain a logical flow between consecutive sentences.", {}),
        "partial_overlap": ("The last noun of each sentence must appear somewhere in the very next sentence.", {}),
        "disjoint": ("No word from the end of a sentence may appear in the beginning of the next sentence.", {}),
    },
    "words:no_consecutive_first_letter": {
        "containment": ("Vary the starting letters of consecutive words where possible.", {}),
        "partial_overlap": ("No three consecutive words in your response can share the same first letter.", {}),
        "disjoint": ("Every pair of consecutive words must start with the same letter (alliteration throughout).", {}),
    },
    "format:title_case": {
        "containment": ("Capitalize the first letter of each sentence.", {}),
        "partial_overlap": ("Write every noun and verb in Title Case.", {}),
        "disjoint": ("Write your entire response in all lowercase.", {}),
    },
    "sentence:increment": {
        "containment": ("Write sentences of varying lengths.", {}),
        "partial_overlap": ("Each sentence must be at least as long as the previous one.", {}),
        "disjoint": ("Each subsequent sentence must contain fewer words than the sentence before it.", {}),
    },
    "words:alphabet_loop": {
        "containment": ("Use varied starting letters across your response.", {}),
        "partial_overlap": ("Each sentence must start with the next letter of the alphabet.", {}),
        "disjoint": ("Every word must start with the same letter.", {}),
    },
}


# ==============================================================================
# 2. Main Perturbation Dispatcher
# ==============================================================================

def perturb_constraint(
    constraint_id: str,
    description: str,
    kwargs: Dict[str, Any],
    mode: str  # "containment", "partial_overlap", "disjoint"
) -> Optional[Dict[str, Any]]:
    """
    Generate a perturbation for a single atomic constraint.
    
    Returns:
        dict with keys: constraint_id, description, kwargs, relationship
        or None if the constraint type is not supported for perturbation.
    """
    assert mode in ("containment", "partial_overlap", "disjoint"), f"Invalid mode: {mode}"
    
    # Parametric constraints with custom perturbation logic
    PARAMETRIC_HANDLERS = {
        "length_constraints:number_words": _perturb_number_words,
        "length_constraints:number_sentences": _perturb_number_sentences,
        "keywords:existence": _perturb_keyword_existence,
        "keywords:frequency": _perturb_keyword_frequency,
        "length_constraints:number_paragraphs": _perturb_number_paragraphs,
        "detectable_format:number_bullet_lists": _perturb_bullet_lists,
        "count:numbers_count": _perturb_numbers_count,
        "count:unique_word_count": _perturb_unique_words,
        "words:limited_repeat": _perturb_limited_repeat,
        "format:date_format_list": _perturb_date_format_list,
        "format:parentheses": _perturb_nested_parentheses,
        "change_case:capital_word_frequency": _perturb_capital_word_frequency,
        "detectable_content:number_placeholders": _perturb_number_placeholders,
        "keywords:forbidden_words": _perturb_forbidden_words,
        "keywords:letter_frequency": _perturb_letter_frequency,
        "language:response_language": _perturb_response_language,
        "detectable_format:multiple_sections": _perturb_multiple_sections,
        "detectable_format:number_highlighted_sections": _perturb_highlighted_sections,
        "length_constraints:nth_paragraph_first_word": _perturb_nth_paragraph_first_word,
        "startend:end_checker": _perturb_end_checker,
    }
    
    # Case constraints need special handling (id may change)
    if constraint_id in ("change_case:english_capital", "change_case:english_lowercase"):
        new_id, new_desc, new_kwargs = _perturb_case(constraint_id, kwargs, mode)
        return {
            "constraint_id": new_id,
            "description": new_desc,
            "kwargs": new_kwargs,
            "relationship": mode,
            "original_constraint_id": constraint_id,
        }
    
    # Parametric handlers
    if constraint_id in PARAMETRIC_HANDLERS:
        handler = PARAMETRIC_HANDLERS[constraint_id]
        new_desc, new_kwargs = handler(kwargs, mode)
        return {
            "constraint_id": constraint_id,
            "description": new_desc,
            "kwargs": new_kwargs,
            "relationship": mode,
            "original_constraint_id": constraint_id,
        }
    
    # Binary/toggle constraints
    if constraint_id in BINARY_PERTURBATIONS:
        pert = BINARY_PERTURBATIONS[constraint_id]
        new_desc, new_kwargs = pert[mode]
        return {
            "constraint_id": constraint_id,
            "description": new_desc,
            "kwargs": new_kwargs,
            "relationship": mode,
            "original_constraint_id": constraint_id,
        }
    
    # Unsupported constraint — return None
    return None


# ==============================================================================
# 3. Cross-Relationship Perturbation for Full Instructions
# ==============================================================================

def generate_crpl_variants(
    record: Dict[str, Any],
    modes: List[str] = None,
    drop_reference_response: bool = True,
) -> List[Dict[str, Any]]:
    """
    Given a record with atomic hard constraints, generate perturbed variants
    for each (constraint, relationship) pair.
    
    Args:
        record: A single record with constraint_ids, constraints_description,
                and ground_truth fields.
        modes: List of relationship modes to generate. 
               Default: ["containment", "partial_overlap", "disjoint"]
        drop_reference_response: If True, removes reference_response as it is
               not needed for GRPO training and prevents data bloat.
    
    Returns:
        List of perturbed records, each with one constraint modified.
    """
    if modes is None:
        modes = ["containment", "partial_overlap", "disjoint"]
    
    is_hybrid = "hard_constraints" in record or "soft_constraints" in record
    
    constraint_ids = []
    descriptions = []
    gt_ids = []
    gt_kwargs = []
    
    if is_hybrid and "hard_constraints" in record:
        hc = record["hard_constraints"]
        constraint_ids = hc.get("constraint_ids", [])
        descriptions = hc.get("constraints_description", [])
        if hc.get("ground_truth") and isinstance(hc["ground_truth"][0], dict):
            gt = hc["ground_truth"][0]
            gt_ids = gt.get("instruction_id", constraint_ids)
            gt_kwargs = gt.get("kwargs", [{}] * len(constraint_ids))
        else:
            gt_ids = constraint_ids
            gt_kwargs = [{}] * len(constraint_ids)
    else:
        constraint_ids = record.get("constraint_ids", [])
        descriptions = record.get("constraints_description", [])
        ground_truth = record.get("ground_truth", [{}])
        if ground_truth and isinstance(ground_truth[0], dict):
            gt_ids = ground_truth[0].get("instruction_id", constraint_ids)
            gt_kwargs = ground_truth[0].get("kwargs", [{}] * len(constraint_ids))
        else:
            gt_ids = constraint_ids
            gt_kwargs = [{}] * len(constraint_ids)
    
    variants = []
    
    for mode in modes:
        for i, (cid, desc) in enumerate(zip(constraint_ids, descriptions)):
            kw = gt_kwargs[i] if i < len(gt_kwargs) else {}
            
            pert = perturb_constraint(cid, desc, kw, mode)
            if pert is None:
                continue
            
            # Build perturbed record
            new_record = copy.deepcopy(record)
            
            # Drop reference_response for GRPO training environment
            if drop_reference_response:
                new_record.pop("reference_response", None)
            
            # Update the i-th constraint
            new_constraint_ids = list(constraint_ids)
            new_descriptions = list(descriptions)
            new_gt_kwargs = list(gt_kwargs)
            new_gt_ids = list(gt_ids) if gt_ids else list(constraint_ids)
            
            new_constraint_ids[i] = pert["constraint_id"]
            new_descriptions[i] = pert["description"]
            new_gt_kwargs[i] = pert["kwargs"]
            new_gt_ids[i] = pert["constraint_id"]
            
            # Rebuild prompt
            base = record.get("base_instruction", "")
            
            if is_hybrid:
                # Rebuild hybrid prompt preserving both Hard and Soft sections
                hard_bullet_list = "\n".join([f"- {d}" for d in new_descriptions])
                
                # Extract soft constraint descriptions
                soft_descriptions = []
                if "soft_constraints" in record and "constraints" in record["soft_constraints"]:
                    for sc in record["soft_constraints"]["constraints"]:
                        s_desc = sc.get("description", "")
                        if s_desc:
                            soft_descriptions.append(f"- {s_desc}")
                
                if soft_descriptions:
                    soft_bullet_list = "\n".join(soft_descriptions)
                    new_prompt = (
                        f"{base}\n\n"
                        f"Please adhere strictly to the following constraints in your response:\n"
                        f"[Formatting & Structure Constraints]\n"
                        f"{hard_bullet_list}\n\n"
                        f"[Content & Semantic Guidelines]\n"
                        f"{soft_bullet_list}"
                    )
                else:
                    new_prompt = (
                        f"{base}\n\n"
                        f"Please adhere strictly to the following constraints in your response:\n"
                        f"[Formatting & Structure Constraints]\n"
                        f"{hard_bullet_list}"
                    )
            else:
                if len(new_descriptions) == 1:
                    new_prompt = f"{base}\n\nConstraint: {new_descriptions[0]}"
                else:
                    bullet_list = "\n".join([f"- {d}" for d in new_descriptions])
                    new_prompt = (
                        f"{base}\n\n"
                        f"Please adhere strictly to the following constraints in your response:\n"
                        f"{bullet_list}"
                    )
            
            new_record["key"] = f"{record.get('key', 'unknown')}_crpl_{mode}_{i}"
            new_record["prompt"] = new_prompt
            new_record["messages"] = [{"role": "user", "content": new_prompt}]
            new_record["crpl_metadata"] = {
                "relationship": mode,
                "perturbed_constraint_index": i,
                "original_constraint_id": cid,
                "perturbed_constraint_id": pert["constraint_id"],
                "original_description": desc,
                "perturbed_description": pert["description"],
            }
            
            if is_hybrid:
                if "hard_constraints" in new_record:
                    new_record["hard_constraints"]["constraint_ids"] = new_constraint_ids
                    new_record["hard_constraints"]["constraints_description"] = new_descriptions
                    new_record["hard_constraints"]["ground_truth"] = [{"instruction_id": new_gt_ids, "kwargs": new_gt_kwargs}]
                new_record.pop("constraint_ids", None)
                new_record.pop("constraints_description", None)
                new_record.pop("ground_truth", None)
            else:
                new_record["constraint_ids"] = new_constraint_ids
                new_record["constraints_description"] = new_descriptions
                new_record["ground_truth"] = [{"instruction_id": new_gt_ids, "kwargs": new_gt_kwargs}]
            
            variants.append(new_record)
    
    return variants


def load_records(input_path: str) -> List[Dict[str, Any]]:
    """Loads records from either a .json or .jsonl file."""
    records = []
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = [data]
    else:  # Assume .jsonl
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_existing_variants_and_base_keys(output_jsonl: str) -> Tuple[List[Dict[str, Any]], set]:
    """Load existing variants and the set of base record keys already processed."""
    existing_variants = []
    processed_base_keys = set()
    if os.path.isfile(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    existing_variants.append(obj)
                    k = obj.get("key", "")
                    if "_crpl_" in k:
                        base_keys = k.split("_crpl_")[0]
                        processed_base_keys.add(base_keys)
                    elif k:
                        processed_base_keys.add(k)
                except Exception:
                    pass
    return existing_variants, processed_base_keys


# ==============================================================================
# 4. CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CRPL: Cross-Relationship Perturbation for GRPO verifiable training data"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input JSON or JSONL file with augmented instructions"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file with CRPL-perturbed variants"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["containment", "partial_overlap", "disjoint", "all"],
        help="Perturbation relationship mode (default: all)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=0,
        help="Max number of input samples to process (0 = all)"
    )
    parser.add_argument(
        "--max-variants-per-sample", type=int, default=0,
        help="Max perturbed variants per input sample (0 = all)"
    )
    parser.add_argument(
        "--keep-reference-response", action="store_true", default=False,
        help="Keep reference_response if present (default: False for GRPO)"
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume generation: skip already perturbed records in output file (default: True)"
    )
    parser.add_argument(
        "--no-resume", action="store_false", dest="resume",
        help="Do not resume; overwrite and process from start"
    )
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    
    input_path = args.input if os.path.isabs(args.input) else os.path.join(repo_root, args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(repo_root, args.output)
    
    if not os.path.isfile(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    modes = ["containment", "partial_overlap", "disjoint"] if args.mode == "all" else [args.mode]
    
    # Load input
    records = load_records(input_path)
    
    existing_variants = []
    processed_base_keys = set()
    if args.resume and os.path.isfile(output_path):
        existing_variants, processed_base_keys = load_existing_variants_and_base_keys(output_path)
        before_cnt = len(records)
        records = [r for r in records if r.get("key") not in processed_base_keys]
        print(f"Resume mode: {len(processed_base_keys)} base records already completed ({len(existing_variants)} variants).")
        print(f"Remaining base records to process: {len(records)} (out of {before_cnt}).")
    
    if args.num_samples > 0:
        records = records[:args.num_samples]
    
    print(f"Loaded {len(records)} input records from: {input_path}")
    print(f"Perturbation modes: {modes}")
    print(f"Drop reference_response: {not args.keep_reference_response}")
    
    # Generate variants
    all_variants = []
    stats = {"containment": 0, "partial_overlap": 0, "disjoint": 0, "skipped": 0}
    
    for rec in records:
        variants = generate_crpl_variants(
            rec,
            modes=modes,
            drop_reference_response=not args.keep_reference_response
        )
        
        if args.max_variants_per_sample > 0 and len(variants) > args.max_variants_per_sample:
            random.shuffle(variants)
            variants = variants[:args.max_variants_per_sample]
        
        for v in variants:
            rel = v.get("crpl_metadata", {}).get("relationship", "unknown")
            if rel in stats:
                stats[rel] += 1
        
        all_variants.extend(variants)
    
    # Write output JSONL (append if resume, else overwrite)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    file_mode = "a" if (args.resume and existing_variants) else "w"
    with open(output_path, file_mode, encoding="utf-8") as f:
        for v in all_variants:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    
    # Sync full JSON
    json_path = output_path.replace(".jsonl", ".json") if output_path.endswith(".jsonl") else output_path + ".json"
    full_variants = existing_variants + all_variants if (args.resume and existing_variants) else all_variants
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_variants, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"CRPL Perturbation Complete!")
    print(f"  Processed new records:   {len(records)}")
    print(f"  Newly added variants:    {len(all_variants)}")
    print(f"  Total variants in file:  {len(full_variants)}")
    print(f"  Breakdown of new variants:")
    for mode, count in stats.items():
        if mode != "skipped":
            print(f"    {mode}: {count}")
    print(f"  Skipped (unsupported):   {stats['skipped']}")
    print(f"  Output JSONL: {output_path}")
    print(f"  Output JSON:  {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
