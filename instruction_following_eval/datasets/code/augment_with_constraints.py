"""
Data Synthesis Engine: Augment Base Instructions with Verifiable Constraints
Inspired by "Generalizing Verifiable Instruction Following" (AI2 - IFBench / IF-RLVR)

Features:
- Samples 1 to 5 non-conflicting verifiable constraints per prompt.
- Compatible with IFEval and IFBench verifiers.
- Memory-efficient streaming (uses < 200MB RAM, safe for 8GB RAM laptops).
- Supports 'vicgalle/alpaca-gpt4' and 'Post-training-Data-Flywheel/gpt4-self-instruct'.
- Outputs JSONL ready for RLVR (GRPO) training or evaluation.
"""

import json
import random
import re
import argparse
from typing import List, Dict, Any, Tuple, Optional
from datasets import load_dataset

# ==============================================================================
# 1. Constraint Templates & Generator Pool
# ==============================================================================

# Common word bank for random keyword insertion
WORD_BANK = [
    "technology", "perspective", "harmony", "innovation", "strategy",
    "journey", "challenge", "balance", "insight", "discovery",
    "momentum", "horizon", "dimension", "foundation", "catalyst",
    "clarity", "reflection", "evolution", "synergy", "paradigm"
]

FORBIDDEN_CANDIDATES = [
    ["however", "therefore", "moreover"],
    ["good", "bad", "great"],
    ["always", "never", "sometimes"],
    ["first", "second", "finally"],
    ["important", "essential", "crucial"]
]

LETTERS = ["e", "a", "t", "s", "o", "i", "n", "r"]
END_PHRASES = [
    "This concludes the explanation.",
    "Hope this information is helpful.",
    "End of response.",
    "Please let me know if you need more details.",
    "That is the complete overview."
]

def make_keyword_existence() -> Tuple[str, str, Dict[str, Any]]:
    chosen = random.sample(WORD_BANK, k=random.choice([1, 2]))
    if len(chosen) == 1:
        text = f"Include the keyword '{chosen[0]}' in your response."
    else:
        text = f"Include the keywords '{chosen[0]}' and '{chosen[1]}' in your response."
    return "keywords:existence", text, {"keywords": chosen}

def make_keyword_frequency() -> Tuple[str, str, Dict[str, Any]]:
    word = random.choice(WORD_BANK)
    freq = random.choice([2, 3, 4])
    rel = random.choice(["at least", "at most", "exactly"])
    text = f"The word '{word}' must appear {rel} {freq} times in your response."
    return "keywords:frequency", text, {"keyword": word, "frequency": freq, "relation": rel}

def make_forbidden_words() -> Tuple[str, str, Dict[str, Any]]:
    chosen = random.choice(FORBIDDEN_CANDIDATES)
    text = f"Do not include any of the following words in your response: {', '.join(chosen)}."
    return "keywords:forbidden_words", text, {"forbidden_words": chosen}

def make_letter_frequency() -> Tuple[str, str, Dict[str, Any]]:
    letter = random.choice(LETTERS)
    rel = random.choice(["at least", "at most"])
    if rel == "at least":
        freq = random.choice([1, 2, 3, 4])
    else:
        freq = random.choice([0, 1, 2])
    text = f"The letter '{letter}' should appear {rel} {freq} times in your entire response."
    return "keywords:letter_frequency", text, {"letter": letter, "let_frequency": freq, "let_relation": rel}

def make_number_words() -> Tuple[str, str, Dict[str, Any]]:
    mode = random.choice(["min", "max", "range"])
    if mode == "min":
        n = random.randint(40, 120)
        return "length_constraints:number_words", f"Your response should contain at least {n} words.", {"num_words": n, "relation": "at least"}
    elif mode == "max":
        n = random.randint(100, 250)
        return "length_constraints:number_words", f"Your response should contain fewer than {n} words.", {"num_words": n, "relation": "less than"}
    else:
        low = random.randint(50, 100)
        high = low + random.randint(40, 80)
        return "length_constraints:number_words", f"Your response should contain between {low} and {high} words.", {"num_words": high, "relation": "range", "min_words": low, "max_words": high}

def make_number_sentences() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 6)
    rel = random.choice(["at least", "less than", "exactly"])
    if rel == "exactly":
        text = f"Your entire response must contain exactly {n} sentences."
    elif rel == "at least":
        text = f"Your entire response must contain at least {n} sentences."
    else:
        text = f"Your entire response must contain fewer than {n} sentences."
    return "length_constraints:number_sentences", text, {"num_sentences": n, "relation": rel}

def make_number_paragraphs() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 4)
    text = f"Your response must be structured into exactly {n} paragraphs, separated by double newlines."
    return "length_constraints:number_paragraphs", text, {"num_paragraphs": n}

def make_bullet_lists() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(3, 6)
    text = f"Your answer must contain exactly {n} bullet points formatted with '*' or '-'."
    return "detectable_format:number_bullet_lists", text, {"num_bullets": n}

def make_json_format() -> Tuple[str, str, Dict[str, Any]]:
    text = "The entire response must be a valid JSON object without any Markdown formatting or extra text outside the JSON."
    return "detectable_format:json_format", text, {}

def make_title() -> Tuple[str, str, Dict[str, Any]]:
    text = "Your response must include a title wrapped in double angular brackets, for example <<Title>>."
    return "detectable_format:title", text, {}

def make_multiple_sections() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 4)
    text = f"Divide your response into {n} distinct sections, with each section clearly marked as 'Section 1:', 'Section 2:', etc."
    return "detectable_format:multiple_sections", text, {"num_sections": n, "section_spliter": "Section"}

def make_highlighted_sections() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 4)
    text = f"Highlight at least {n} key sections or phrases using Markdown asterisks (e.g., *highlighted text*)."
    return "detectable_format:number_highlighted_sections", text, {"num_highlights": n}

def make_postscript() -> Tuple[str, str, Dict[str, Any]]:
    text = "Add a postscript starting with 'P.S.' at the very end of your response."
    return "detectable_content:postscript", text, {"postscript_marker": "P.S."}

def make_end_checker() -> Tuple[str, str, Dict[str, Any]]:
    phrase = random.choice(END_PHRASES)
    text = f"Your response must end with the exact sentence: '{phrase}'"
    return "startend:end_checker", text, {"end_phrase": phrase}

def make_quotation() -> Tuple[str, str, Dict[str, Any]]:
    text = "Wrap your entire response in double quotation marks (\"... \")."
    return "startend:quotation", text, {}

def make_capital() -> Tuple[str, str, Dict[str, Any]]:
    text = "Your entire response must be written in ALL CAPITAL LETTERS (no lowercase letters)."
    return "change_case:english_capital", text, {}

def make_lowercase() -> Tuple[str, str, Dict[str, Any]]:
    text = "Your entire response must be written in all lowercase letters (no uppercase letters)."
    return "change_case:english_lowercase", text, {}

def make_no_comma() -> Tuple[str, str, Dict[str, Any]]:
    text = "Do not use any commas (',') anywhere in your entire response."
    return "punctuation:no_comma", text, {}

# Additional IFTrain / OOD Generalization Constraints
def make_xml_wrapper() -> Tuple[str, str, Dict[str, Any]]:
    text = "Wrap your entire answer inside <response> and </response> XML tags."
    return "format:xml_wrapper", text, {"tag": "response"}

def make_numbered_list() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(3, 5)
    text = f"Format your response as a numbered list with exactly {n} numbered items (1., 2., ...)."
    return "format:numbered_list", text, {"num_items": n}

# 9 Additional IFTrain / IFEval Canonical Constraints
def make_response_language() -> Tuple[str, str, Dict[str, Any]]:
    langs = ["Spanish", "French", "German", "Vietnamese"]
    chosen = random.choice(langs)
    text = f"Your entire response must be written in {chosen}."
    return "language:response_language", text, {"language": chosen}

def make_nth_paragraph_first_word() -> Tuple[str, str, Dict[str, Any]]:
    nth = random.randint(1, 3)
    words = ["First", "Furthermore", "Finally", "Specifically", "Indeed", "However"]
    word = random.choice(words)
    text = f"Paragraph {nth} must start with the word '{word}'."
    return "length_constraints:nth_paragraph_first_word", text, {"nth_paragraph": nth, "first_word": word}

def make_number_placeholders() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 4)
    text = f"Include at least {n} placeholders in brackets (e.g. [name], [address], [date]) in your response."
    return "detectable_content:number_placeholders", text, {"num_placeholders": n}

def make_constrained_response() -> Tuple[str, str, Dict[str, Any]]:
    options = random.choice([
        ["Option A", "Option B", "Option C"],
        ["Agree", "Disagree", "Neutral"],
        ["Positive", "Negative", "Uncertain"]
    ])
    opt_str = ", ".join(f"'{o}'" for o in options)
    text = f"Your response must strictly be one of the following choices: {opt_str}."
    return "detectable_format:constrained_response", text, {"constrained_responses": options}

def make_two_responses() -> Tuple[str, str, Dict[str, Any]]:
    text = "Provide two distinct responses to the prompt. Separate them with six asterisks: ******."
    return "combination:two_responses", text, {"separator": "******"}

def make_repeat_prompt() -> Tuple[str, str, Dict[str, Any]]:
    text = "First repeat the request or prompt verbatim, and then provide your response."
    return "combination:repeat_prompt", text, {}

def make_capital_word_frequency() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 5)
    text = f"Your response must contain at least {n} capitalized words / words written in ALL CAPS."
    return "change_case:capital_word_frequency", text, {"capital_words": n}

def make_table_format() -> Tuple[str, str, Dict[str, Any]]:
    text = "Organize your response or include a section formatted as a Markdown table (using | Column 1 | Column 2 |)."
    return "detectable_format:table_format", text, {}

def make_no_period() -> Tuple[str, str, Dict[str, Any]]:
    text = "Do not use any period ('.') anywhere in your entire response."
    return "punctuation:no_period", text, {}


# ==============================================================================
# 2. Conflict Groups & Compatibility Engine
# ==============================================================================

# ==============================================================================
# 2. SEEN vs. UNSEEN Constraint Pools (Generalizing VIF Methodology)
# ==============================================================================

# --- SEEN POOL: IFEval (25) + IFTrain Hand-annotated Templates ---
SEEN_CONSTRAINT_DEFINITIONS = [
    {
        "id": "keywords:existence",
        "func": make_keyword_existence,
        "group": "KEYWORDS",
        "conflicts_with": ["JSON"]
    },
    {
        "id": "keywords:frequency",
        "func": make_keyword_frequency,
        "group": "KEYWORDS",
        "conflicts_with": ["JSON"]
    },
    {
        "id": "keywords:forbidden_words",
        "func": make_forbidden_words,
        "group": "KEYWORDS",
        "conflicts_with": []
    },
    {
        "id": "keywords:letter_frequency",
        "func": make_letter_frequency,
        "group": "LETTER",
        "conflicts_with": ["ALL_CAPS", "ALL_LOWER"]
    },
    {
        "id": "length_constraints:number_words",
        "func": make_number_words,
        "group": "LENGTH_WORD",
        "conflicts_with": ["LENGTH_SENTENCE"]
    },
    {
        "id": "length_constraints:number_sentences",
        "func": make_number_sentences,
        "group": "LENGTH_SENTENCE",
        "conflicts_with": ["LENGTH_WORD", "PARAGRAPHS"]
    },
    {
        "id": "length_constraints:number_paragraphs",
        "func": make_number_paragraphs,
        "group": "PARAGRAPHS",
        "conflicts_with": ["LENGTH_SENTENCE", "JSON"]
    },
    {
        "id": "detectable_format:number_bullet_lists",
        "func": make_bullet_lists,
        "group": "LIST_FORMAT",
        "conflicts_with": ["JSON", "NUMBERED_LIST", "ALL_CAPS"]
    },
    {
        "id": "format:numbered_list",
        "func": make_numbered_list,
        "group": "NUMBERED_LIST",
        "conflicts_with": ["JSON", "LIST_FORMAT"]
    },
    {
        "id": "detectable_format:json_format",
        "func": make_json_format,
        "group": "JSON",
        "conflicts_with": ["LIST_FORMAT", "NUMBERED_LIST", "PARAGRAPHS", "POSTSCRIPT", "SECTIONS", "XML_TAG", "ALL_CAPS"]
    },
    {
        "id": "detectable_format:title",
        "func": make_title,
        "group": "TITLE",
        "conflicts_with": ["JSON"]
    },
    {
        "id": "detectable_format:multiple_sections",
        "func": make_multiple_sections,
        "group": "SECTIONS",
        "conflicts_with": ["JSON", "PARAGRAPHS"]
    },
    {
        "id": "detectable_format:number_highlighted_sections",
        "func": make_highlighted_sections,
        "group": "HIGHLIGHT",
        "conflicts_with": ["JSON", "ALL_CAPS"]
    },
    {
        "id": "detectable_content:postscript",
        "func": make_postscript,
        "group": "POSTSCRIPT",
        "conflicts_with": ["JSON", "END_SENTENCE"]
    },
    {
        "id": "startend:end_checker",
        "func": make_end_checker,
        "group": "END_SENTENCE",
        "conflicts_with": ["JSON", "POSTSCRIPT", "ALL_CAPS", "ALL_LOWER"]
    },
    {
        "id": "startend:quotation",
        "func": make_quotation,
        "group": "QUOTATION",
        "conflicts_with": []
    },
    {
        "id": "change_case:english_capital",
        "func": make_capital,
        "group": "ALL_CAPS",
        "conflicts_with": ["ALL_LOWER", "LETTER", "HIGHLIGHT", "LIST_FORMAT", "END_SENTENCE", "JSON"]
    },
    {
        "id": "change_case:english_lowercase",
        "func": make_lowercase,
        "group": "ALL_LOWER",
        "conflicts_with": ["ALL_CAPS", "LETTER", "END_SENTENCE"]
    },
    {
        "id": "punctuation:no_comma",
        "func": make_no_comma,
        "group": "PUNCTUATION",
        "conflicts_with": []
    },
    {
        "id": "format:xml_wrapper",
        "func": make_xml_wrapper,
        "group": "XML_TAG",
        "conflicts_with": ["JSON"]
    },
    {
        "id": "language:response_language",
        "func": make_response_language,
        "group": "LANGUAGE",
        "conflicts_with": ["ALL_CAPS", "ALL_LOWER", "END_SENTENCE"]
    },
    {
        "id": "length_constraints:nth_paragraph_first_word",
        "func": make_nth_paragraph_first_word,
        "group": "PARAGRAPHS_WORD",
        "conflicts_with": ["JSON", "ONE_LINE"]
    },
    {
        "id": "detectable_content:number_placeholders",
        "func": make_number_placeholders,
        "group": "PLACEHOLDERS",
        "conflicts_with": ["JSON"]
    },
    {
        "id": "detectable_format:constrained_response",
        "func": make_constrained_response,
        "group": "CONSTRAINED_RESP",
        "conflicts_with": ["LENGTH_WORD", "PARAGRAPHS", "SECTIONS", "LIST_FORMAT", "NUMBERED_LIST", "TABLE", "TWO_RESPONSES"]
    },
    {
        "id": "combination:two_responses",
        "func": make_two_responses,
        "group": "TWO_RESPONSES",
        "conflicts_with": ["JSON", "CONSTRAINED_RESP"]
    },
    {
        "id": "combination:repeat_prompt",
        "func": make_repeat_prompt,
        "group": "REPEAT_PROMPT",
        "conflicts_with": ["JSON", "CONSTRAINED_RESP"]
    },
    {
        "id": "change_case:capital_word_frequency",
        "func": make_capital_word_frequency,
        "group": "CAPITAL_WORDS",
        "conflicts_with": ["ALL_CAPS", "ALL_LOWER"]
    },
    {
        "id": "detectable_format:table_format",
        "func": make_table_format,
        "group": "TABLE",
        "conflicts_with": ["JSON", "LIST_FORMAT", "NUMBERED_LIST", "CONSTRAINED_RESP"]
    },
    {
        "id": "punctuation:no_period",
        "func": make_no_period,
        "group": "PUNCTUATION_PERIOD",
        "conflicts_with": ["JSON"]
    }
]

# --- UNSEEN POOL: 24 Curated Out-Of-Domain Verifiable Constraints from IFBench ---
def make_unique_words() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(25, 60)
    text = f"Your response must contain at least {n} unique words."
    return "count:unique_word_count", text, {"num_unique_words": n}

def make_nested_parentheses() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(1, 3)
    text = f"Include at least {n} instances of nested parentheses ((like this)) in your response."
    return "format:parentheses", text, {"num_nested": n}

def make_palindrome() -> Tuple[str, str, Dict[str, Any]]:
    palindromes = ["level", "radar", "kayak", "rotor", "madam", "civic"]
    chosen = random.sample(palindromes, k=2)
    text = f"Include the palindromic words '{chosen[0]}' and '{chosen[1]}' in your response."
    return "words:palindrome", text, {"palindromes": chosen}

def make_sentence_increment() -> Tuple[str, str, Dict[str, Any]]:
    text = "Each subsequent sentence in your response must contain more words than the sentence before it."
    return "sentence:increment", text, {}

def make_start_verb() -> Tuple[str, str, Dict[str, Any]]:
    text = "Every paragraph in your response must start with a strong action verb."
    return "words:start_verb", text, {}

def make_punctuation_count() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(4, 7)
    text = f"Use at least {n} different types of punctuation marks across your entire response."
    return "count:punctuation", text, {"num_punctuation_types": n}

def make_alphabet_loop() -> Tuple[str, str, Dict[str, Any]]:
    text = "Each word in your response must start with the next letter of the alphabet in sequential order, looping back to 'a' after 'z'."
    return "words:alphabet_loop", text, {}

def make_prime_lengths() -> Tuple[str, str, Dict[str, Any]]:
    text = "Every word in your response must have a length (character count) that is a prime number (e.g., 2, 3, 5, 7, 11 letters long)."
    return "words:prime_lengths", text, {}

def make_nested_quotes() -> Tuple[str, str, Dict[str, Any]]:
    text = "Include quotes within quotes within quotes, at least 3 levels deep (alternating double and single quotation marks)."
    return "format:nested_quotes", text, {}

def make_numbers_count() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(3, 6)
    text = f"Include exactly {n} numbers (written as digits, e.g. 1, 2, 3) in your entire response."
    return "count:numbers_count", text, {"num_numbers": n}

def make_title_case() -> Tuple[str, str, Dict[str, Any]]:
    text = "Write your entire response in Title Case (capitalize the first letter of every major word)."
    return "format:title_case", text, {}

def make_last_word_first_next() -> Tuple[str, str, Dict[str, Any]]:
    text = "The last word of each sentence must become the first word of the very next sentence."
    return "sentence:last_word_first_next", text, {}

def make_no_consecutive_first_letter() -> Tuple[str, str, Dict[str, Any]]:
    text = "No two consecutive words in your response can share the same first letter."
    return "words:no_consecutive_first_letter", text, {}

def make_limited_repeat() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(3, 5)
    text = f"Do not repeat any word more than {n} times across your entire response."
    return "words:limited_repeat", text, {"max_repeats": n}

def make_word_count_step() -> Tuple[str, str, Dict[str, Any]]:
    step = random.randint(2, 4)
    text = f"Each subsequent sentence must contain exactly {step} more words than the sentence before it."
    return "sentence:word_count_step", text, {"step": step}

def make_date_format_list() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(2, 4)
    text = f"Include at least {n} dates formatted strictly as YYYY-MM-DD separated by commas in your response."
    return "format:date_format_list", text, {"min_dates": n}

def make_csv_format() -> Tuple[str, str, Dict[str, Any]]:
    rows = random.randint(3, 5)
    text = f"Format your entire response as CSV data with exactly {rows} data rows and columns: ID, Name, Category, Value."
    return "format:csv_format", text, {"num_rows": rows}

def make_output_template() -> Tuple[str, str, Dict[str, Any]]:
    text = "Use this exact template structure for your response:\nMy Answer: [answer]\nMy Conclusion: [conclusion]\nFuture Outlook: [outlook]"
    return "format:output_template", text, {}

def make_paragraph_last_first_match() -> Tuple[str, str, Dict[str, Any]]:
    text = "Each paragraph in your response must end with the exact same word it started with."
    return "words:paragraph_last_first_match", text, {}

def make_conjunction_count() -> Tuple[str, str, Dict[str, Any]]:
    n = random.randint(3, 5)
    text = f"Use at least {n} different coordinating conjunctions (from: and, but, for, nor, or, so, yet) in your response."
    return "words:conjunction_count", text, {"min_conjunctions": n}

def make_indent_stairs() -> Tuple[str, str, Dict[str, Any]]:
    text = "Create a staircase effect by incrementally indenting each new line with 2 additional spaces."
    return "format:indent_stairs", text, {}

def make_special_bullet() -> Tuple[str, str, Dict[str, Any]]:
    text = "Format your response as a list using '-> ' instead of standard bullet points for every item."
    return "format:special_bullet", text, {}

def make_no_whitespace() -> Tuple[str, str, Dict[str, Any]]:
    text = "Your response must not contain any whitespace characters (no spaces, no tabs, no newlines)."
    return "manipulation:no_whitespace", text, {}

def make_symbol_end() -> Tuple[str, str, Dict[str, Any]]:
    text = "Every sentence in your response must end with the symbol combination '!*'."
    return "sentence:symbol_end", text, {}

UNSEEN_CONSTRAINT_DEFINITIONS = [
    {
        "id": "count:unique_word_count",
        "func": make_unique_words,
        "group": "UNIQUE_WORDS",
        "conflicts_with": ["NO_WHITESPACE"]
    },
    {
        "id": "format:parentheses",
        "func": make_nested_parentheses,
        "group": "PARENTHESES",
        "conflicts_with": ["JSON", "NO_WHITESPACE"]
    },
    {
        "id": "words:palindrome",
        "func": make_palindrome,
        "group": "PALINDROME",
        "conflicts_with": ["JSON", "NO_WHITESPACE"]
    },
    {
        "id": "sentence:increment",
        "func": make_sentence_increment,
        "group": "SENTENCE_INCREMENT",
        "conflicts_with": ["JSON", "ALL_CAPS", "SENTENCE_STEP", "NO_WHITESPACE"]
    },
    {
        "id": "words:start_verb",
        "func": make_start_verb,
        "group": "START_VERB",
        "conflicts_with": ["JSON", "NO_WHITESPACE"]
    },
    {
        "id": "count:punctuation",
        "func": make_punctuation_count,
        "group": "PUNCTUATION_DIVERSITY",
        "conflicts_with": ["PUNCTUATION", "JSON", "NO_WHITESPACE"]
    },
    {
        "id": "words:alphabet_loop",
        "func": make_alphabet_loop,
        "group": "ALPHABET_LOOP",
        "conflicts_with": ["NO_CONSECUTIVE_LETTER", "PRIME_LENGTHS", "NO_WHITESPACE"]
    },
    {
        "id": "words:prime_lengths",
        "func": make_prime_lengths,
        "group": "PRIME_LENGTHS",
        "conflicts_with": ["ALPHABET_LOOP", "NO_WHITESPACE"]
    },
    {
        "id": "format:nested_quotes",
        "func": make_nested_quotes,
        "group": "NESTED_QUOTES",
        "conflicts_with": ["JSON", "NO_WHITESPACE"]
    },
    {
        "id": "count:numbers_count",
        "func": make_numbers_count,
        "group": "NUMBERS_COUNT",
        "conflicts_with": ["JSON"]
    },
    {
        "id": "format:title_case",
        "func": make_title_case,
        "group": "TITLE_CASE",
        "conflicts_with": ["ALL_CAPS", "ALL_LOWER", "JSON", "NO_WHITESPACE"]
    },
    {
        "id": "sentence:last_word_first_next",
        "func": make_last_word_first_next,
        "group": "LAST_FIRST_CHAIN",
        "conflicts_with": ["JSON", "NO_WHITESPACE"]
    },
    {
        "id": "words:no_consecutive_first_letter",
        "func": make_no_consecutive_first_letter,
        "group": "NO_CONSECUTIVE_LETTER",
        "conflicts_with": ["ALPHABET_LOOP", "NO_WHITESPACE"]
    },
    {
        "id": "words:limited_repeat",
        "func": make_limited_repeat,
        "group": "LIMITED_REPEAT",
        "conflicts_with": ["JSON", "NO_WHITESPACE"]
    },
    {
        "id": "sentence:word_count_step",
        "func": make_word_count_step,
        "group": "SENTENCE_STEP",
        "conflicts_with": ["SENTENCE_INCREMENT", "JSON", "NO_WHITESPACE"]
    },
    {
        "id": "format:date_format_list",
        "func": make_date_format_list,
        "group": "DATE_FORMAT",
        "conflicts_with": ["JSON", "NO_WHITESPACE"]
    },
    {
        "id": "format:csv_format",
        "func": make_csv_format,
        "group": "CSV_FORMAT",
        "conflicts_with": ["JSON", "NO_WHITESPACE", "OUTPUT_TEMPLATE", "INDENT_STAIRS"]
    },
    {
        "id": "format:output_template",
        "func": make_output_template,
        "group": "OUTPUT_TEMPLATE",
        "conflicts_with": ["CSV_FORMAT", "JSON", "NO_WHITESPACE"]
    },
    {
        "id": "words:paragraph_last_first_match",
        "func": make_paragraph_last_first_match,
        "group": "PARA_MATCH",
        "conflicts_with": ["JSON", "NO_WHITESPACE"]
    },
    {
        "id": "words:conjunction_count",
        "func": make_conjunction_count,
        "group": "CONJUNCTION_COUNT",
        "conflicts_with": ["JSON", "NO_WHITESPACE"]
    },
    {
        "id": "format:indent_stairs",
        "func": make_indent_stairs,
        "group": "INDENT_STAIRS",
        "conflicts_with": ["CSV_FORMAT", "JSON", "NO_WHITESPACE"]
    },
    {
        "id": "format:special_bullet",
        "func": make_special_bullet,
        "group": "SPECIAL_BULLET",
        "conflicts_with": ["JSON", "CSV_FORMAT", "NO_WHITESPACE"]
    },
    {
        "id": "manipulation:no_whitespace",
        "func": make_no_whitespace,
        "group": "NO_WHITESPACE",
        "conflicts_with": ["UNIQUE_WORDS", "PARENTHESES", "PALINDROME", "SENTENCE_INCREMENT", "START_VERB", "PUNCTUATION_DIVERSITY", "ALPHABET_LOOP", "PRIME_LENGTHS", "NESTED_QUOTES", "TITLE_CASE", "LAST_FIRST_CHAIN", "NO_CONSECUTIVE_LETTER", "LIMITED_REPEAT", "SENTENCE_STEP", "DATE_FORMAT", "CSV_FORMAT", "OUTPUT_TEMPLATE", "PARA_MATCH", "CONJUNCTION_COUNT", "INDENT_STAIRS", "SPECIAL_BULLET", "SYMBOL_END"]
    },
    {
        "id": "sentence:symbol_end",
        "func": make_symbol_end,
        "group": "SYMBOL_END",
        "conflicts_with": ["JSON", "NO_WHITESPACE"]
    }
]


def sample_compatible_constraints(k: int, pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sample k compatible, non-conflicting constraints from a specified pool."""
    available = list(pool)
    selected = []
    forbidden_groups = set()

    random.shuffle(available)

    for item in available:
        if len(selected) >= k:
            break
        # Check if item conflict group is forbidden
        if item["group"] in forbidden_groups:
            continue
        # Check if item conflicts with already selected groups
        if any(g in forbidden_groups for g in item["conflicts_with"]):
            continue

        selected.append(item)
        forbidden_groups.add(item["group"])
        forbidden_groups.update(item["conflicts_with"])

    return selected


# ==============================================================================
# 3. Prompt Augmentation Engine
# ==============================================================================

def clean_instruction(text: str) -> Optional[str]:
    """Clean and filter out base instructions that already enforce hard constraints."""
    if not text or len(text.strip()) < 15:
        return None
    
    # Filter out instructions already having rigid format/length demands
    lower = text.lower()
    triggers = [
        "json format", "in json", "word limit", "words or less", 
        "bullet points", "numbered list", "all caps", "lowercase",
        "table format", "csv format"
    ]
    if any(t in lower for t in triggers):
        return None
    
    return text.strip()

def augment_instruction(
    base_text: str, 
    k: int, 
    constraint_split: str = "seen"
) -> Dict[str, Any]:
    """Combine base instruction with k compatible verifiable constraints from chosen split."""
    pool = SEEN_CONSTRAINT_DEFINITIONS if constraint_split == "seen" else UNSEEN_CONSTRAINT_DEFINITIONS
    chosen_defs = sample_compatible_constraints(k, pool=pool)
    
    constraint_texts = []
    instruction_ids = []
    kwargs_list = []

    for c_def in chosen_defs:
        inst_id, desc_text, kwargs = c_def["func"]()
        constraint_texts.append(desc_text)
        instruction_ids.append(inst_id)
        kwargs_list.append(kwargs)

    # Format into complete prompt
    if len(constraint_texts) == 1:
        augmented_prompt = f"{base_text}\n\nConstraint: {constraint_texts[0]}"
    else:
        bullet_list = "\n".join([f"- {ct}" for ct in constraint_texts])
        augmented_prompt = (
            f"{base_text}\n\n"
            f"Please adhere strictly to the following constraints in your response:\n"
            f"{bullet_list}"
        )

    # Ground truth format: list of dict with 'instruction_id' and 'kwargs' lists
    ground_truth = [{
        "instruction_id": instruction_ids,
        "kwargs": kwargs_list
    }]

    return {
        "base_instruction": base_text,
        "prompt": augmented_prompt,
        "num_constraints": len(chosen_defs),
        "constraint_type": "single" if len(chosen_defs) == 1 else "multi",
        "constraint_ids": instruction_ids,
        "constraints_description": constraint_texts,
        "ground_truth": ground_truth,
        "constraint_split": constraint_split
    }


# ==============================================================================
# 4. Stream Processor & Dataset Exporter
# ==============================================================================

def run_augmentation(
    dataset_name: str,
    output_path: str,
    num_samples: int = 1000,
    k_weights: Optional[List[float]] = None,
    resume: bool = False,
    append: bool = False,
    constraint_split: str = "seen"
):
    """
    Stream dataset, augment prompts, and write JSONL line-by-line.
    Memory usage: ~100MB (constant throughout).
    
    Supports:
    - resume: skips existing lines and continues until total samples in file reaches num_samples.
    - append: appends num_samples new samples to the existing file.
    - constraint_split: 'seen' (IFEval + IFTrain) or 'unseen' (IFBench OOD).
    """
    import os

    existing_keys = set()
    existing_count = 0

    if (resume or append) and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "key" in data:
                        existing_keys.add(data["key"])
                    existing_count += 1
                except json.JSONDecodeError:
                    existing_count += 1
        print(f"Detected existing file '{output_path}' with {existing_count} samples.")

    if resume:
        if existing_count >= num_samples:
            print(f"File already contains {existing_count} samples (>= target {num_samples}). Nothing to do.")
            print("Tip: Increase --num_samples or use --append to add more.")
            return
        target_to_add = num_samples - existing_count
        print(f"Resuming: Target is {num_samples} total. Will generate {target_to_add} more samples (split={constraint_split})...")
        open_mode = "a"
    elif append and existing_count > 0:
        target_to_add = num_samples
        print(f"Appending {target_to_add} new samples (split={constraint_split}) to existing {existing_count} samples...")
        open_mode = "a"
    else:
        target_to_add = num_samples
        open_mode = "w"

    print(f"Loading dataset: {dataset_name} in streaming mode (Split: {constraint_split})...")
    ds = load_dataset(dataset_name, split="train", streaming=True)

    # Probabilities for K=1, 2, 3, 4, 5
    if not k_weights:
        k_options = [1, 2, 3, 4, 5]
        k_weights = [0.20, 0.35, 0.25, 0.15, 0.05]
    else:
        k_options = list(range(1, len(k_weights) + 1))

    saved_this_run = 0
    skipped_existing = 0

    with open(output_path, open_mode, encoding="utf-8") as f_out:
        for idx, item in enumerate(ds):
            sample_key = f"{dataset_name.split('/')[-1]}_{idx}"
            if sample_key in existing_keys:
                skipped_existing += 1
                continue

            # Extract base instruction and reference response depending on dataset structure
            base_instruction = None
            reference_response = None
            if "instruction" in item:
                # Alpaca format
                inst = item["instruction"]
                inp = item.get("input", "")
                reference_response = item.get("output", "")
                if inp and len(inp.strip()) > 0:
                    base_instruction = f"{inst}\n\nInput Context:\n{inp}"
                else:
                    base_instruction = inst
            elif "messages" in item:
                # GPT4-self-instruct format (list of message dicts)
                msgs = item["messages"]
                for m in msgs:
                    if m.get("role") == "user":
                        base_instruction = m.get("content")
                    elif m.get("role") == "assistant":
                        reference_response = m.get("content")
            
            cleaned = clean_instruction(base_instruction)
            if not cleaned:
                continue

            # Pick number of constraints K
            k = random.choices(k_options, weights=k_weights, k=1)[0]
            
            aug = augment_instruction(cleaned, k=k, constraint_split=constraint_split)
            
            # Format record exactly matching user requested schema
            record = {
                "key": sample_key,
                "dataset_source": dataset_name,
                "base_instruction": cleaned,
                "reference_response": reference_response,
                "prompt": aug["prompt"],
                "num_constraints": aug["num_constraints"],
                "constraint_type": aug["constraint_type"],
                "constraint_ids": aug["constraint_ids"],
                "constraints_description": aug["constraints_description"],
                "ground_truth": aug["ground_truth"],
                "constraint_split": constraint_split,
                "messages": [{"role": "user", "content": aug["prompt"]}]
            }
            
            # Write line-by-line
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved_this_run += 1
            current_total = existing_count + saved_this_run

            if saved_this_run % 100 == 0 or saved_this_run == target_to_add:
                print(f"Progress: +{saved_this_run}/{target_to_add} newly generated (Total in file: {current_total})...")

            if saved_this_run >= target_to_add:
                break

    final_total = existing_count + saved_this_run
    print(f"Done! Newly added: {saved_this_run}. Total samples in '{output_path}': {final_total}")

    # Automatically sync formatted .json file
    if output_path.endswith('.jsonl'):
        json_path = output_path[:-1]
        try:
            with open(output_path, 'r', encoding='utf-8') as f_in:
                all_records = [json.loads(line) for line in f_in if line.strip()]
            with open(json_path, 'w', encoding='utf-8') as f_json:
                json.dump(all_records, f_json, indent=2, ensure_ascii=False)
            print(f"✓ Synced {len(all_records)} records to formatted JSON: '{json_path}'")
        except Exception as e:
            print(f"Warning: Could not sync .json: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment base instructions with verifiable constraints.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="vicgalle/alpaca-gpt4", 
        choices=["vicgalle/alpaca-gpt4", "Post-training-Data-Flywheel/gpt4-self-instruct"],
        help="HuggingFace dataset to augment"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="datasets/augmented_rlvr_prompts.jsonl", 
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=500, 
        help="Target number of samples (or additional samples if --append is used)"
    )
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume generation: skip existing samples and continue until total count reaches --num_samples"
    )
    parser.add_argument(
        "--append", 
        action="store_true", 
        help="Append mode: generate --num_samples NEW samples and append to existing file"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="seen", 
        choices=["seen", "unseen"],
        help="Constraint split: 'seen' for training or 'unseen' for OOD evaluation"
    )
    args = parser.parse_args()

    run_augmentation(
        dataset_name=args.dataset,
        output_path=args.output,
        num_samples=args.num_samples,
        resume=args.resume,
        append=args.append,
        constraint_split=args.split
    )
