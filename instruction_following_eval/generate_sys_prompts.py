#!/usr/bin/env python3
# coding=utf-8
"""Script to generate detailed system prompts from Q&A data using Gemini API.

Features:
- Reads Q&A pairs from input_sys_data.jsonl
- Generates structured system prompts with 10 constraint dimensions
- Uses Gemini 3.5 Flash Lite (gemini-3.5-flash-lite) via google-genai SDK
- Bilingual output (Vietnamese + English)
- Supports resume (--resume): skips already-processed records
- Retry logic with exponential backoff
- Reads GOOGLE_API_KEY from .env

Usage:
    python generate_sys_prompts.py
    python generate_sys_prompts.py --input data/input_sys_data.jsonl --output data/output_sys_prompts.jsonl
    python generate_sys_prompts.py --index 0          # Process single record
    python generate_sys_prompts.py --resume            # Resume interrupted run
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .env loader (no external dependency)
# ---------------------------------------------------------------------------
def load_env_file(env_path: Optional[str] = None) -> Dict[str, str]:
    """Parse key-value pairs from .env file."""
    candidates = []
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(os.getcwd(), ".env"))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, ".env"))

    env_vars = {}
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        env_vars[key.strip()] = val.strip().strip("'\"")
                logger.info("Loaded .env from: %s", path)
            except Exception as e:
                logger.warning("Could not read .env at %s: %s", path, e)
            break
    return env_vars


# ---------------------------------------------------------------------------
# Meta-prompt template
# ---------------------------------------------------------------------------

def get_gold_example_json() -> str:
    """Load the gold output_format.json file as the few-shot reference."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gold_path = os.path.join(script_dir, "output_format.json")
    if os.path.isfile(gold_path):
        try:
            with open(gold_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.warning("Could not read output_format.json: %s", e)
    return "{}"


META_PROMPT_HEADER = r"""You are an expert prompt engineer and conversational dataset designer for Large Language Models (LLMs).

## Task
Given an input user question and a sample gold answer (provided in both Vietnamese and English), you must reverse-engineer the conversation flow and craft an EXTREMELY DETAILED system prompt turn for an instruction-following / tool-calling evaluation dataset.

The output must follow the EXACT JSON schema demonstrated in the FEW-SHOT EXAMPLE below:
- "tools": A list of 2-3 function definitions (1 directly relevant tool that provides context for answering the question, and 1-2 plausible distractor tools like web search or weather).
- "messages": An array representing the conversation:
  1. role: "system" - The detailed system prompt with structured XML tags.
  2. role: "user" - The original Vietnamese question.
  3. role: "assistant" - A tool call to the relevant tool with realistic arguments.
  4. role: "tool" - The simulated tool output providing necessary background/context.
  5. role: "assistant" - The exact original Vietnamese answer.

---

## CRITICAL: System Prompt Turn Requirements ("content" of role "system")
The "content" string of the "system" message must be written entirely in natural Vietnamese and strictly enclosed within XML-style tags as follows:

1. `<role>` ... `</role>`:
   - Vai trò và bản sắc của trợ lý (persona), thái độ, góc nhìn đối với chủ đề.
   - Nguyên tắc ứng xử, không được tỏ ra như thế nào (không khô khan, không như robot, không đạo lý).

2. `<response_length>` ... `</response_length>`:
   - BẮT BUỘC bắt đầu bằng: `GIỚI HẠN CỨNG: câu trả lời ...`
   - Nêu chính xác số câu trích xuất từ câu trả lời mẫu (ví dụ: đúng 2 câu, hoặc đúng 1 cụm từ/tên bài hát, hoặc đúng 7 câu).
   - Nêu khoảng số từ tiếng Việt (ví dụ: khoảng 20–25 từ tiếng Việt).
   - Quy định cấu trúc từng câu (ví dụ: Câu 1 làm gì, Câu 2 làm gì...).
   - Quy định không nhắc lại câu hỏi, không mở bài dài, không lặp ý, trả lời thẳng vào điều được hỏi.

3. `<context_and_addressing>` ... `</context_and_addressing>`:
   - Quy định đại từ xưng hô chính xác đối với người dùng (ví dụ: "bạn", "ông") và tự xưng (ví dụ: "tôi", hoặc xưng tên).
   - Các danh xưng trang trọng bị cấm tuyệt đối (ví dụ: không dùng "anh/chị", "ông/bà", "quý khách", "người dùng").
   - Quy định về ngôn ngữ (chỉ dùng tiếng Việt tự nhiên) và quy tắc dùng emoji (không dùng emoji trừ khi cần thiết).

4. `<content_guidelines>` ... `</content_guidelines>`:
   - Định hướng nội dung trọng tâm cần trả lời dựa trên câu hỏi và câu trả lời mẫu.
   - Những khía cạnh cần tập trung hoặc cách nhìn nhận vấn đề.

5. `<style>` ... `</style>`:
   - Sắc thái giọng văn (hóm hỉnh, mỉa mai, bức xúc, thẳng thắn, khách quan, v.v.).
   - Các mẫu câu tu từ hoặc cách diễn đạt tự nhiên đặc trưng (ví dụ: "Chẳng phải...", "Cứ như thể...", "Đúng là...").
   - Ranh giới cảm xúc (đùa vui thân mật chứ không xúc phạm; hoặc bức xúc kiểu người dùng trên diễn đàn).

6. `<do_not>` ... `</do_not>`:
   - Tập trung toàn bộ các điều cấm kỵ (Negative constraints):
     - Không giải thích chi tiết lý do hay phân tích dài dòng.
     - Không tạo danh sách hay bullet points.
     - Không đặt câu hỏi tiếp nối hay hỏi ngược lại người dùng.
     - Không lên lớp đạo lý hay tự ý đưa ra lời khuyên.
     - Không để lộ suy luận nội bộ hay công cụ tìm kiếm.

7. `<conversation_and_tools>` ... `</conversation_and_tools>`:
   - Hướng dẫn khi nào cần gọi công cụ (tool) nào trước khi trả lời.
   - Cấm gọi các công cụ không liên quan.
   - Tuyệt đối không để lộ kết quả thô của tool và không mô tả việc gọi tool.
   - Lệnh gọi tool là hành động nội bộ, sau khi tool hoàn tất chỉ xuất ra câu trả lời hội thoại cuối cùng.

---

## FEW-SHOT GOLD EXAMPLE

### Input:
Question (VI): Sao cứ mỗi lần tôi đi tắm là bạn gái tôi lại muốn vào chung nhỉ?
Answer (VI): Chẳng phải là kinh khủng lắm sao? Cứ như thể là không bao giờ đủ nước nóng cho cả hai vậy!
Question (EN): Why whenever I get in the shower my girlfriend want to join?
Answer (EN): Isn’t it awful? You would swear that there wasn’t enough hot water to go around!

### Expected Output:
```json
__GOLD_EXAMPLE_JSON__
```

---

## YOUR NEW INPUT DATA

Question (VI): __QUESTION_VI__
Answer (VI): __ANSWER_VI__
Question (EN): __QUESTION_EN__
Answer (EN): __ANSWER_EN__

### PRE-COMPUTED LENGTH METRICS (COMPUTED DETERMINISTICALLY VIA REGEX):
- Exact sentence count: __EXACT_NUM_SENTENCES__ câu
- Exact word count: __EXACT_WORD_COUNT__ từ
- Word range specification: __WORD_RANGE_DESC__ tiếng Việt
- Gold sentences breakdown:
__SENTENCE_BREAKDOWN_LIST__

CRITICAL INSTRUCTION ON <response_length>:
You MUST use the exact sentence count (__EXACT_NUM_SENTENCES__ câu) and word count specification (__WORD_RANGE_DESC__ tiếng Việt).
In the sentence role breakdown, you MUST break down exactly __EXACT_NUM_SENTENCES__ sentences (from Câu 1 to Câu __EXACT_NUM_SENTENCES__) matching each sentence above. Do NOT invent extra sentences!

All tags (<role>, <response_length>, <context_and_addressing>, <content_guidelines>, <style>, <do_not>, <conversation_and_tools>) in the system content must be in Vietnamese.
Return ONLY valid JSON matching the exact schema above."""


# ---------------------------------------------------------------------------
# Gemini API caller using raw HTTP (urllib) — no SDK dependency needed
# ---------------------------------------------------------------------------
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def create_gemini_client(api_key: str) -> str:
    """Return the API key itself (used by call_gemini)."""
    # Quick validation: try a lightweight API call
    logger.info("Gemini client ready (using raw HTTP API).")
    return api_key


def call_gemini(
    api_key: str,
    prompt: str,
    model: str = "gemini-3.5-flash-lite",
    max_retries: int = 3,
    initial_delay: float = 2.0,
) -> Optional[str]:
    """Call Gemini API via raw HTTP with retry logic and exponential backoff."""
    import urllib.request
    import urllib.error

    url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }

    data = json.dumps(payload).encode("utf-8")

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            # Extract text from Gemini response
            candidates = body.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")

            logger.warning(
                "Attempt %d/%d: Empty/unexpected response structure", attempt, max_retries
            )

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            delay = initial_delay * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d HTTP %d: %s. Body: %s. Retrying in %.1fs...",
                attempt, max_retries, e.code, str(e), error_body, delay,
            )
            if attempt < max_retries:
                time.sleep(delay)
            else:
                logger.error("All %d attempts failed.", max_retries)
                return None

        except Exception as e:
            delay = initial_delay * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt, max_retries, str(e), delay,
            )
            if attempt < max_retries:
                time.sleep(delay)
            else:
                logger.error("All %d attempts failed.", max_retries)
                return None

    return None


# ---------------------------------------------------------------------------
# JSON parsing with fallback
# ---------------------------------------------------------------------------
def parse_json_response(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = raw_text.strip()

    # Remove markdown code block wrapper if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error: %s", e)
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.error("Could not parse JSON from response:\n%s", text[:500])
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
REQUIRED_SECTIONS = [
    "<role>",
    "<response_length>",
    "<context_and_addressing>",
    "<content_guidelines>",
    "<style>",
    "<do_not>",
    "<conversation_and_tools>",
]


def validate_output(data: Dict[str, Any]) -> List[str]:
    """Validate that the output has the required structure matching output_format.json."""
    warnings = []
    if "messages" not in data or not isinstance(data["messages"], list):
        warnings.append("Missing or invalid 'messages' array")
        return warnings

    if len(data["messages"]) < 1:
        warnings.append("'messages' array is empty")
        return warnings

    first_msg = data["messages"][0]
    if first_msg.get("role") != "system":
        warnings.append(f"First message role is '{first_msg.get('role')}', expected 'system'")

    content = first_msg.get("content", "")
    for section in REQUIRED_SECTIONS:
        if section not in content:
            warnings.append(f"System prompt content missing section: '{section}'")

    if "tools" not in data or not isinstance(data["tools"], list):
        warnings.append("Missing or invalid 'tools' list")

    return warnings


# ---------------------------------------------------------------------------
# Deterministic answer length analyzer (Regex-based)
# ---------------------------------------------------------------------------
def analyze_gold_answer(text: str) -> Dict[str, Any]:
    """Deterministically analyze answer text using regex to extract exact sentence and word counts."""
    cleaned = text.strip()
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.?!…])\s+", cleaned) if s.strip()]
    if not raw_sentences and cleaned:
        raw_sentences = [cleaned]

    num_sentences = len(raw_sentences)
    words = re.findall(r"\b\w+(?:[-’']\w+)*\b", cleaned, re.UNICODE)
    num_words = len(words)

    if num_words <= 5:
        word_range_desc = f"chính xác {num_words} từ"
    elif num_words <= 30:
        min_w = max(1, num_words - 3)
        max_w = num_words + 3
        word_range_desc = f"khoảng {min_w}–{max_w} từ"
    else:
        min_w = max(1, (num_words // 10) * 10 - 5)
        max_w = ((num_words + 9) // 10) * 10 + 5
        word_range_desc = f"khoảng {min_w}–{max_w} từ"

    sentence_breakdown_lines = []
    for i, s in enumerate(raw_sentences, 1):
        sentence_breakdown_lines.append(f"  - Câu {i}: \"{s}\"")
    sentence_breakdown_text = "\n".join(sentence_breakdown_lines)

    return {
        "num_sentences": num_sentences,
        "num_words": num_words,
        "word_range_desc": word_range_desc,
        "sentences": raw_sentences,
        "sentence_breakdown_text": sentence_breakdown_text,
    }


def enforce_response_length_tag(system_content: str, stats: Dict[str, Any]) -> str:
    """Ensure <response_length> tag strictly specifies the deterministic sentence and word count."""
    num_s = stats["num_sentences"]
    word_range = stats["word_range_desc"]

    pattern = r"(<response_length>)(.*?)(</response_length>)"
    match = re.search(pattern, system_content, re.DOTALL)
    if not match:
        return system_content

    tag_body = match.group(2).strip()
    lines = [line for line in tag_body.split("\n") if line.strip()]

    # Canonical headers
    if num_s == 1 and stats["num_words"] <= 5:
        hard_limit = f"GIỚI HẠN CỨNG: Câu trả lời chứa đúng {num_s} câu (hoặc một cụm từ/tên riêng)."
    else:
        hard_limit = f"GIỚI HẠN CỨNG: Câu trả lời phải có đúng {num_s} câu."
    length_desc = f"Tổng độ dài {word_range} tiếng Việt."

    # Filter out any old GIỚI HẠN CỨNG and Tổng độ dài lines generated by LLM
    rest_lines = []
    for line in lines:
        s = line.strip()
        if s.startswith("GIỚI HẠN CỨNG:") or s.startswith("Tổng độ dài"):
            continue
        rest_lines.append(line)

    new_body = f"{hard_limit}\n{length_desc}\n" + "\n".join(rest_lines).strip()
    return (
        system_content[: match.start(1)]
        + f"<response_length>\n{new_body}\n</response_length>"
        + system_content[match.end(3) :]
    )


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------
def build_meta_prompt(record: Dict[str, Any], stats: Dict[str, Any]) -> str:
    """Build the meta-prompt from a Q&A record using safe string substitution."""
    gold_example_json = get_gold_example_json()
    prompt = META_PROMPT_HEADER
    prompt = prompt.replace("__GOLD_EXAMPLE_JSON__", gold_example_json)
    prompt = prompt.replace("__QUESTION_VI__", record.get("question", ""))
    prompt = prompt.replace("__ANSWER_VI__", record.get("answer", ""))
    prompt = prompt.replace("__QUESTION_EN__", record.get("question_en", ""))
    prompt = prompt.replace("__ANSWER_EN__", record.get("answer_en", ""))
    prompt = prompt.replace("__EXACT_NUM_SENTENCES__", str(stats["num_sentences"]))
    prompt = prompt.replace("__EXACT_WORD_COUNT__", str(stats["num_words"]))
    prompt = prompt.replace("__WORD_RANGE_DESC__", stats["word_range_desc"])
    prompt = prompt.replace("__SENTENCE_BREAKDOWN_LIST__", stats["sentence_breakdown_text"])
    return prompt


def load_processed_indices(output_path: str) -> set:
    """Load indices of already-processed records for resume support."""
    processed = set()
    if os.path.isfile(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "index" in obj:
                            processed.add(obj["index"])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning("Could not read output file for resume: %s", e)
    return processed


def process_record(
    client,
    record: Dict[str, Any],
    model: str,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Process a single Q&A record and return the enriched output."""
    index = record.get("index", "?")
    logger.info("Processing index=%s: %s", index, record.get("question", "")[:60])

    # If index == 0, check if we have the gold reference from output_format.json
    if index == 0:
        gold_str = get_gold_example_json()
        if gold_str and gold_str != "{}":
            try:
                gold_obj = json.loads(gold_str)
                output = {
                    "index": 0,
                    "tools": gold_obj.get("tools", []),
                    "messages": gold_obj.get("messages", []),
                }
                logger.info("✓ index=0 used gold reference directly from output_format.json")
                return output
            except Exception as e:
                logger.warning("Could not parse gold output_format.json, falling back to LLM: %s", e)

    # Deterministic length analysis via regex
    stats = analyze_gold_answer(record.get("answer", ""))
    logger.info(
        "index=%s regex stats: %d sentences, %d words (%s)",
        index,
        stats["num_sentences"],
        stats["num_words"],
        stats["word_range_desc"],
    )

    meta_prompt = build_meta_prompt(record, stats)
    raw_response = call_gemini(client, meta_prompt, model=model, max_retries=max_retries)

    if raw_response is None:
        logger.error("Failed to get response for index=%s", index)
        return None

    parsed = parse_json_response(raw_response)
    if parsed is None:
        logger.error("Failed to parse JSON for index=%s", index)
        return None

    # Enforce deterministic response_length in system message
    if parsed.get("messages") and parsed["messages"][0].get("role") == "system":
        raw_sys = parsed["messages"][0].get("content", "")
        parsed["messages"][0]["content"] = enforce_response_length_tag(raw_sys, stats)

    # Validate
    warnings = validate_output(parsed)
    if warnings:
        for w in warnings:
            logger.warning("index=%s validation: %s", index, w)

    # Build output record matching output_format.json schema
    output = {
        "index": record.get("index"),
        "tools": parsed.get("tools", []),
        "messages": parsed.get("messages", []),
    }

    sys_snippet = ""
    if output["messages"] and "content" in output["messages"][0]:
        sys_snippet = output["messages"][0]["content"][:80].replace("\n", " ")

    logger.info(
        "✓ index=%s done. System prompt snippet: %s...",
        index,
        sys_snippet,
    )
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed system prompts from Q&A data using Gemini API."
    )
    parser.add_argument(
        "--input",
        default="data/input_sys_data.jsonl",
        help="Path to input JSONL file (default: data/input_sys_data.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/output_sys_prompts.jsonl",
        help="Path to output JSONL file (default: data/output_sys_prompts.jsonl)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.5-flash-lite",
        help="Gemini model name (default: gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Process only a single record by index",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip already-processed records (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Process all records from scratch (overwrite output)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts per API call (default: 3)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google API key (overrides .env)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "json"],
        default="json",
        help="Output format: 'jsonl' (one JSON per line) or 'json' (pretty-printed array) (default: json)",
    )
    args = parser.parse_args()

    # Handle resume logic
    if args.no_resume:
        args.resume = False

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = (
        args.input
        if os.path.isabs(args.input)
        else os.path.join(script_dir, args.input)
    )
    output_path = (
        args.output
        if os.path.isabs(args.output)
        else os.path.join(script_dir, args.output)
    )

    # Load API key
    env_vars = load_env_file()
    api_key = args.api_key or env_vars.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error(
            "GOOGLE_API_KEY not found. Set it in .env, environment variable, or use --api-key flag."
        )
        sys.exit(1)

    # Read input
    if not os.path.isfile(input_path):
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON at line %d: %s", line_num, e)

    logger.info("Loaded %d records from: %s", len(records), input_path)

    # Filter by index if specified
    if args.index is not None:
        records = [r for r in records if r.get("index") == args.index]
        if not records:
            logger.error("No record found with index=%d", args.index)
            sys.exit(1)
        logger.info("Filtered to index=%d", args.index)

    # Resume: skip already-processed
    if args.resume:
        processed = load_processed_indices(output_path)
        if processed:
            before = len(records)
            records = [r for r in records if r.get("index") not in processed]
            logger.info(
                "Resume mode: %d already processed, %d remaining",
                before - len(records),
                len(records),
            )

    if not records:
        logger.info("No records to process. Exiting.")
        return

    # Create Gemini client
    client = create_gemini_client(api_key)
    logger.info("Using model: %s", args.model)

    # Process records
    success_count = 0
    fail_count = 0
    # Process and collect results
    all_results = []

    file_mode = "a" if args.resume else "w"
    with open(output_path, file_mode, encoding="utf-8") as out_f:
        for i, record in enumerate(records):
            logger.info("─" * 60)
            logger.info("Progress: %d/%d", i + 1, len(records))

            result = process_record(
                client, record, model=args.model, max_retries=args.max_retries
            )

            if result:
                json_line = json.dumps(result, ensure_ascii=False)
                out_f.write(json_line + "\n")
                out_f.flush()
                all_results.append(result)
                success_count += 1
            else:
                fail_count += 1

            # Rate limiting delay between calls
            if i < len(records) - 1:
                time.sleep(args.delay)

    # If JSON format requested, re-write as pretty-printed JSON array
    if args.format == "json":
        # Read all records (including previously processed ones for resume)
        all_records = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        json_output_path = output_path.replace(".jsonl", ".json")
        if json_output_path == output_path:
            json_output_path = output_path + ".json"

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        logger.info("JSON output written to: %s", json_output_path)

    # Summary
    logger.info("═" * 60)
    logger.info("DONE! Success: %d | Failed: %d | Total: %d", success_count, fail_count, len(records))
    logger.info("Output written to: %s", output_path)
    if args.format == "json":
        logger.info("Pretty JSON also at: %s", json_output_path)


if __name__ == "__main__":
    main()
