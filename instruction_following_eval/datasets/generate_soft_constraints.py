"""
Task 2 Data Synthesis Engine: Augment Hard Constraints with Soft Constraints
Uses Gemini (gemini-3.5-flash-lite) to generate semantic, stylistic, and tone constraints (VerIF methodology).

Features:
- CLI flags for fine-grained chunk processing:
    --id / --key: Process a single record (by key or numeric ID)
    --start-idx / --end-idx: Process a specific range (e.g. 0 to 50)
    --limit: Max records to process in one execution
- Built-in 429 Rate-Limit resilience:
    Exponential backoff on HTTP 429 (Resource Exhausted)
    Configurable delay between calls (--delay)
- Safe Resume (--resume): Skips already processed keys
- Dual format output: Streams to .jsonl and syncs to pretty .json matching sample_task2_format.json
"""

import os
import sys
import json
import time
import re
import logging
import argparse
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Set

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_soft_constraints")

# ---------------------------------------------------------------------------
# API Key resolution
# ---------------------------------------------------------------------------
def resolve_api_key(cli_key: Optional[str] = None) -> str:
    """Retrieve Gemini API key from CLI, env var, or local .env file."""
    if cli_key:
        return cli_key
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key
    
    # Search for .env in current dir and parent dirs
    search_dirs = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ]
    for d in search_dirs:
        env_file = os.path.join(d, ".env")
        if os.path.isfile(env_file):
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GOOGLE_API_KEY="):
                            k = line.split("=", 1)[1].strip().strip("\"'")
                            if k:
                                return k
            except Exception:
                pass
    return ""


# ---------------------------------------------------------------------------
# Prompt Generator for Soft Constraints (Reverse Instruction Engineering)
# ---------------------------------------------------------------------------
def build_soft_constraint_meta_prompt(
    base_instruction: str, 
    hard_constraints: List[str],
    reference_response: Optional[str] = None,
    split: str = "seen",
) -> str:
    """Constructs prompt for Gemini to reverse-engineer non-conflicting soft constraints from gold response."""
    hard_list_str = "\n".join([f"- {c}" for c in hard_constraints]) if hard_constraints else "- None"
    
    is_vietnamese = bool(re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", base_instruction.lower()))
    lang_instruction = (
        "The description and eval_rubric must be written in Vietnamese because the base instruction is in Vietnamese."
        if is_vietnamese
        else "The description and eval_rubric must be written in English because the base instruction is in English."
    )

    gold_section = ""
    if reference_response and len(reference_response.strip()) > 10:
        gold_section = f"""Gold Reference Response (Demonstrating high-quality benchmark completion):
\"\"\"
{reference_response.strip()[:2500]}
\"\"\"
"""

    if split == "unseen":
        task_instruction = """We are constructing an Out-Of-Domain (UNSEEN) evaluation benchmark for Instruction Following (Double-OOD Generalization).
In the training (SEEN) set, models were trained exclusively on basic 'semantic_completeness' and generic 'style_and_tone'.
To rigorously evaluate the model's true generalization capacity on novel, unobserved soft constraints, you MUST REVERSE-ENGINEER 1 to 2 novel SOFT CONSTRAINTS chosen strictly from the following OUT-OF-DOMAIN (UNSEEN) categories (DO NOT use standard 'semantic_completeness' or simple 'style_and_tone'):

OUT-OF-DOMAIN (UNSEEN) CATEGORIES:
1. 'target_audience': Explicitly adapt the depth, tone, and framing for a specific audience profile (e.g. elementary school students, non-technical beginners, executive decision-makers, peer researchers).
2. 'reasoning_and_logic': Require explicit multi-step reasoning, explaining the cause-and-effect mechanism or comparative rationale behind recommendations.
3. 'role_play_persona': Strictly maintain a distinctive professional, occupational, or historical persona (e.g. a seasoned investigative journalist, a senior safety inspector, an ancient historian).
4. 'counterfactual_context': Adhere strictly to a counterfactual, hypothetical, or scenario-bounded premise.
5. 'safety_and_neutrality': Present balanced multi-perspective views objectively without taking sides on contentious matters.

Select 1 or 2 UNSEEN categories that fit naturally with the Base Instruction and Gold Response.
"""
        json_example = """[
  {
    "id": "<unseen_category>:<concise_identifier>",
    "category": "<target_audience | reasoning_and_logic | role_play_persona | counterfactual_context | safety_and_neutrality>",
    "description": "<Clear instruction specifying the novel requirement>",
    "eval_rubric": "<Clear evaluation criteria for an LLM Judge returning binary 1 (pass) or 0 (fail)>"
  }
]"""
    elif reference_response and len(reference_response.strip()) > 10:
        task_instruction = """We have an instruction task with existing hard constraints, along with an expert Gold Reference Response.
Your goal is to REVERSE-ENGINEER 1 to 2 meaningful, non-conflicting SOFT CONSTRAINTS (categories: 'semantic_completeness' and 'style_and_tone') that capture the key qualities of the Gold Response, turning them into explicit requirements for the student model.

GUIDANCE FOR REVERSE-ENGINEERING:
1. Category 'semantic_completeness': Identify the core conceptual pillars, principles, or depth of analysis demonstrated in the Gold Response. Frame this as a clear requirement (e.g. 'Must cover both X and Y aspects'). DO NOT copy verbatim trivia, exact numbers, or leak the exact answer.
2. Category 'style_and_tone': Identify the persona, professional register, or stylistic tone displayed in the Gold Response (e.g. 'Maintain an authoritative yet accessible clinical tone').
"""
        json_example = """[
  {
    "id": "semantic:<concise_identifier>",
    "category": "semantic_completeness",
    "description": "<Clear instruction specifying what specific semantic depth or key concept must be covered>",
    "eval_rubric": "<Clear evaluation criteria for an LLM Judge returning binary 1 (pass) or 0 (fail)>"
  },
  {
    "id": "tone:<concise_identifier>",
    "category": "style_and_tone",
    "description": "<Clear instruction specifying the tone, register, or style to maintain>",
    "eval_rubric": "<Clear evaluation criteria for an LLM Judge returning binary 1 (pass) or 0 (fail)>"
  }
]"""
    else:
        task_instruction = """We have an instruction task that already has hard constraints. Your goal is to generate 1 to 2 meaningful, non-conflicting SOFT CONSTRAINTS (categories: 'semantic_completeness' and 'style_and_tone') that test whether the model truly understands the subject matter and does not just 'hack' the hard formatting constraints."""
        json_example = """[
  {
    "id": "semantic:<concise_identifier>",
    "category": "semantic_completeness",
    "description": "<Clear instruction specifying what specific semantic depth or key concept must be covered>",
    "eval_rubric": "<Clear evaluation criteria for an LLM Judge returning binary 1 (pass) or 0 (fail)>"
  },
  {
    "id": "tone:<concise_identifier>",
    "category": "style_and_tone",
    "description": "<Clear instruction specifying the tone, register, or style to maintain>",
    "eval_rubric": "<Clear evaluation criteria for an LLM Judge returning binary 1 (pass) or 0 (fail)>"
  }
]"""

    prompt = f"""You are an AI research scientist designing training datasets for Instruction Following and VerIF (Verification Engineering for RLVR).

{task_instruction}

Base Instruction:
{base_instruction}

{gold_section}
Existing Hard Constraints:
{hard_list_str}

Language Requirement:
{lang_instruction}

Return a JSON array of 1 to 2 objects with this EXACT structure:
{json_example}

CRITICAL RULES:
1. Soft constraints MUST be realistically achievable within the limitations of the Existing Hard Constraints (e.g. if the hard constraint is 'under 4 sentences', the required concepts must be concise enough to fit).
2. DO NOT leak the exact answer into the description by verbatim copying details from the gold response. Focus on principles and core concepts.
3. Return ONLY valid JSON array. No explanations, no markdown code fence wrapping if possible.
"""
    return prompt


# ---------------------------------------------------------------------------
# Gemini API Caller with 429 Exponential Backoff
# ---------------------------------------------------------------------------
def call_gemini(
    api_key: str,
    prompt: str,
    model: str = "gemini-3.5-flash-lite",
    max_retries: int = 5,
    initial_delay: float = 3.0,
) -> Optional[str]:
    """Call Gemini API via raw HTTP with exponential backoff on 429."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2048,
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
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            candidates = body.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()

            logger.warning("Attempt %d/%d: Empty candidates response", attempt, max_retries)

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:400]
            except Exception:
                pass

            # Check for Rate Limit (429) or Resource Exhausted
            if e.code == 429 or "RESOURCE_EXHAUSTED" in error_body:
                delay = initial_delay * (2 ** (attempt - 1)) + 2.0
                logger.warning(
                    "⚠️ [Quota/Rate-Limit 429] Hit quota. Backing off for %.1fs (attempt %d/%d)...",
                    delay, attempt, max_retries,
                )
                time.sleep(delay)
            else:
                delay = initial_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt %d/%d HTTP %d: %s. Retrying in %.1fs...",
                    attempt, max_retries, e.code, error_body[:150], delay,
                )
                time.sleep(delay)

        except Exception as e:
            delay = initial_delay * (2 ** (attempt - 1))
            logger.warning("Attempt %d/%d error: %s. Retrying in %.1fs...", attempt, max_retries, str(e), delay)
            time.sleep(delay)

    logger.error("Failed to generate response after %d attempts.", max_retries)
    return None


def parse_soft_constraints(raw_text: str) -> Optional[List[Dict[str, Any]]]:
    """Parse and validate soft constraints JSON from Gemini output."""
    if not raw_text:
        return None
    # Strip markdown if present
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and len(data) > 0:
            validated = []
            for item in data:
                if isinstance(item, dict) and "description" in item and "eval_rubric" in item:
                    validated.append({
                        "id": str(item.get("id", f"soft_{len(validated)+1}")),
                        "category": str(item.get("category", "semantic_completeness")),
                        "description": str(item.get("description", "")).strip(),
                        "eval_rubric": str(item.get("eval_rubric", "")).strip(),
                    })
            return validated if validated else None
    except Exception as e:
        logger.warning("Failed to parse JSON array: %s. Raw: %s", e, raw_text[:200])
    return None


# ---------------------------------------------------------------------------
# Task 2 Record Assembler
# ---------------------------------------------------------------------------
def assemble_task2_record(
    record: Dict[str, Any],
    soft_constraints: List[Dict[str, Any]],
    model_judge: str = "google/gemma-4-31B-it",
) -> Dict[str, Any]:
    """Assemble complete sample matching sample_task2_format.json schema."""
    base_inst = record.get("base_instruction", record.get("instruction", "")).strip()
    hard_descs = record.get("constraints_description", [])
    hard_ids = record.get("constraint_ids", [])
    ground_truth = record.get("ground_truth", [])

    is_vietnamese = bool(re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", base_inst.lower()))

    # Section headers
    if is_vietnamese:
        hdr_intro = "Vui lòng tuân thủ nghiêm ngặt các yêu cầu sau trong phản hồi:"
        hdr_hard = "[Yêu cầu định dạng & cấu trúc]"
        hdr_soft = "[Yêu cầu nội dung & ngữ nghĩa]"
    else:
        hdr_intro = "Please adhere strictly to the following constraints in your response:"
        hdr_hard = "[Formatting & Structure Constraints]"
        hdr_soft = "[Content & Semantic Guidelines]"

    hard_bullet = "\n".join([f"- {h}" for h in hard_descs])
    soft_bullet = "\n".join([f"- {s['description']}" for s in soft_constraints])

    prompt = (
        f"{base_inst}\n\n"
        f"{hdr_intro}\n"
        f"{hdr_hard}\n"
        f"{hard_bullet}\n\n"
        f"{hdr_soft}\n"
        f"{soft_bullet}"
    )

    return {
        "key": record.get("key", f"sample_{int(time.time()*1000)}"),
        "dataset_source": record.get("dataset_source", "unknown"),
        "constraint_split": record.get("constraint_split", "seen"),
        "base_instruction": base_inst,
        "reference_response": record.get("reference_response", ""),
        "prompt": prompt,
        "num_constraints": {
            "total": len(hard_ids) + len(soft_constraints),
            "hard_count": len(hard_ids),
            "soft_count": len(soft_constraints),
        },
        "hard_constraints": {
            "verifier_type": "python_rule_engine",
            "constraint_ids": hard_ids,
            "constraints_description": hard_descs,
            "ground_truth": ground_truth,
        },
        "soft_constraints": {
            "verifier_type": "llm_reasoning_judge",
            "model_judge": model_judge,
            "constraints": soft_constraints,
        },
        "reward_spec": {
            "aggregation_strategy": "hard_priority_gated",
            "weights": {
                "w_hard": 0.75,
                "w_soft": 0.25,
            },
            "formula": "Reward = (0.75 * R_hard + 0.25 * R_soft) if (R_hard > 0 and R_soft >= 0.5) else 0.0",
            "anti_reward_hacking_note": "Nếu R_hard == 0 (không đạt quy tắc cứng) hoặc R_soft < 0.5 (sai lệch nội dung), toàn bộ reward sẽ bằng 0.",
        },
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Helpers for Resume and Output
# ---------------------------------------------------------------------------
def load_processed_keys(output_jsonl: str) -> Set[str]:
    """Read processed keys from existing JSONL file."""
    keys = set()
    if os.path.isfile(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "key" in obj:
                        keys.add(obj["key"])
                except Exception:
                    pass
    return keys


def sync_json_from_jsonl(jsonl_path: str, json_path: str):
    """Sync all lines from JSONL into a formatted JSON array file."""
    records = []
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("Synced %d records to JSON file: %s", len(records), json_path)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate Soft Constraints for Task 2 using Gemini Flash-Lite with range and quota control."
    )
    parser.add_argument(
        "--input",
        default="datasets/augmented_alpaca_gpt4.jsonl",
        help="Input JSON or JSONL file with hard constraints",
    )
    parser.add_argument(
        "--output",
        default="datasets/task2_hybrid_alpaca_gpt4.jsonl",
        help="Output JSONL file (a pretty .json file will also be synced)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.5-flash-lite",
        help="Gemini model name (default: gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--model-judge",
        default="google/gemma-4-31B-it",
        help="Judge model identifier in metadata (default: google/gemma-4-31B-it)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google API Key (defaults to GOOGLE_API_KEY in .env)",
    )

    # Range and selection flags
    parser.add_argument(
        "--id",
        "--key",
        dest="target_id",
        default=None,
        help="Process a single record by key (e.g. alpaca-gpt4_0) or numeric index",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start index (0-based, inclusive, e.g. 0)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End index (0-based, inclusive, e.g. 50)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process in this run",
    )

    # Rate limit and delay flags
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay in seconds between requests to avoid 429 rate limit (default: 1.5)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries with exponential backoff on 429 quota errors (default: 5)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip already processed records (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Do not resume; process from start",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=["auto", "seen", "unseen"],
        help="Constraint split mode: 'seen' for standard training soft constraints (semantic & style), 'unseen' for novel out-of-domain soft constraints (target_audience, reasoning_and_logic, role_play, counterfactual, neutrality). 'auto' infers from record (default: auto).",
    )

    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        logger.error("No GOOGLE_API_KEY found. Please set it in .env or via --api-key.")
        sys.exit(1)

    # Resolve paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = args.input if os.path.isabs(args.input) else os.path.join(base_dir, args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(base_dir, args.output)

    if not os.path.isfile(input_path):
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Load input records
    records = []
    logger.info("Reading input records from: %s", input_path)
    if input_path.endswith(".jsonl"):
        with open(input_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if "index" not in rec:
                            rec["_idx"] = idx
                        records.append(rec)
                    except Exception as e:
                        logger.warning("Skipping bad line %d: %s", idx, e)
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for idx, rec in enumerate(data):
                    rec["_idx"] = idx
                    records.append(rec)
            else:
                logger.error("Expected JSON array in %s", input_path)
                sys.exit(1)

    total_loaded = len(records)
    logger.info("Loaded %d input records.", total_loaded)

    # Filtering by single ID/key
    if args.target_id is not None:
        target = str(args.target_id).strip()
        matched = []
        for r in records:
            key_val = str(r.get("key", ""))
            idx_val = str(r.get("_idx", ""))
            if key_val == target or idx_val == target or key_val.endswith(f"_{target}"):
                matched.append(r)
        if not matched:
            logger.error("Target ID '%s' not found in %d records.", target, total_loaded)
            sys.exit(1)
        records = matched
        logger.info("Filtered to single target ID '%s' (key=%s)", target, records[0].get("key"))

    # Slicing by start-idx and end-idx
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else total_loaded - 1

    if args.start_idx is not None or args.end_idx is not None:
        records = [r for r in records if start_idx <= r.get("_idx", 0) <= end_idx]
        logger.info("Applied index range [%d, %d]: %d records selected.", start_idx, end_idx, len(records))

    # Apply limit
    if args.limit is not None and args.limit > 0:
        records = records[:args.limit]
        logger.info("Applied --limit %d: %d records will be processed.", args.limit, len(records))

    # Filter out already processed if resume
    if args.resume:
        processed_keys = load_processed_keys(output_path)
        before_cnt = len(records)
        records = [r for r in records if r.get("key") not in processed_keys]
        logger.info("Resume mode: %d already completed, %d remaining to process.", before_cnt - len(records), len(records))

    if not records:
        logger.info("No records left to process. Exiting.")
        # Ensure pretty JSON is in sync
        json_path = output_path.replace(".jsonl", ".json")
        sync_json_from_jsonl(output_path, json_path)
        return

    # Prepare output directories
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    file_mode = "a" if args.resume else "w"

    success_cnt = 0
    fail_cnt = 0

    logger.info("Starting processing using model: %s | Inter-request delay: %.1fs", args.model, args.delay)
    logger.info("=" * 70)

    with open(output_path, file_mode, encoding="utf-8") as out_f:
        for i, rec in enumerate(records):
            key = rec.get("key", f"idx_{rec.get('_idx')}")
            base_inst = rec.get("base_instruction", rec.get("instruction", ""))
            hard_constraints = rec.get("constraints_description", [])
            ref_resp = rec.get("reference_response", rec.get("response", rec.get("output", "")))

            logger.info("[%d/%d] Processing key: %s (idx: %s)", i + 1, len(records), key, rec.get("_idx"))
            logger.info("  Base Instruction: %s...", base_inst[:70].replace("\n", " "))
            if ref_resp:
                logger.info("  Ref Gold Response: %s...", ref_resp[:70].replace("\n", " "))

            rec_split = rec.get("constraint_split", "seen") if args.split == "auto" else args.split
            meta_prompt = build_soft_constraint_meta_prompt(
                base_instruction=base_inst,
                hard_constraints=hard_constraints,
                reference_response=ref_resp,
                split=rec_split,
            )
            raw_resp = call_gemini(
                api_key=api_key,
                prompt=meta_prompt,
                model=args.model,
                max_retries=args.max_retries,
            )

            if not raw_resp:
                logger.error("❌ Failed to get response for key: %s", key)
                fail_cnt += 1
                continue

            soft_constraints = parse_soft_constraints(raw_resp)
            if not soft_constraints:
                logger.error("❌ Failed to parse valid soft constraints for key: %s", key)
                fail_cnt += 1
                continue

            # Build Task 2 final record
            task2_record = assemble_task2_record(rec, soft_constraints, model_judge=args.model_judge)

            # Write line immediately
            out_f.write(json.dumps(task2_record, ensure_ascii=False) + "\n")
            out_f.flush()
            success_cnt += 1

            logger.info("  ✓ Generated %d soft constraints for %s", len(soft_constraints), key)
            for sc in soft_constraints:
                logger.info("    * [%s] %s", sc["id"], sc["description"][:60])

            # Sleep between requests to avoid hitting rate limits
            if i < len(records) - 1:
                time.sleep(args.delay)

    # Sync to JSON
    json_path = output_path.replace(".jsonl", ".json")
    sync_json_from_jsonl(output_path, json_path)

    logger.info("=" * 70)
    logger.info("ALL DONE! Succeeded: %d | Failed: %d | Output saved to:", success_cnt, fail_cnt)
    logger.info("  JSONL: %s", output_path)
    logger.info("  JSON:  %s", json_path)


if __name__ == "__main__":
    main()
