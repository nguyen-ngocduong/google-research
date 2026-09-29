#!/usr/bin/env python3
# coding=utf-8
"""Script to generate responses for IFEval using an OpenAI-compatible vLLM server.

Features:
- Reads prompts from input_data.jsonl.
- Sends requests sequentially (concurrency=1) or concurrently in batches to vLLM endpoint.
- Uses temperature=0.0.
- Supports running a single key (--key).
- Supports running a range of keys (--start-key, --end-key).
- Supports resuming interrupted runs (--resume, enabled by default).
- Appends results immediately with flush to ensure safety against interruptions.
- Reads default BASE_URL and MODEL from .env if present.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_env_file(env_path: Optional[str] = None) -> Dict[str, str]:
    """Parse key-value pairs from .env file without external dependencies.
    Looks in current directory, script directory, or specified path.
    """
    candidates = []
    if env_path:
        candidates.append(env_path)
    # Check cwd
    candidates.append(os.path.join(os.getcwd(), ".env"))
    # Check script's own folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, ".env"))

    resolved_path = None
    for path in candidates:
        if os.path.isfile(path):
            resolved_path = path
            break

    env_vars = {}
    if not resolved_path:
        return env_vars

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                env_vars[key] = val
    except Exception as e:
        logging.warning("Could not read .env file at %s: %s", resolved_path, e)
    return env_vars


def normalize_v1_base_url(url: str) -> str:
    """Normalize base URL so that it points to the v1 root (e.g.

    http://host:port/v1).
    Handles cases like:
      - https://host.com/docs -> https://host.com/v1
      - https://host.com/docs/ -> https://host.com/v1
      - http://host:8000 -> http://host:8000/v1
      - http://host:8000/v1/ -> http://host:8000/v1
    """
    url = url.strip().rstrip("/")
    if url.endswith("/docs"):
        url = url[:-5].rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def load_input_data(input_path: str) -> List[Dict[str, Any]]:
    """Load JSONL input data."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                items.append(data)
            except json.JSONDecodeError as e:
                logging.warning("Failed to parse JSON at line %d in %s: %s", line_num, input_path, e)
    return items


def load_completed_keys(output_path: str) -> Set[str]:
    """Read output file and return the set of keys that already have responses."""
    completed = set()
    if not os.path.exists(output_path):
        return completed
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "key" in data and data.get("response") is not None:
                        completed.add(str(data["key"]))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logging.warning("Error reading existing output file %s: %s", output_path, e)
    return completed


def filter_items_by_keys(
    items: List[Dict[str, Any]],
    single_key: Optional[str] = None,
    start_key: Optional[str] = None,
    end_key: Optional[str] = None,
    range_mode: str = "file_order",
) -> List[Dict[str, Any]]:
    """Filter input items based on single_key, start_key, and end_key."""
    if single_key is not None:
        target = str(single_key)
        matched = [item for item in items if str(item.get("key")) == target]
        if not matched:
            raise ValueError(f"Key '{single_key}' was not found in the input dataset.")
        return matched

    if start_key is None and end_key is None:
        return items

    if range_mode == "numeric":
        s_val = float(start_key) if start_key is not None else float("-inf")
        e_val = float(end_key) if end_key is not None else float("inf")
        if s_val > e_val:
            s_val, e_val = e_val, s_val
        matched = []
        for item in items:
            try:
                k_val = float(item.get("key"))
                if s_val <= k_val <= e_val:
                    matched.append(item)
            except (ValueError, TypeError):
                continue
        return matched
    else:  # file_order
        str_keys = [str(item.get("key")) for item in items]
        start_idx = 0
        end_idx = len(items) - 1

        if start_key is not None:
            s_str = str(start_key)
            if s_str in str_keys:
                start_idx = str_keys.index(s_str)
            else:
                raise ValueError(f"Start key '{start_key}' was not found in the input dataset.")

        if end_key is not None:
            e_str = str(end_key)
            if e_str in str_keys:
                end_idx = str_keys.index(e_str)
            else:
                raise ValueError(f"End key '{end_key}' was not found in the input dataset.")

        if start_idx > end_idx:
            raise ValueError(
                f"Start key '{start_key}' (index {start_idx}) appears after end key '{end_key}' (index {end_idx}) in the dataset."
            )

        return items[start_idx : end_idx + 1]


def call_vllm_api(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    mode: str = "chat",
    api_key: Optional[str] = None,
    timeout: int = 120,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Send request to vLLM server with retries and return generated text."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if mode == "chat":
        endpoint = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    else:
        endpoint = f"{base_url}/completions"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    encoded_payload = json.dumps(payload).encode("utf-8")

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                endpoint,
                data=encoded_payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                resp_bytes = response.read()
                resp_json = json.loads(resp_bytes.decode("utf-8"))

                if mode == "chat":
                    choices = resp_json.get("choices", [])
                    if not choices:
                        raise ValueError(f"No choices returned in vLLM response: {resp_json}")
                    content = choices[0].get("message", {}).get("content", "")
                    return content or ""
                else:
                    choices = resp_json.get("choices", [])
                    if not choices:
                        raise ValueError(f"No choices returned in vLLM response: {resp_json}")
                    return choices[0].get("text", "") or ""

        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_error = e
            err_msg = str(e)
            if isinstance(e, urllib.error.HTTPError):
                try:
                    err_msg += f" - Response: {e.read().decode('utf-8')[:300]}"
                except Exception:
                    pass
            logging.warning(
                "Attempt %d/%d failed for endpoint %s: %s",
                attempt,
                max_retries,
                endpoint,
                err_msg,
            )
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)

    raise RuntimeError(f"Failed to get response after {max_retries} attempts: {last_error}")


def main():
    # Load defaults from .env and environment variables
    env_vars = load_env_file()
    default_base_url = os.environ.get("BASE_URL") or env_vars.get("BASE_URL", "http://localhost:8000/v1")
    default_model = os.environ.get("MODEL") or env_vars.get("MODEL", "google/gemma-4-31B-it")

    parser = argparse.ArgumentParser(
        description="Generate LLM responses for IFEval dataset using vLLM server."
    )
    parser.add_argument(
        "--input-data",
        "--input",
        "-i",
        default="data/input_data.jsonl",
        help="Path to input_data.jsonl (default: data/input_data.jsonl)",
    )
    parser.add_argument(
        "--output-data",
        "--output",
        "-o",
        default="data/output_data.jsonl",
        help="Path to output_data.jsonl (default: data/output_data.jsonl)",
    )
    parser.add_argument(
        "--base-url",
        default=default_base_url,
        help=f"Base URL of vLLM server (default: {default_base_url})",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"Model name (default: {default_model})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum completion tokens (default: 2048)",
    )
    parser.add_argument(
        "--key",
        "-k",
        default=None,
        help="Process only a specific key (e.g. --key 1000)",
    )
    parser.add_argument(
        "--start-key",
        default=None,
        help="Start processing from this key (inclusive)",
    )
    parser.add_argument(
        "--end-key",
        default=None,
        help="Stop processing at this key (inclusive)",
    )
    parser.add_argument(
        "--range-mode",
        choices=["file_order", "numeric"],
        default="file_order",
        help="Range mode for start-key and end-key: 'file_order' (default) or 'numeric'",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1 for sequential)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from previous run by skipping already completed keys (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not skip completed keys",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite output file instead of resuming or appending",
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "completion"],
        default="chat",
        help="API mode: 'chat' (/v1/chat/completions) or 'completion' (/v1/completions)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", None),
        help="Optional API key for Authorization header",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for failed requests (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected keys and exit without sending requests",
    )

    args = parser.parse_args()

    # Normalize Base URL
    v1_base_url = normalize_v1_base_url(args.base_url)
    logging.info("Target vLLM URL: %s", v1_base_url)
    logging.info("Model: %s", args.model)
    logging.info("Temperature: %.2f", args.temperature)

    # 1. Load input dataset
    all_items = load_input_data(args.input_data)
    logging.info("Loaded %d items from %s", len(all_items), args.input_data)

    # 2. Filter items according to key flags
    target_items = filter_items_by_keys(
        all_items,
        single_key=args.key,
        start_key=args.start_key,
        end_key=args.end_key,
        range_mode=args.range_mode,
    )
    logging.info("Target items to process after filtering: %d", len(target_items))

    # 3. Handle overwrite or resume
    if args.overwrite and os.path.exists(args.output_data):
        logging.info("Flag --overwrite set. Removing existing output file: %s", args.output_data)
        os.remove(args.output_data)

    completed_keys = set()
    if args.resume and os.path.exists(args.output_data):
        completed_keys = load_completed_keys(args.output_data)
        logging.info("Found %d completed keys in %s", len(completed_keys), args.output_data)

    items_to_run = [
        item for item in target_items if str(item.get("key")) not in completed_keys
    ]
    logging.info(
        "Items remaining to run: %d (Skipped %d already completed)",
        len(items_to_run),
        len(target_items) - len(items_to_run),
    )

    if args.dry_run:
        logging.info("[DRY-RUN] Items that would be processed:")
        for item in items_to_run:
            print(f"Key: {item.get('key')} | Prompt: {item.get('prompt', '')[:60]}...")
        return

    if not items_to_run:
        logging.info("All selected items are already completed. Nothing to do!")
        return

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output_data))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_lock = threading.Lock()
    total = len(items_to_run)
    success_count = 0
    failure_count = 0

    def process_item(item: Dict[str, Any], index: int) -> Tuple[bool, Any]:
        key = item.get("key")
        prompt = item.get("prompt", "")
        logging.info("[%d/%d] Sending request for key: %s ...", index + 1, total, key)
        start_t = time.time()
        try:
            response_text = call_vllm_api(
                base_url=v1_base_url,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                mode=args.mode,
                api_key=args.api_key,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            elapsed = time.time() - start_t
            logging.info("[%d/%d] Completed key: %s (took %.2fs)", index + 1, total, key, elapsed)

            # Build output record: keep original fields and add 'response'
            record = dict(item)
            record["response"] = response_text

            # Thread-safe write and flush
            line_json = json.dumps(record, ensure_ascii=False) + "\n"
            with file_lock:
                with open(args.output_data, "a", encoding="utf-8") as out_f:
                    out_f.write(line_json)
                    out_f.flush()

            return True, key
        except Exception as e:
            elapsed = time.time() - start_t
            logging.error(
                "[%d/%d] Error processing key %s (after %.2fs): %s",
                index + 1,
                total,
                key,
                elapsed,
                e,
            )
            return False, key

    # Run sequentially or concurrently
    if args.concurrency <= 1:
        logging.info("Running sequentially (concurrency=1)...")
        for i, item in enumerate(items_to_run):
            success, _ = process_item(item, i)
            if success:
                success_count += 1
            else:
                failure_count += 1
    else:
        logging.info("Running concurrently with %d workers...", args.concurrency)
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(process_item, item, i): item.get("key")
                for i, item in enumerate(items_to_run)
            }
            for future in as_completed(futures):
                success, _ = future.result()
                if success:
                    success_count += 1
                else:
                    failure_count += 1

    logging.info(
        "Execution finished. Total processed: %d | Success: %d | Failed: %d",
        total,
        success_count,
        failure_count,
    )
    logging.info("Output saved to: %s", os.path.abspath(args.output_data))


if __name__ == "__main__":
    main()
