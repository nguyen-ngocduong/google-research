#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Công cụ kiểm tra tính khả thi thực nghiệm qua LLM Rollouts (Execution-Guided Feasibility Filter):
Mục đích: Phát hiện các tổ hợp ràng buộc "bất khả thi" hoặc quá khắt khe mà static conflict groups không bắt được.
Phương pháp:
1. Cho mô hình LLM mạnh (google/gemma-4-31B-it) sinh N rollouts (mặc định N=3, temp=0.7) cho mỗi prompt.
2. Đánh giá từng rollout qua verifier_engine_vi.py.
3. Nếu cả N rollouts đều Pass@N = 0 (không rollout nào thỏa mãn 100% hard constraints), prompt được gắn cờ xem xét/loại bỏ.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from verifier_engine_vi import evaluate_instruction_following_vi

load_dotenv()


def get_client_and_model(args) -> tuple:
    base_url = args.base_url or os.getenv("BASE_URL", "").strip().rstrip("/")
    model = args.model or os.getenv("MODEL", "google/gemma-4-31B-it").strip()
    if base_url:
        if base_url.endswith("/models"): base_url = base_url[:-7].rstrip("/")
        if not base_url.endswith("/v1") and "googleapis" not in base_url:
            base_url = f"{base_url}/v1"
    client = OpenAI(
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        timeout=60.0
    )
    return client, model


def test_prompt_feasibility(client, model: str, prompt: str, verifier_spec: str, n_rollouts: int = 3) -> Dict[str, Any]:
    """Sinh n_rollouts và kiểm tra verifier."""
    rollout_results = []
    any_strict_pass = False
    best_partial = 0.0

    for _ in range(n_rollouts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )
            ans = resp.choices[0].message.content or ""
            eval_res = evaluate_instruction_following_vi(ans, verifier_spec)
            is_strict = eval_res.get("strict_score", 0.0) == 1.0
            partial = eval_res.get("partial_score", 0.0)
            best_partial = max(best_partial, partial)
            if is_strict:
                any_strict_pass = True
            rollout_results.append({
                "strict": is_strict,
                "partial": partial,
                "passed_count": eval_res.get("passed_count", 0),
                "total_hard": eval_res.get("total_hard", 0)
            })
        except Exception as e:
            rollout_results.append({"error": str(e), "strict": False, "partial": 0.0})

    return {
        "pass_at_n": any_strict_pass,
        "best_partial": best_partial,
        "rollout_details": rollout_results
    }


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra tính khả thi thực nghiệm bằng Rollouts.")
    parser.add_argument("--input", type=str, default="datasets/eval/test_unseen.jsonl",
                        help="Đường dẫn file dataset cần kiểm thử")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Số lượng mẫu kiểm thử (mặc định: 10 mẫu mẫu thử)")
    parser.add_argument("--num-rollouts", type=int, default=3,
                        help="Số lượng rollout mỗi prompt (Pass@N)")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--filter-output", type=str, default=None,
                        help="Đường dẫn lưu file jsonl chỉ chứa các prompt đạt Pass@N > 0")
    args = parser.parse_args()

    client, model = get_client_and_model(args)
    print("=" * 65)
    print(f"🔍 BẮT ĐẦU KIỂM ĐỊNH TÍNH KHẢ THI (ROLLOUT-BASED FEASIBILITY VERIFIER)")
    print(f"• Model: {model}")
    print(f"• File test: {args.input}")
    print(f"• Số rollout / prompt: {args.num_rollouts}")
    print("=" * 65)

    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    test_samples = samples[:args.num_samples]
    feasible_count = 0
    passed_samples = []

    for idx, s in enumerate(test_samples, 1):
        key = s.get("key", f"sample_{idx}")
        prompt = s.get("prompt", "")
        verifier = s.get("verifier", "[]")
        res = test_prompt_feasibility(client, model, prompt, verifier, n_rollouts=args.num_rollouts)
        status = "✅ KHẢ THI (Pass@N)" if res["pass_at_n"] else f"⚠️ Khó/Bế tắc (Best partial: {res['best_partial']:.0%})"
        print(f"[{idx}/{len(test_samples)}] {key}: {status}")
        if res["pass_at_n"] or res["best_partial"] >= 0.6:
            feasible_count += 1
            passed_samples.append(s)

    print("=" * 65)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ TÍNH KHẢ THI:")
    print(f"• Tỷ lệ Prompt khả thi (Pass@N > 0 hoặc Partial >= 60%): {feasible_count}/{len(test_samples)} ({feasible_count/len(test_samples):.1%})")
    print("=" * 65)


if __name__ == "__main__":
    main()
