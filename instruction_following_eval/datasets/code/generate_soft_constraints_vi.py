#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 2: generate_soft_constraints_vi.py
Mục đích: Sinh tập Soft Constraints Tiếng Việt (Task 2 Hybrid) theo từng Batch 16 mẫu.
          Sử dụng kiến trúc Dual-Engine:
          - Engine 1 (Sinh soft): gemma-4-31B-it (qua BASE_URL & MODEL trong .env)
          - Engine 2 (Cross-verify): gemini-3.5-flash-lite (qua GOOGLE_API_KEY)

Tính năng:
- Nhận input từ kết quả của Module 1 (results/module_hard_constraints/).
- Reverse-engineer 1 đến 2 soft constraints tiếng Việt từ reference response.
- Phân tách In-Domain (5 Seen Categories) và Out-of-Domain (7 Unseen Categories).
- Kiểm tra chéo (Cross-verification) chất lượng rubric bằng Gemini 3.5 Flash Lite.
- Giữ nguyên số lượng mẫu trong batch: 1 batch = đúng 16 mẫu.
- Lưu kết quả vào: 'results/module_soft_constraints/module2_soft_batch_{id}_{timestamp}.json'.
- Hỗ trợ các cờ: --id, --start-idx, --end-idx, --resume, --batch-size 16, --split (seen/unseen).
"""

import os
import sys
import json
import time
import argparse
import datetime
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from soft_constraints_policy import normalize_soft_constraints, soft_reward_spec

load_dotenv()

DEFAULT_BASE_URL = os.getenv("BASE_URL", "").strip().rstrip("/")
DEFAULT_MODEL = os.getenv("MODEL", "google/gemma-4-31B-it").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

if DEFAULT_BASE_URL:
    if DEFAULT_BASE_URL.endswith("/models"):
        DEFAULT_BASE_URL = DEFAULT_BASE_URL[:-7].rstrip("/")
    if DEFAULT_BASE_URL.endswith("/docs"):
        DEFAULT_BASE_URL = DEFAULT_BASE_URL[:-5].rstrip("/")
    if not DEFAULT_BASE_URL.endswith("/v1") and "googleapis" not in DEFAULT_BASE_URL:
        DEFAULT_BASE_URL = f"{DEFAULT_BASE_URL}/v1"


def get_openai_client(base_url: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        timeout=60.0
    )


# ==============================================================================
# 1. CALL GEMINI API (Cho Engine 2: Cross-Verification)
# ==============================================================================

def call_gemini_verify(
    api_key: str,
    prompt: str,
    model: str = "gemini-3.5-flash-lite",
    max_retries: int = 5,
    initial_delay: float = 2.0
) -> Optional[str]:
    """Gọi Gemini API qua HTTP POST thô để cross-verify soft constraints."""
    if not api_key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
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
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                text = resp_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return text
        except urllib.error.HTTPError as e:
            if e.code == 429:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)
            elif attempt == max_retries:
                return None
        except Exception:
            if attempt == max_retries:
                return None
            time.sleep(initial_delay)
    return None


# ==============================================================================
# 2. PROMPT BUILDER FOR SOFT CONSTRAINTS (Tiếng Việt)
# ==============================================================================

def build_soft_meta_prompt_vi(
    base_instruction: str,
    hard_constraints_desc: List[str],
    reference_response: str,
    split: str = "seen"
) -> str:
    hard_list_str = "\n".join([f"- {c}" for c in hard_constraints_desc]) if hard_constraints_desc else "- Không có"
    ref_snippet = reference_response.strip()[:2000] if reference_response else "Không có câu trả lời mẫu."

    if split == "unseen":
        category_guide = """BẠN PHẢI CHỌN 1 ĐẾN 2 RÀNG BUỘC THUỘC 7 NHÓM OUT-OF-DOMAIN (UNSEEN) SAU:
1. 'target_audience': Điều chỉnh độ sâu, từ vựng phù hợp với đối tượng độc giả cụ thể (học sinh tiểu học, người mới bắt đầu, chuyên gia nghiên cứu, giám đốc điều hành).
2. 'reasoning_and_logic': Yêu cầu suy luận nhiều bước rõ ràng, giải thích cơ chế nguyên nhân - kết quả hoặc phân tích so sánh lý do đằng sau các đề xuất.
3. 'role_play_persona': Đóng vai một chuyên gia hoặc nhân vật cụ thể trong toàn bộ bài viết (ví dụ: thanh tra an toàn cao cấp, nhà báo điều tra, bác sĩ lâm sàng).
4. 'counterfactual_context': Tuân thủ nghiêm ngặt một giả định phản thực tế hoặc kịch bản tưởng tượng (ví dụ: 'giả sử điện năng chưa bao giờ được phát minh').
5. 'safety_and_neutrality': Trình bày góc nhìn đa chiều, khách quan, không thiên vị về một vấn đề còn tranh cãi.
6. 'pedagogical_analogy': Bắt buộc sử dụng một phép ẩn dụ hoặc ví dụ đời thường trực quan để làm sáng tỏ khái niệm phức tạp.
7. 'critique_and_limitations': Chỉ rõ những rủi ro tiềm ẩn, điều kiện biên hoặc các tình huống ngoại lệ/thất bại của giải pháp."""
    else:
        category_guide = """BẠN PHẢI CHỌN 1 ĐẾN 2 RÀNG BUỘC THUỘC 5 NHÓM IN-DOMAIN (SEEN) SAU:
1. 'semantic_completeness': Yêu cầu bao phủ đầy đủ các trụ cột nội dung cốt lõi của chủ đề, không bỏ sót các ý quan trọng.
2. 'style_and_tone': Xác định giọng văn, phong cách giao tiếp chuẩn mực (ví dụ: giọng văn tư vấn chuyên nghiệp, học thuật, khuyến khích).
3. 'clarity_and_coherence': Yêu cầu mạch lạc trong luận điểm, sự chuyển tiếp rõ ràng giữa các ý.
4. 'conciseness_and_efficiency': Yêu cầu súc tích, hàm lượng thông tin cao, không dùng từ sáo rỗng hoặc câu mở đầu thừa thãi.
5. 'practical_examples': Yêu cầu cung cấp ví dụ thực tiễn hoặc trường hợp áp dụng cụ thể."""

    prompt = f"""Bạn là một chuyên gia nghiên cứu AI thiết kế dữ liệu huấn luyện RLVR cho tác vụ Instruction Following tiếng Việt.
Dưới đây là một yêu cầu bài toán kèm theo câu trả lời chuẩn (Gold Reference) và các ràng buộc cứng hiện có:

Yêu cầu gốc:
{base_instruction}

Câu trả lời mẫu chất lượng cao:
\"\"\"
{ref_snippet}
\"\"\"

Các ràng buộc cứng (Hard Constraints) đã có:
{hard_list_str}

{category_guide}

YÊU CẦU ĐẦU RA:
Hãy phân tích câu trả lời chuẩn và 'sinh ngược' (reverse-engineer) ra 1 đến 2 Soft Constraints tiếng Việt để thử thách mô hình sinh ra câu trả lời có chất lượng tương xứng.
Trả về DUY NHẤT một mảng JSON (không bọc giải thích rườm rà) với cấu trúc chính xác:
[
  {{
    "id": "<tên_category>:<định_danh_ngắn_gọn>",
    "category": "<chính xác một trong các category ở trên>",
    "description": "Câu lệnh yêu cầu ngắn gọn rõ ràng bằng tiếng Việt cho học viên",
    "eval_rubric": "Tiêu chí chấm điểm nhị phân bằng tiếng Việt: Chấm 1 nếu đạt tiêu chí... Chấm 0 nếu không đạt..."
  }}
]

QUY TẮC SỐNG CÒN:
1. KHÔNG được xung đột với Hard Constraints hiện có.
2. KHÔNG chép nguyên văn đáp án mẫu làm lộ lời giải; chỉ tập trung vào nguyên tắc chất lượng, chiều sâu nội dung hoặc phong cách.
3. Toàn bộ description và eval_rubric phải viết bằng tiếng Việt tự nhiên chuẩn mực.
4. Description phải công bố TẤT CẢ điều kiện đạt trong rubric: số lượng tối thiểu, nội dung, vai diễn hoặc định dạng. Không thêm điều kiện ẩn chỉ judge mới biết.
5. Mỗi rubric phải xác định cả trường hợp đạt và không đạt, kể cả câu trả lời sát ngưỡng. Với nội dung, các ý phải đúng và liên quan, không chỉ nhắc từ khóa.
6. Phong cách/vai diễn phải có dấu hiệu quan sát được trong cách giải thích và giao tiếp; không thưởng chỉ vì tự xưng chuyên gia hoặc chèn thuật ngữ.
7. Không thay nhiệm vụ gốc (ví dụ: yêu cầu chỉ xuất 0/1 không được thêm lời giải dài). Không dùng hình thức bảng/đếm từ làm tiêu chí mềm nếu đã được kiểm tra bằng luật cứng."""

    return prompt


def verify_soft_constraint_with_gemini(
    api_key: str,
    base_instruction: str,
    hard_constraints_desc: List[str],
    soft_constraints: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Cross-verify soft constraints bằng Gemini 3.5 Flash Lite."""
    if not api_key:
        for sc in soft_constraints:
            sc["verification_status"] = "UNVERIFIED"
            sc["verified_by"] = "none"
        return soft_constraints

    prompt = f"""Bạn là Giám khảo AI kiểm định chất lượng dữ liệu huấn luyện.
Dưới đây là một nhiệm vụ tiếng Việt kèm các ràng buộc cứng và các Soft Constraints vừa được sinh:

Yêu cầu gốc:
{base_instruction}

Ràng buộc cứng:
{json.dumps(hard_constraints_desc, ensure_ascii=False)}

Các Soft Constraints cần thẩm định:
{json.dumps(soft_constraints, ensure_ascii=False, indent=2)}

Nhiệm vụ:
Kiểm tra từng soft constraint xem:
1. Có bị mâu thuẫn trực tiếp với các Ràng buộc cứng không?
2. Tiêu chí eval_rubric có rõ ràng để chấm điểm nhị phân (1 hoặc 0), bao gồm trường hợp sát ngưỡng không?
3. Description có công bố đầy đủ tất cả yêu cầu của rubric, không có điều kiện ẩn không?
4. Có làm thay đổi nhiệm vụ gốc, bắt buộc câu trả lời sai hoặc thưởng chỉ vì nhắc từ khóa/tự xưng chuyên gia không?
5. Trả về mảng JSON giữ nguyên id và thêm "verification_status": "PASS" hoặc "REJECT", cùng "verification_reason" giải thích ngắn. Chỉ PASS khi đạt tất cả kiểm tra."""

    gemini_resp = call_gemini_verify(api_key, prompt, model="gemini-3.5-flash-lite")
    if gemini_resp:
        try:
            clean_resp = gemini_resp.strip()
            if "```json" in clean_resp:
                clean_resp = clean_resp.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_resp:
                clean_resp = clean_resp.split("```")[1].split("```")[0].strip()
            verified_list = json.loads(clean_resp)
            if isinstance(verified_list, list) and len(verified_list) == len(soft_constraints):
                by_id = {v.get("id"): v for v in verified_list if isinstance(v, dict)}
                expected = {s["id"] for s in soft_constraints}
                if set(by_id) != expected or len(by_id) != len(verified_list):
                    raise ValueError("Verification IDs missing, duplicate or unknown")
                if any(v.get("verification_status") not in ("PASS", "REJECT") or
                       not isinstance(v.get("verification_reason"), str) or not v["verification_reason"].strip()
                       for v in by_id.values()):
                    raise ValueError("Verification status/reason missing")
                for item in soft_constraints:
                    verified = by_id[item["id"]]
                    item["verification_status"] = verified["verification_status"]
                    item["verification_reason"] = verified["verification_reason"]
                    item["verified_by"] = "gemini-3.5-flash-lite"
                return soft_constraints
        except Exception:
            pass

    for sc in soft_constraints:
        sc["verification_status"] = "UNVERIFIED"
        sc["verification_reason"] = "Verification API failed or returned invalid IDs/status/reason"
        sc["verified_by"] = "none"
    return soft_constraints


def main():
    parser = argparse.ArgumentParser(description="Module 2: Sinh Soft Constraints tiếng Việt theo Batch 16 mẫu.")
    parser.add_argument("--input-dir", type=str, default="results/module_hard_constraints",
                        help="Thư mục chứa các file batch từ Module 1 (hoặc file json/jsonl)")
    parser.add_argument("--output-dir", type=str, default="results/module_soft_constraints",
                        help="Thư mục lưu các file batch JSON kết quả Module 2")
    parser.add_argument("--split", type=str, default="seen", choices=["seen", "unseen"],
                        help="Tập ràng buộc: 'seen' (5 categories) hoặc 'unseen' (7 categories OOD)")
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
                        help="Tiếp tục từ mẫu/batch chưa xử lý")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help="Base URL của LLM sinh soft constraints")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Tên mô hình LLM sinh soft constraints")
    parser.add_argument("--save-unified", action="store_true",
                        help="Đồng thời lưu file gộp vào datasets/seen hoặc datasets/unseen")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Thời gian nghỉ (giây) giữa các request")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    from llm_client import get_llm_client_and_model
    client, resolved_model, resolved_url = get_llm_client_and_model(args.model, args.base_url)
    print("=" * 65)
    print(f"🌐 [Module 2 Soft Constraints] Khởi tạo LLM Engine 1: '{resolved_model}'")
    print(f"   • API Endpoint: {resolved_url}")
    print("=" * 65)

    # Thu thập tất cả các mẫu từ input-dir hoặc input file
    all_input_samples = []
    if os.path.isdir(args.input_dir):
        batch_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".json") and f.startswith("module1_hard_batch_")])
        for bf in batch_files:
            bpath = os.path.join(args.input_dir, bf)
            try:
                with open(bpath, "r", encoding="utf-8") as f_in:
                    data = json.load(f_in)
                    if isinstance(data, list):
                        all_input_samples.extend(data)
            except Exception as e:
                print(f"⚠️ Lỗi đọc file batch {bf}: {e}")
    elif os.path.isfile(args.input_dir):
        if args.input_dir.endswith(".jsonl"):
            with open(args.input_dir, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if line.strip():
                        all_input_samples.append(json.loads(line))
        else:
            with open(args.input_dir, "r", encoding="utf-8") as f_in:
                all_input_samples = json.load(f_in)
    # Tự động lọc trùng lặp theo key
    seen_keys = set()
    deduped_samples = []
    for s in all_input_samples:
        k = s.get("key") or str(s.get("id"))
        if k not in seen_keys:
            seen_keys.add(k)
            deduped_samples.append(s)
    all_input_samples = deduped_samples

    print(f"📖 Đã nạp {len(all_input_samples)} mẫu đầu vào duy nhất từ '{args.input_dir}'.")

    # Lọc theo --id
    if args.id:
        target_id = str(args.id).strip()
        all_input_samples = [
            s for s in all_input_samples 
            if str(s.get("id")) == target_id or s.get("key") == target_id
        ]
        print(f"🎯 Lọc theo --id '{args.id}': tìm thấy {len(all_input_samples)} mẫu.")

    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(all_input_samples)
    if args.num_samples is not None:
        end_idx = min(end_idx, start_idx + args.num_samples)

    selected_samples = all_input_samples[start_idx:end_idx]
    print(f"⚙️ Bắt đầu sinh Soft Constraints cho {len(selected_samples)} mẫu (Split: {args.split}).")

    # Kiểm tra các key đã xử lý nếu --resume
    processed_keys = set()
    if args.resume:
        for fname in os.listdir(args.output_dir):
            if fname.endswith(".json") and fname.startswith("module2_soft_batch_"):
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
        print(f"🔄 [RESUME] Đã phát hiện {len(processed_keys)} mẫu đã sinh soft constraints.")

    batch_size = args.batch_size
    current_batch = []
    batch_idx = 1
    total_saved = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    unified_file = f"datasets/{args.split}/task2_hybrid_vi.jsonl"
    if args.save_unified:
        os.makedirs(os.path.dirname(unified_file), exist_ok=True)

    for s_idx, item in enumerate(selected_samples, 1):
        key = item.get("key", f"sample_{item.get('id', 0)}")
        if args.resume and key in processed_keys:
            continue

        base_inst = item.get("base_instruction", "")
        ref_resp = item.get("reference_response", "")
        hard_desc = item.get("hard_constraints", {}).get("constraints_description", [])

        print(f"🔄 [{s_idx}/{len(selected_samples)}] Sinh soft constraint cho {key}...")

        # 1. Gọi Engine 1 để sinh Soft Constraints (kèm retry tự động nếu gặp 429)
        meta_prompt = build_soft_meta_prompt_vi(base_inst, hard_desc, ref_resp, split=args.split)
        soft_constraints_generated = []

        for attempt in range(1, 4):
            try:
                resp = client.chat.completions.create(
                    model=resolved_model,
                    messages=[
                        {"role": "system", "content": "Bạn là chuyên gia thiết kế Soft Constraints tiếng Việt. Trả về duy nhất JSON array."},
                        {"role": "user", "content": meta_prompt}
                    ],
                    temperature=0.4,
                    max_tokens=2048,
                )
                raw_text = resp.choices[0].message.content or ""
                raw_text = raw_text.strip()
                clean_json = raw_text
                if "```json" in clean_json:
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json:
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()

                parsed_soft = json.loads(clean_json)
                if not isinstance(parsed_soft, list) or not 1 <= len(parsed_soft) <= 2:
                    raise ValueError("Expected 1–2 soft constraints")
                # Model-generated PASS is not evidence of external verification.
                for sc in parsed_soft:
                    sc["verification_status"] = "UNVERIFIED"
                normalize_soft_constraints(parsed_soft, require_verified=False, split=args.split)
                soft_constraints_generated = parsed_soft
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    wait_sec = 25.0 * attempt
                    print(f"⏳ [429 Quota Exceeded] Chờ {wait_sec}s rồi thử lại cho {key} (lần {attempt}/3)...")
                    time.sleep(wait_sec)
                else:
                    print(f"⚠️ [Soft Gen Error] Lỗi sinh soft constraint cho {key}: {e}")
                    break

        # Fallback nếu model không sinh được
        if not soft_constraints_generated:
            soft_constraints_generated = [{
                "id": "semantic_completeness:co_ban",
                "category": "semantic_completeness",
                "description": "Câu trả lời phải giải quyết trọn vẹn và chuẩn xác yêu cầu trọng tâm của đề bài.",
                "eval_rubric": "Chấm 1 nếu phản hồi trả lời đúng trọng tâm và đầy đủ ý. Chấm 0 nếu lan man hoặc thiếu ý chính.",
                "verification_status": "UNVERIFIED",
                "verified_by": "fallback"
            }]

        # 2. Kiểm định riêng, kể cả khi engine sinh cũng là Gemini.
        # Không có key / lỗi kiểm định: giữ UNVERIFIED và không đóng gói để train.
        if GOOGLE_API_KEY:
            soft_constraints_generated = verify_soft_constraint_with_gemini(
                GOOGLE_API_KEY,
                base_inst,
                hard_desc,
                soft_constraints_generated
            )
        else:
            for sc in soft_constraints_generated:
                if "verification_status" not in sc:
                    sc["verification_status"] = "UNVERIFIED"
                sc["verified_by"] = "none"

        # Cập nhật prompt mở rộng 2 tầng (Hard + Soft)
        accepted_soft, excluded_soft = normalize_soft_constraints(soft_constraints_generated, split=args.split)
        soft_bullets = "\n".join([f"- {sc['description']}" for sc in accepted_soft])
        hard_prompt = item.get("prompt", "")
        hybrid_prompt = f"{hard_prompt}\n\n[Hướng dẫn Nội dung & Ngữ nghĩa]\n{soft_bullets}"

        record = dict(item)
        record["hybrid_prompt"] = hybrid_prompt
        record["soft_constraints"] = {
            "verifier_type": "llm_reasoning_judge",
            "model_judge": f"dual_engine ({resolved_model} & gemini-3.5-flash-lite)",
            "constraints": soft_constraints_generated
        }
        record["num_constraints"] = {
            "total": len(hard_desc) + len(accepted_soft),
            "hard_count": len(hard_desc),
            "soft_count": len(accepted_soft)
        }
        record["soft_quality"] = {"accepted_count": len(accepted_soft), "excluded": excluded_soft,
                                  "ready_for_hybrid_training": bool(accepted_soft)}
        record["reward_spec"] = soft_reward_spec()
        record["batch_id"] = batch_idx
        record["batch_sample_idx"] = len(current_batch) + 1
        record["updated_at"] = datetime.datetime.now().isoformat()

        current_batch.append(record)

        if len(current_batch) >= batch_size:
            batch_filename = f"module2_soft_batch_{batch_idx:03d}_{timestamp}.json"
            batch_filepath = os.path.join(args.output_dir, batch_filename)
            with open(batch_filepath, "w", encoding="utf-8") as f_b:
                json.dump(current_batch, f_b, ensure_ascii=False, indent=2)

            if args.save_unified:
                with open(unified_file, "a", encoding="utf-8") as f_uni:
                    for rec in current_batch:
                        f_uni.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"📦 [Module 2 - Batch {batch_idx:03d}] Đã lưu {len(current_batch)} mẫu vào '{batch_filepath}'.")
            total_saved += len(current_batch)
            current_batch = []
            batch_idx += 1

        if args.delay > 0:
            time.sleep(args.delay)

    if current_batch:
        batch_filename = f"module2_soft_batch_{batch_idx:03d}_{timestamp}.json"
        batch_filepath = os.path.join(args.output_dir, batch_filename)
        with open(batch_filepath, "w", encoding="utf-8") as f_b:
            json.dump(current_batch, f_b, ensure_ascii=False, indent=2)

        if args.save_unified:
            with open(unified_file, "a", encoding="utf-8") as f_uni:
                for rec in current_batch:
                    f_uni.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"📦 [Module 2 - Batch {batch_idx:03d} (Cuối)] Đã lưu {len(current_batch)} mẫu vào '{batch_filepath}'.")
        total_saved += len(current_batch)

    print("\n" + "=" * 60)
    print(f"🎉 HOÀN THÀNH MODULE 2 (Soft Constraints Tiếng Việt - Dual Engine):")
    print(f"   • Tổng số mẫu đã lưu : {total_saved} mẫu")
    print(f"   • Số batch hoàn thành: {batch_idx}")
    print(f"   • Thư mục kết quả    : {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
