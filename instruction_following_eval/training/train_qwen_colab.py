#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training/train_qwen_colab.py
Hướng dẫn & Mã nguồn Huấn luyện Zero-SFT GRPO (RLVR Direct-to-GRPO) 
cho Qwen3.5-0.8B trên Google Colab Free T4 (15.3GB VRAM) hoặc Kaggle.

Nguyên lý Zero-SFT RLVR:
- Bỏ qua hoàn toàn SFT thủ công. Sử dụng thẳng checkpoint post-trained đã có khả năng nói tiếng Việt.
- Sử dụng trl.GRPOTrainer: Model sinh G=4 completions cho mỗi prompt.
- Reward Function sử dụng verifier_engine_vi.py để chấm điểm tất định:
    + Continuous Partial Hard Reward (tránh gradient chết khi G=4).
    + Thưởng bonus khi vượt qua 100% hard constraints.
    + Gated Soft Reward (chỉ chấm ngữ nghĩa khi đã đạt 100% luật cứng).
- Cột 'verifier' và 'soft' trong dataset được lưu dưới dạng chuỗi JSON để tương thích 100% với PyArrow.

LLM Judge Soft Constraints hỗ trợ 2 engine:
  1. Gemini (gemini-2.5-flash) — qua Google AI Studio API.
  2. Gemma-4-31B-it — qua vLLM/Cloudflare tunnel (OpenAI-compatible).
"""

import os
import sys
import re
import json
from typing import Any, Optional, Dict, List, Tuple

# Thêm đường dẫn tới module verifier & llm_client
_CODE_PATHS = ["datasets/code", "/kaggle/working/datasets/code", "/content/datasets/code"]
for _p in _CODE_PATHS:
    if _p not in sys.path:
        sys.path.append(_p)

try:
    from verifier_engine_vi import evaluate_instruction_following_vi
    print("✅ Đã nạp Verifier Engine từ verifier_engine_vi.py")
except ImportError:
    raise ImportError(
        "❌ Không tìm thấy 'verifier_engine_vi.py'. "
        "Vui lòng đặt file này trong datasets/code/ hoặc /kaggle/working/datasets/code/"
    )

try:
    from llm_client import get_llm_client_and_model
except ImportError:
    # Fallback độc lập khi chạy trên Kaggle / Colab không có llm_client.py
    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️ Chưa cài openai. Chạy: pip install openai")
        OpenAI = None

    def get_llm_client_and_model(model_name=None, base_url=None, timeout=60.0):
        if OpenAI is None:
            raise RuntimeError("Cần cài 'pip install openai' để sử dụng LLM Judge.")
        req_model = model_name or os.getenv("MODEL", "gemini-2.5-flash")
        m_lower = req_model.lower()
        if "gemini" in m_lower:
            endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/"
            api_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
            norm_model = "gemini-2.5-flash" if "2.5" in m_lower else req_model
        elif "gemma" in m_lower:
            env_url = base_url or os.getenv("GEMMA_BASE_URL", "") or os.getenv("BASE_URL", "")
            if env_url and ("trycloudflare" in env_url or "ngrok" in env_url or "googleapis" not in env_url):
                # Gemma qua vLLM / Cloudflare tunnel / ngrok
                endpoint = env_url.rstrip("/")
                if not endpoint.endswith("/v1"):
                    endpoint = endpoint + "/v1"
                api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
                norm_model = "google/gemma-4-31B-it" if "31b" in m_lower else req_model
            else:
                # Gemma qua Google AI Studio
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/"
                api_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
                norm_model = "gemma-4-31b-it"
        else:
            endpoint = base_url or os.getenv("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
            api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
            norm_model = req_model
        client = OpenAI(base_url=endpoint, api_key=api_key or "EMPTY", timeout=timeout)
        return client, norm_model, endpoint

# ==============================================================================
# 1. CẤU HÌNH MÔ HÌNH VÀ LORA (TỐI ƯU CHO COLAB / KAGGLE GPU 16GB)
# ==============================================================================
MODEL_NAME = "Qwen/Qwen3.5-0.8B"

LORA_CONFIG_PARAMS = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Cấu hình Reward toàn cục
GLOBAL_REWARD_CONFIG = {
    "mode": "hard_only",               # 'hard_only' (Exp 2), 'hybrid_gated' (Exp 3)
    "judge_model": "gemini-2.5-flash",  # 'gemini-2.5-flash' hoặc 'google/gemma-4-31B-it'
    "judge_base_url": None,            # Base URL khi chạy gemma qua Cloudflare tunnel / vLLM
    "w_hard": 0.75,
    "w_soft": 0.25,
    "hard_partial_weight": 0.70,        # Điểm từng phần cho số hard constraint đạt
    "hard_bonus_all_pass": 0.30,        # Bonus khi đúng 100% hard constraints
}

# Bộ nhớ đệm (Cache) kết quả Judge tránh gọi API trùng lặp cho các rollout giống nhau
_SOFT_JUDGE_CACHE = {}
_JUDGE_CLIENT_INSTANCE = None
_JUDGE_MODEL_RESOLVED = None


# ==============================================================================
# 2. HÀM TRÍCH XUẤT VĂN BẢN VÀ LÀM SẠCH THINKING TOKENS
# ==============================================================================
def extract_text_from_completion(comp) -> str:
    """
    Trích xuất an toàn chuỗi văn bản do mô hình sinh ra từ cấu trúc completion của TRL.
    Hỗ trợ cả dạng list [{'role': 'assistant', 'content': '...'}], dict, và raw string.
    """
    text = ""
    if isinstance(comp, list) and len(comp) > 0:
        first = comp[0]
        if isinstance(first, dict):
            text = first.get("content", "")
        else:
            text = str(first)
    elif isinstance(comp, dict):
        text = comp.get("content", "")
    elif isinstance(comp, str):
        text = comp
    else:
        text = str(comp)

    # Loại bỏ reasoning tokens (<think>...</think>) nếu model kích hoạt thinking mode
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return clean if clean else text


# ==============================================================================
# 3. BỘ CHẤM ĐIỂM SOFT CONSTRAINTS DUAL-ENGINE (GEMMA-4-31B | GEMINI)
# ==============================================================================
def evaluate_soft_constraints_llm(
    response_text: str,
    prompt_text: str = "",
    soft_spec: Any = None,
    judge_model: str = "gemini-2.5-flash",
    base_url: Optional[str] = None
) -> Optional[float]:
    """
    Thẩm định Soft Constraints bằng LLM-as-a-Judge (Hỗ trợ cả Gemma4-31B và Gemini):
    - Nhận cả đề bài yêu cầu và nội dung phản hồi của mô hình.
    - Đọc danh sách các ràng buộc ngữ nghĩa và rubric nhị phân (1/0).
    - Caching kết quả để không gọi lặp lại.
    - Trả về điểm trung bình [0.0, 1.0], hoặc None nếu gặp lỗi API / JSON hỏng.
    """
    global _JUDGE_CLIENT_INSTANCE, _JUDGE_MODEL_RESOLVED

    if not response_text.strip():
        return None

    # Parse soft_spec
    if isinstance(soft_spec, str):
        try:
            soft_list = json.loads(soft_spec)
        except Exception:
            return None
    elif isinstance(soft_spec, list):
        soft_list = soft_spec
    else:
        return None

    if not soft_list:
        return 1.0  # Không có soft constraints nào -> mặc định đạt

    # Kiểm tra cache
    cache_key = (prompt_text.strip()[:150], response_text.strip()[:200], json.dumps(soft_list, ensure_ascii=False, sort_keys=True))
    if cache_key in _SOFT_JUDGE_CACHE:
        return _SOFT_JUDGE_CACHE[cache_key]

    # Khởi tạo client dùng chung
    if _JUDGE_CLIENT_INSTANCE is None:
        client, resolved_m, _ = get_llm_client_and_model(judge_model, base_url)
        _JUDGE_CLIENT_INSTANCE = client
        _JUDGE_MODEL_RESOLVED = resolved_m
    else:
        client = _JUDGE_CLIENT_INSTANCE
        resolved_m = _JUDGE_MODEL_RESOLVED

    system_prompt = """Bạn là Giám khảo AI công tâm chấm điểm chất lượng câu trả lời theo rubric ngữ nghĩa (Soft Constraints).
Nhiệm vụ: Căn cứ vào Đề bài, Nội dung phản hồi và Rubric của từng tiêu chí để xác định câu trả lời đạt (1) hay không đạt (0).
Chỉ trả về DUY NHẤT một chuỗi JSON hợp lệ theo định dạng:
[
  {"id": "<id_tiêu_chí>", "score": 1, "reason": "<giải thích ngắn>"}
]"""

    user_prompt = f"""[ĐỀ BÀI YÊU CẦU]:
\"\"\"
{prompt_text[:2000]}
\"\"\"

[CÂU TRẢ LỜI CỦA MÔ HÌNH]:
\"\"\"
{response_text[:3000]}
\"\"\"

[DANH SÁCH TIÊU CHÍ SOFT CONSTRAINTS CẦN CHẤM]:
{json.dumps(soft_list, ensure_ascii=False, indent=2)}

Hãy thẩm định từng tiêu chí căn cứ vào Đề bài và Câu trả lời, sau đó trả về JSON:"""

    try:
        resp = client.chat.completions.create(
            model=resolved_m,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
            timeout=25.0
        )
        content = (resp.choices[0].message.content or "").strip()

        # Bóc tách JSON an toàn
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            m = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
            if m:
                json_str = m.group(0).strip()

        judged_items = json.loads(json_str)
        if isinstance(judged_items, list) and len(judged_items) > 0:
            scores = [float(item.get("score", 0)) for item in judged_items]
            final_soft_score = sum(scores) / len(scores)
        else:
            final_soft_score = None

    except Exception as e:
        print(f"⚠️ [Judge Warning - {resolved_m}]: {e}. Trả về None để loại khỏi thống kê.")
        final_soft_score = None

    if final_soft_score is not None:
        _SOFT_JUDGE_CACHE[cache_key] = final_soft_score
    return final_soft_score


# ==============================================================================
# 4. REWARD FUNCTION ĐỒNG BỘ VỚI VERIFIER ENGINE TIẾNG VIỆT
# ==============================================================================
def reward_func_verifiable(prompts, completions, verifier, soft=None, **kwargs):
    """
    Hàm tính Reward cho GRPOTrainer:
    - Nhận completions từ mô hình (danh sách B*G câu trả lời).
    - Tự động đồng bộ số lượng verifier & soft tương ứng với completions (chống lệch G).
    - Chấm Hard Constraints bằng Python Verifier Engine tất định.
    - Chấm Soft Constraints bằng Dual-Engine LLM Judge (Gemma4-31B hoặc Gemini).
    """
    n_comp = len(completions)
    n_v = len(verifier) if verifier is not None else 0
    n_p = len(prompts) if prompts is not None else 1
    G = max(1, n_comp // n_p)

    rewards = []
    cfg = GLOBAL_REWARD_CONFIG

    for idx, comp in enumerate(completions):
        # Trích xuất luật tương ứng cho từng rollout (tương thích cả batch repeat lẫn prompt-indexed)
        v_spec = verifier[idx] if n_v == n_comp else verifier[idx // G]
        s_spec = None
        if soft is not None:
            s_spec = soft[idx] if len(soft) == n_comp else soft[idx // G]

        response_text = extract_text_from_completion(comp)

        # 1. Chấm điểm qua Verifier Engine tiếng Việt (Hard Constraints)
        res = evaluate_instruction_following_vi(response_text, v_spec)
        partial_score = res.get("partial_score", 0.0)
        all_passed = res.get("all_hard_passed", False)

        # Công thức Hard Reward liên tục (Tránh Dead Gradients)
        hard_reward = (partial_score * cfg["hard_partial_weight"]) + (
            cfg["hard_bonus_all_pass"] if all_passed else 0.0
        )

        # 2. Gated Soft Reward (Được kích hoạt khi bật hybrid_gated và đã ĐẠT 100% hard constraints)
        soft_reward = 0.0
        if cfg["mode"] == "hybrid_gated" and all_passed and s_spec is not None:
            p_text = extract_text_from_completion(prompts[idx // G] if n_p > 0 else "")
            judge_score = evaluate_soft_constraints_llm(
                response_text=response_text,
                prompt_text=p_text,
                soft_spec=s_spec,
                judge_model=cfg["judge_model"],
                base_url=cfg["judge_base_url"]
            )
            soft_reward = judge_score if judge_score is not None else 0.5

        if cfg["mode"] == "hybrid_gated":
            final_reward = (cfg["w_hard"] * hard_reward) + (cfg["w_soft"] * soft_reward)
        else:
            final_reward = hard_reward

        rewards.append(final_reward)

    return rewards


# ==============================================================================
# 5. SMOKE TEST — KIỂM TRA NHANH TRƯỚC KHI TRAIN
# ==============================================================================
def smoke_test_reward(data_file: str, num_samples: int = 3):
    """
    Chạy smoke test reward function trên vài mẫu đầu tiên.
    Mục đích: xác nhận len(verifier) khớp len(completions) và reward có variance.
    """
    print("\n🔬 [SMOKE TEST] Kiểm tra Reward Function trên dữ liệu thật...")
    rows = []
    with open(data_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            rows.append(json.loads(line))

    for i, row in enumerate(rows):
        prompts = [row["prompt"]]
        # Giả lập G=2 completions đơn giản
        fake_completions = [
            [{"role": "assistant", "content": "Đây là câu trả lời mẫu số 1."}],
            [{"role": "assistant", "content": '"Đây là câu trả lời trong ngoặc kép."'}],
        ]
        verifier_expanded = [row["verifier"]] * 2  # Mô phỏng TRL expand
        soft_expanded = [row.get("soft", "[]")] * 2

        rewards = reward_func_verifiable(
            prompts=prompts,
            completions=fake_completions,
            verifier=verifier_expanded,
            soft=soft_expanded
        )
        print(f"   Mẫu {i+1} ({row['key']}): rewards = {rewards}, std = {max(rewards)-min(rewards):.4f}")

    print("✅ Smoke test hoàn tất! Reward function hoạt động bình thường.\n")


# ==============================================================================
# 6. HUẤN LUYỆN GRPO TRÊN GOOGLE COLAB / KAGGLE
# ==============================================================================
def run_grpo_colab(
    grpo_data_file="datasets/grpo/train_seen.jsonl",
    output_dir="checkpoints/qwen_grpo",
    num_generations=4,
    epochs=1
):
    """
    Thực thi huấn luyện RLVR GRPO không cần đáp án mẫu.
    """
    import sys
    for m in list(sys.modules.keys()):
        if "torchao" in m:
            del sys.modules[m]
    try:
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False
    except Exception:
        pass

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from peft import LoraConfig, TaskType
    from trl import GRPOTrainer, GRPOConfig

    print("=" * 65)
    print(f"🚀 [ZERO-SFT GRPO] Khởi động huấn luyện từ: {MODEL_NAME}")
    print(f"   • Tập dữ liệu train   : {grpo_data_file}")
    print(f"   • Số rollout G       : {num_generations}")
    print(f"   • Chế độ Reward      : {GLOBAL_REWARD_CONFIG['mode']}")
    if GLOBAL_REWARD_CONFIG["mode"] == "hybrid_gated":
        print(f"   • LLM Judge Provider : {GLOBAL_REWARD_CONFIG['judge_model']}")
        if GLOBAL_REWARD_CONFIG["judge_base_url"]:
            print(f"   • Judge Base URL     : {GLOBAL_REWARD_CONFIG['judge_base_url']}")
    print("=" * 65)

    lora_config = LoraConfig(
        r=LORA_CONFIG_PARAMS["r"],
        lora_alpha=LORA_CONFIG_PARAMS["lora_alpha"],
        target_modules=LORA_CONFIG_PARAMS["target_modules"],
        lora_dropout=LORA_CONFIG_PARAMS["lora_dropout"],
        bias=LORA_CONFIG_PARAMS["bias"],
        task_type=TaskType.CAUSAL_LM
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect bf16 vs fp16
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    print(f"   • Precision          : {'bf16' if use_bf16 else 'fp16'}")

    # Cấu hình GRPO tối ưu bộ nhớ cho Colab T4 / Kaggle 16GB
    import inspect
    grpo_params = dict(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=1,       # 1 prompt per step
        gradient_accumulation_steps=8,       # Effective batch size = 8
        num_generations=num_generations,     # G = 4 completions
        max_completion_length=384,
        num_train_epochs=epochs,
        logging_steps=2,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to="none"
    )
    sig_params = inspect.signature(GRPOConfig.__init__).parameters
    if "max_prompt_length" in sig_params:
        grpo_params["max_prompt_length"] = 384

    training_args = GRPOConfig(**grpo_params)

    # Nạp dataset (PyArrow đọc cột 'verifier' và 'soft' dạng string JSON an toàn tuyệt đối)
    ds = load_dataset("json", data_files={"train": grpo_data_file})
    print(f"📖 Đã nạp thành công {len(ds['train'])} mẫu huấn luyện.")

    # Smoke test trước khi train
    smoke_test_reward(grpo_data_file, num_samples=2)

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=[reward_func_verifiable],
        args=training_args,
        train_dataset=ds["train"],
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"🎉 Huấn luyện hoàn tất! Checkpoint đã lưu tại: {output_dir}")


def check_dataset_health(data_file: str):
    """Kiểm tra toàn diện tính toàn vẹn của tập dữ liệu train GRPO trước khi nạp vào Trainer."""
    if not os.path.exists(data_file):
        print(f"❌ File không tồn tại: '{data_file}'")
        return False

    print(f"🔍 Đang kiểm tra định dạng tập dữ liệu: '{data_file}' ({os.path.getsize(data_file):,} bytes)...")
    
    with open(data_file, "r", encoding="utf-8-sig") as f:
        head_sample = f.read(200).strip()
    
    if not head_sample:
        print(f"❌ File '{data_file}' rỗng hoàn toàn (0 bytes).")
        return False

    if head_sample.startswith("<"):
        print(f"❌ LỖI: File '{data_file}' là định dạng HTML (do tải nhầm URL web thay vì file RAW/Dataset):")
        print(head_sample[:100])
        return False

    count = 0
    errors = []
    with open(data_file, "r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            count += 1
            try:
                row = json.loads(line_str)
                for fld in ["key", "prompt", "verifier", "soft", "num_hard"]:
                    if fld not in row:
                        errors.append(f"Dòng {idx}: Thiếu trường '{fld}'")
                v_list = json.loads(row["verifier"])
                if not isinstance(v_list, list):
                    errors.append(f"Dòng {idx}: 'verifier' không phải là list")
                if not isinstance(row["prompt"], list) or len(row["prompt"]) == 0:
                    errors.append(f"Dòng {idx}: 'prompt' không đúng định dạng list")
            except Exception as e:
                errors.append(f"Dòng {idx}: Lỗi cú pháp JSON: {e} | Ký tự: {repr(line_str[:60])}")

    if errors:
        print(f"❌ Phát hiện {len(errors)} lỗi trong {count} mẫu:")
        for err in errors[:5]:
            print(f"   • {err}")
        return False
    else:
        print(f"✅ Tuyệt vời! Toàn bộ {count} mẫu trong '{data_file}' hợp lệ 100% chuẩn GRPOTrainer (PyArrow safe).")
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Huấn luyện Zero-SFT GRPO cho Qwen trên Google Colab / Kaggle")
    parser.add_argument("--data-file", type=str, default="datasets/grpo/train_seen.jsonl",
                        help="Đường dẫn file dataset train GRPO")
    parser.add_argument("--output-dir", type=str, default="checkpoints/qwen_grpo",
                        help="Thư mục lưu checkpoint mô hình")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="Tên checkpoint HuggingFace (mặc định: Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Số rollout completions G cho mỗi prompt (mặc định: 4)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Số epoch huấn luyện (mặc định: 1)")
    parser.add_argument("--mode", type=str, default="hard_only", choices=["hard_only", "hybrid_gated"],
                        help="Chế độ reward: 'hard_only' hoặc 'hybrid_gated'")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Tên LLM Judge Soft Constraints: ví dụ 'gemini-2.5-flash' hoặc 'google/gemma-4-31B-it' / 'gemma'")
    parser.add_argument("--judge-base-url", type=str, default=None,
                        help="Base URL API cho LLM Judge (dùng cho server vLLM/Cloudflare của Gemma)")
    parser.add_argument("--check-data", action="store_true",
                        help="Chỉ kiểm tra tính hợp lệ của file dataset mà không chạy train")

    args = parser.parse_args()

    GLOBAL_REWARD_CONFIG["mode"] = args.mode
    if args.judge_model:
        GLOBAL_REWARD_CONFIG["judge_model"] = args.judge_model
    if args.judge_base_url:
        GLOBAL_REWARD_CONFIG["judge_base_url"] = args.judge_base_url

    MODEL_NAME = args.model

    if args.check_data:
        check_dataset_health(args.data_file)
    else:
        is_ok = check_dataset_health(args.data_file)
        if is_ok:
            run_grpo_colab(
                grpo_data_file=args.data_file,
                output_dir=args.output_dir,
                num_generations=args.num_generations,
                epochs=args.epochs
            )
        else:
            print("❌ Vui lòng sửa lỗi dataset trước khi tiến hành train.")

