#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: fetch_and_translate.py
Mục đích: Tải dữ liệu từ 2 tập HuggingFace:
          1. vicgalle/alpaca-gpt4 (mặc định 250 mẫu, có thể tùy chỉnh)
          2. Post-training-Data-Flywheel/gpt4-self-instruct (mặc định 250 mẫu, có thể tùy chỉnh)
          và dịch sang tiếng Việt bằng LLM qua OpenAI-compatible API (BASE_URL & MODEL trong .env).

Tính năng đánh Index & Tùy biến mẫu:
- Đánh chỉ số toàn cục (Global ID): 1, 2, 3, ... N tuần tự.
- Đánh chỉ số nguồn (Source Index): source_idx (0, 1, 2, ...) tương ứng với vị trí trong tập gốc.
- Đánh mã định danh (Key): 'alpaca-gpt4_0000', 'self-instruct_0000'...
- Tùy chỉnh số lượng mẫu từng tập: --num-alpaca, --num-self-instruct, --alpaca-start-idx, --self-instruct-start-idx.
- Cờ lọc: --id (lọc theo ID/key), --start-idx, --end-idx, --resume (tiếp tục an toàn).
- Hoàn toàn loại bỏ mã thừa SFT; tập trung tạo tập dữ liệu nền chuẩn cho Zero-SFT GRPO pipeline.
"""

import os
import sys
import json
import time
import argparse
import datetime
from typing import Dict, Any, List, Optional, Tuple
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
from llm_client import get_llm_client_and_model, translate_instruction_vi

load_dotenv()

DEFAULT_MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite").strip()


TRANSLATION_SYSTEM_PROMPT_PROMPT_ONLY = """Bạn là một chuyên gia dịch thuật cao cấp Anh - Việt chuyên sâu về AI & Khoa học máy tính.
Nhiệm vụ của bạn là dịch 2 trường 'instruction' và 'input' từ tiếng Anh sang tiếng Việt tự nhiên, chuẩn mực.

QUY TẮC BẮT BUỘC:
1. Dịch chuẩn xác, tự nhiên, đúng ngữ cảnh công nghệ và ngôn ngữ học tiếng Việt.
2. Giữ NGUYÊN vẹn formatting: Markdown, code blocks, bullet points, công thức toán, biến số hoặc tên hàm lập trình.
3. KHÔNG dịch tên riêng, tên thư viện (ví dụ: PyTorch, Pandas, Hugging Face, Python), URLs hoặc chuỗi regex.
4. Nếu trường 'input' rỗng hoặc None, hãy trả về chuỗi rỗng "".
5. KHÔNG thêm bất kỳ lời bình luận hay chào hỏi mở đầu nào.
6. Trả về DUY NHẤT một chuỗi JSON hợp lệ với đúng 2 khóa:
{
  "instruction_vi": "bản dịch tiếng Việt của instruction",
  "input_vi": "bản dịch tiếng Việt của input (hoặc rỗng nếu input gốc rỗng)"
}"""

TRANSLATION_SYSTEM_PROMPT_FULL = """Bạn là một chuyên gia dịch thuật cao cấp Anh - Việt chuyên sâu về AI & Khoa học máy tính.
Nhiệm vụ của bạn là dịch chính xác 3 trường 'instruction', 'input', và 'output' từ tiếng Anh sang tiếng Việt tự nhiên, chuẩn mực.

QUY TẮC BẮT BUỘC:
1. Dịch chuẩn xác, tự nhiên, đúng ngữ cảnh công nghệ và ngôn ngữ học tiếng Việt.
2. Giữ NGUYÊN vẹn formatting: Markdown, code blocks, bullet points, công thức toán, biến số hoặc tên hàm lập trình.
3. KHÔNG dịch tên riêng, tên thư viện (ví dụ: PyTorch, Pandas, Hugging Face, Python), URLs hoặc chuỗi regex.
4. Nếu trường 'input' rỗng hoặc None, hãy trả về chuỗi rỗng "".
5. KHÔNG thêm bất kỳ lời bình luận hay chào hỏi mở đầu nào.
6. Trả về DUY NHẤT một chuỗi JSON hợp lệ với đúng 3 khóa:
{
  "instruction_vi": "bản dịch tiếng Việt của instruction",
  "input_vi": "bản dịch tiếng Việt của input (hoặc rỗng nếu input gốc rỗng)",
  "output_vi": "bản dịch tiếng Việt của output"
}"""


def translate_item(
    client: OpenAI,
    model: str,
    instruction: str,
    input_text: str = "",
    output_text: str = "",
    translate_output: bool = False,
    max_retries: int = 5,
    initial_delay: float = 2.0
) -> Tuple[str, str, str]:
    """
    Dịch các trường sang tiếng Việt. 
    Mặc định (translate_output=False) chỉ dịch instruction và input (tối ưu hóa cho Zero-SFT GRPO).
    """
    if translate_output:
        system_prompt = TRANSLATION_SYSTEM_PROMPT_FULL
        payload = {
            "instruction": instruction.strip() if instruction else "",
            "input": input_text.strip() if input_text else "",
            "output": output_text.strip() if output_text else ""
        }
    else:
        system_prompt = TRANSLATION_SYSTEM_PROMPT_PROMPT_ONLY
        payload = {
            "instruction": instruction.strip() if instruction else "",
            "input": input_text.strip() if input_text else ""
        }

    user_prompt = f"Hãy dịch JSON sau sang tiếng Việt:\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            raw_content = response.choices[0].message.content.strip()

            json_text = raw_content
            if "```json" in raw_content:
                json_text = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                json_text = raw_content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(json_text)
            inst_vi = parsed.get("instruction_vi", "").strip() or instruction
            inp_vi = parsed.get("input_vi", "").strip() or input_text
            out_vi = parsed.get("output_vi", "").strip() if translate_output else output_text

            return inst_vi, inp_vi, out_vi

        except Exception as e:
            if attempt == max_retries:
                print(f"⚠️ [Translate Error] Lỗi sau {max_retries} lần thử: {e}. Fallback giữ nguyên bản gốc.")
                return instruction, input_text, output_text
            sleep_time = initial_delay * (2 ** (attempt - 1))
            print(f"🔄 [Retry {attempt}/{max_retries}] Lỗi: {e}. Đợi {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    return instruction, input_text, output_text


def extract_sample_fields(item: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Trích xuất chuẩn hóa instruction, input, output từ cả 2 dạng dataset."""
    instruction = ""
    input_text = ""
    output_text = ""

    if "instruction" in item:
        # vicgalle/alpaca-gpt4
        instruction = item.get("instruction", "") or ""
        input_text = item.get("input", "") or ""
        output_text = item.get("output", "") or ""

    elif "messages" in item:
        # Post-training-Data-Flywheel/gpt4-self-instruct dạng hội thoại
        msgs = item.get("messages", [])
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "") or ""
            if role == "user" and not instruction:
                instruction = content
            elif role == "assistant" and not output_text:
                output_text = content

    elif "prompt" in item and "response" in item:
        instruction = item.get("prompt", "") or ""
        output_text = item.get("response", "") or ""

    instruction = instruction.strip()
    output_text = output_text.strip()
    input_text = input_text.strip()

    if not instruction or len(instruction) < 5 or not output_text:
        return None

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }


def fetch_raw_hf_samples(
    repo_name: str,
    source_prefix: str,
    count: int,
    start_skip: int = 0
) -> List[Dict[str, Any]]:
    """
    Tải dữ liệu từ HuggingFace với chỉ số source_idx được đánh số chính xác.
    """
    print(f"📥 Đang kết nối HuggingFace: '{repo_name}' (Bắt đầu từ index {start_skip}, lấy {count} mẫu)...")
    collected = []
    skipped = 0
    valid_seen = 0

    try:
        ds = load_dataset(repo_name, split="train", streaming=True)
        for raw_idx, item in enumerate(ds):
            parsed = extract_sample_fields(item)
            if not parsed:
                continue

            # Bỏ qua các mẫu trước start_skip
            if skipped < start_skip:
                skipped += 1
                continue

            formatted_key = f"{source_prefix}_{valid_seen:04d}"
            collected.append({
                "source_name": repo_name,
                "source_prefix": source_prefix,
                "source_idx": valid_seen,
                "key": formatted_key,
                "instruction_en": parsed["instruction"],
                "input_en": parsed["input"],
                "output_en": parsed["output"]
            })
            valid_seen += 1

            if len(collected) >= count:
                break

        print(f"   ✅ Đã thu thập thành công {len(collected)} mẫu từ '{repo_name}'.")
    except Exception as e:
        print(f"   ❌ Lỗi khi tải dữ liệu từ {repo_name}: {e}")

    return collected


def main():
    parser = argparse.ArgumentParser(description="Tải dữ liệu 250 mẫu mỗi tập và dịch sang tiếng Việt.")
    parser.add_argument("--num-alpaca", type=int, default=250,
                        help="Số lượng mẫu lấy từ vicgalle/alpaca-gpt4 (mặc định: 250)")
    parser.add_argument("--num-self-instruct", type=int, default=250,
                        help="Số lượng mẫu lấy từ Post-training-Data-Flywheel/gpt4-self-instruct (mặc định: 250)")
    parser.add_argument("--alpaca-start-idx", type=int, default=0,
                        help="Chỉ số bắt đầu lấy mẫu của tập alpaca (mặc định: 0)")
    parser.add_argument("--self-instruct-start-idx", type=int, default=0,
                        help="Chỉ số bắt đầu lấy mẫu của tập self-instruct (mặc định: 0)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Tổng số mẫu mục tiêu (nếu truyền sẽ chia đều cho 2 tập)")
    parser.add_argument("--id", type=str, default=None,
                        help="Chạy duy nhất 1 sample theo key (ví dụ: 'alpaca-gpt4_0001' hoặc ID số nguyên '1')")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Chỉ mục bắt đầu duyệt trong danh sách tổng hợp (0-indexed)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Chỉ mục kết thúc duyệt trong danh sách tổng hợp")
    parser.add_argument("--output", type=str, default="datasets/raw/base_vi.jsonl",
                        help="File JSONL đầu ra (mặc định: datasets/raw/base_vi.jsonl)")
    parser.add_argument("--resume", action="store_true",
                        help="Bỏ qua các mẫu đã có trong file output và tiếp tục dịch tiếp")
    parser.add_argument("--no-translate", action="store_true",
                        help="Chỉ tải và đánh index các mẫu từ Hugging Face mà không gọi API dịch thuật.")
    parser.add_argument("--translate-output", action="store_true",
                        help="Dịch cả trường output (mặc định False vì Zero-SFT GRPO không cần đáp án mẫu, tiết kiệm 80%% token)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="OpenAI API Base URL (mặc định tự nhận diện theo model)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Mô hình LLM dịch: 'gemini-3.5-flash-lite' hoặc 'gemma4-31B-it' (mặc định: gemini-3.5-flash-lite)")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Thời gian nghỉ (giây) giữa các request API")
    args = parser.parse_args()

    # Tính toán số lượng mẫu cho từng nguồn
    count_alpaca = args.num_alpaca
    count_self_instruct = args.num_self_instruct

    if args.num_samples is not None:
        count_alpaca = args.num_samples // 2
        count_self_instruct = args.num_samples - count_alpaca

    print("=" * 65)
    print("📊 KẾ HOẠCH THU THẬP VÀ ĐÁNH INDEX DỮ LIỆU:")
    print(f"   • vicgalle/alpaca-gpt4                     : {count_alpaca} mẫu (start_idx: {args.alpaca_start_idx})")
    print(f"   • Post-training-Data-Flywheel/gpt4-self-instruct: {count_self_instruct} mẫu (start_idx: {args.self_instruct_start_idx})")
    print(f"   • Tổng số mẫu mục tiêu                     : {count_alpaca + count_self_instruct} mẫu")
    print(f"   • File đích đầu ra                         : {args.output}")
    print(f"   • Chế độ dịch                              : {'Không dịch (--no-translate)' if args.no_translate else ('Dịch đầy đủ (+output)' if args.translate_output else 'Dịch Prompt (Zero-SFT)')}")
    print("=" * 65)

    # 1. Thu thập và đánh index cho tập Alpaca
    raw_alpaca = fetch_raw_hf_samples(
        repo_name="vicgalle/alpaca-gpt4",
        source_prefix="alpaca-gpt4",
        count=count_alpaca,
        start_skip=args.alpaca_start_idx
    )

    # 2. Thu thập và đánh index cho tập Self-Instruct
    raw_self = fetch_raw_hf_samples(
        repo_name="Post-training-Data-Flywheel/gpt4-self-instruct",
        source_prefix="self-instruct",
        count=count_self_instruct,
        start_skip=args.self_instruct_start_idx
    )

    # Hợp nhất danh sách và gán Global ID tuần tự (1, 2, 3...)
    all_raw_samples = []
    global_id = 1

    for item in raw_alpaca:
        item["id"] = global_id
        all_raw_samples.append(item)
        global_id += 1

    for item in raw_self:
        item["id"] = global_id
        all_raw_samples.append(item)
        global_id += 1

    print(f"\n📦 Đã chuẩn bị {len(all_raw_samples)} mẫu thô với đầy đủ chỉ mục (Global ID: 1 -> {len(all_raw_samples)}).")

    # Lọc theo cờ --id nếu có
    if args.id:
        target_id_str = str(args.id).strip()
        all_raw_samples = [
            it for it in all_raw_samples 
            if str(it["id"]) == target_id_str or it["key"] == target_id_str or target_id_str in it["key"]
        ]
        print(f"🎯 Lọc theo --id '{args.id}': tìm thấy {len(all_raw_samples)} mẫu.")

    # Lọc theo dải --start-idx và --end-idx
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(all_raw_samples)
    selected_samples = all_raw_samples[start_idx:end_idx]

    print(f"⚙️ Duyệt xử lý {len(selected_samples)} mẫu trong đợt chạy này (index {start_idx} -> {end_idx}).")

    # Kiểm tra các mẫu đã có trong output nếu dùng --resume
    existing_keys = set()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f_check:
            for line in f_check:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if "key" in obj:
                            existing_keys.add(obj["key"])
                    except Exception:
                        pass
        print(f"🔄 [RESUME] Đã phát hiện {len(existing_keys)} mẫu đã tồn tại trong '{args.output}'.")

    # Xử lý chế độ không dịch
    if args.no_translate:
        print("⚡ [Chế độ No-Translate] Lưu trữ trực tiếp dữ liệu thô đã đánh chỉ mục vào file output.")
        open_mode = "a" if (args.resume and os.path.exists(args.output)) else "w"
        saved_count = 0
        skipped_count = 0

        with open(args.output, open_mode, encoding="utf-8") as f_out:
            for item in selected_samples:
                key = item["key"]
                if args.resume and key in existing_keys:
                    skipped_count += 1
                    continue

                record = {
                    "id": item["id"],
                    "source_idx": item["source_idx"],
                    "key": key,
                    "dataset_source": item["source_name"],
                    "instruction_en": item["instruction_en"],
                    "input_en": item["input_en"],
                    "output_en": item["output_en"],
                    "instruction_vi": item["instruction_en"],  # Fallback sẵn sàng cho pipeline
                    "input_vi": item["input_en"],
                    "output_vi": item["output_en"],
                    "status": "raw_indexed",
                    "indexed_at": datetime.datetime.now().isoformat()
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                existing_keys.add(key)
                saved_count += 1

        print("\n" + "=" * 65)
        print("🎉 HOÀN THÀNH THU THẬP VÀ ĐÁNH INDEX DỮ LIỆU:")
        print(f"   • Đã lưu mới  : {saved_count} mẫu")
        print(f"   • Đã bỏ qua   : {skipped_count} mẫu (đã có từ trước)")
        print(f"   • File đích   : {args.output}")
        print("=" * 65)
        return

    # Bắt đầu dịch thuật qua API
    client, resolved_model, resolved_url = get_llm_client_and_model(args.model, args.base_url)
    print("=" * 65)
    print(f"🌐 [Chế độ Dịch] Khởi tạo Client cho model: '{resolved_model}'")
    print(f"   • API Endpoint: {resolved_url}")
    print("=" * 65)

    open_mode = "a" if (args.resume and os.path.exists(args.output)) else "w"
    translated_count = 0
    skipped_count = 0

    with open(args.output, open_mode, encoding="utf-8") as f_out:
        for idx, item in enumerate(selected_samples, start=1):
            key = item["key"]

            if args.resume and key in existing_keys:
                skipped_count += 1
                continue

            print(f"[{idx}/{len(selected_samples)}] Dịch Global ID {item['id']} ({key}) [{item['source_name']}]...")

            inst_vi, inp_vi, out_vi = translate_item(
                client=client,
                model=args.model,
                instruction=item["instruction_en"],
                input_text=item["input_en"],
                output_text=item["output_en"],
                translate_output=args.translate_output
            )

            record = {
                "id": item["id"],
                "source_idx": item["source_idx"],
                "key": key,
                "dataset_source": item["source_name"],
                "instruction_en": item["instruction_en"],
                "input_en": item["input_en"],
                "output_en": item["output_en"],
                "instruction_vi": inst_vi,
                "input_vi": inp_vi,
                "output_vi": out_vi,
                "status": "translated",
                "translated_at": datetime.datetime.now().isoformat()
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            existing_keys.add(key)
            translated_count += 1

            if args.delay > 0:
                time.sleep(args.delay)

    print("\n" + "=" * 65)
    print("🎉 HOÀN THÀNH MODULE THU THẬP VÀ DỊCH DỮ LIỆU:")
    print(f"   • Đã dịch mới : {translated_count} mẫu")
    print(f"   • Đã bỏ qua   : {skipped_count} mẫu (đã có từ trước)")
    print(f"   • File lưu    : {args.output}")
    print("=" * 65)


if __name__ == "__main__":
    main()
