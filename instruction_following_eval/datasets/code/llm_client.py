#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: llm_client.py
Mục đích: Bộ điều phối Client LLM dùng chung cho toàn bộ Pipeline:
          - Tự động chuyển đổi giữa 'gemini-3.5-flash-lite' (hoặc họ Gemini) và 'gemma4-31B-it' (hoặc họ Gemma).
          - Định tuyến chính xác API endpoint (Google AI Studio vs. vLLM Remote/Local endpoint).
          - Hàm dịch thuật chuẩn hóa Anh -> Việt chuyên biệt cho Zero-SFT RLVR.
"""

import os
import sys
import json
import time
from typing import Tuple, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEFAULT_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"

TRANSLATION_SYSTEM_PROMPT = """Bạn là một chuyên gia dịch thuật cao cấp Anh - Việt chuyên sâu về AI & Khoa học máy tính.
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


def get_llm_client_and_model(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 60.0
) -> Tuple[OpenAI, str, str]:
    """
    Tự động nhận diện và cấu hình OpenAI Client:
    - Nếu model_name chứa 'gemini' (ví dụ 'gemini-3.5-flash-lite'):
        + base_url: https://generativelanguage.googleapis.com/v1beta/openai/
        + api_key: GOOGLE_API_KEY từ .env
        + model: chuẩn hóa tên model gemini
    - Nếu model_name chứa 'gemma' (ví dụ 'gemma4-31B-it', 'google/gemma-4-31B-it'):
        + Nếu có base_url hoặc BASE_URL trong .env: dùng endpoint vLLM/LMDeploy
        + Nếu không có BASE_URL: tự động dùng Google endpoint với GOOGLE_API_KEY
    
    Trả về: (client, normalized_model, resolved_base_url)
    """
    google_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    env_base_url = os.getenv("BASE_URL", "").strip()
    env_model = os.getenv("MODEL", "gemini-3.5-flash-lite").strip()

    req_model = model_name or env_model
    m_lower = req_model.lower()

    if "gemini" in m_lower:
        resolved_url = DEFAULT_GEMINI_ENDPOINT
        api_key = google_key
        if "3.5" in m_lower and "lite" in m_lower:
            norm_model = "gemini-3.5-flash-lite"
        elif "2.5" in m_lower and "lite" in m_lower:
            norm_model = "gemini-2.5-flash-lite"
        elif "2.5" in m_lower:
            norm_model = "gemini-2.5-flash"
        else:
            norm_model = req_model
    elif "gemma" in m_lower:
        endpoint_candidate = (base_url or env_base_url).strip().rstrip("/")
        if endpoint_candidate and ("trycloudflare" in endpoint_candidate or not "googleapis" in endpoint_candidate):
            if endpoint_candidate.endswith("/models"):
                endpoint_candidate = endpoint_candidate[:-7].rstrip("/")
            if endpoint_candidate.endswith("/docs"):
                endpoint_candidate = endpoint_candidate[:-5].rstrip("/")
            resolved_url = endpoint_candidate if endpoint_candidate.endswith("/v1") else f"{endpoint_candidate}/v1"
            api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
            norm_model = "google/gemma-4-31B-it" if "31b" in m_lower else req_model
        else:
            resolved_url = DEFAULT_GEMINI_ENDPOINT
            api_key = google_key
            norm_model = "gemma-4-31b-it"
    else:
        resolved_url = base_url or env_base_url or DEFAULT_GEMINI_ENDPOINT
        api_key = google_key if "googleapis" in resolved_url else os.getenv("OPENAI_API_KEY", "EMPTY")
        norm_model = req_model

    client = OpenAI(base_url=resolved_url, api_key=api_key or "EMPTY", timeout=timeout)
    return client, norm_model, resolved_url


def translate_instruction_vi(
    client: OpenAI,
    model: str,
    instruction: str,
    input_text: str = "",
    max_retries: int = 5,
    initial_delay: float = 2.0
) -> Tuple[str, str]:
    """
    Dịch 2 trường instruction và input sang tiếng Việt (chuẩn Zero-SFT RLVR).
    Trả về: (instruction_vi, input_vi)
    """
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
                    {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            raw_content = response.choices[0].message.content or ""
            raw_content = raw_content.strip()
            if not raw_content:
                raise ValueError("LLM returned empty content")

            json_text = raw_content
            if "```json" in raw_content:
                json_text = raw_content.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_content:
                json_text = raw_content.split("```")[1].split("```")[0].strip()

            parsed = json.loads(json_text)
            inst_vi = parsed.get("instruction_vi", "").strip() or instruction
            inp_vi = parsed.get("input_vi", "").strip() or input_text

            return inst_vi, inp_vi

        except Exception as e:
            if attempt == max_retries:
                print(f"⚠️ [Dịch thất bại sau {max_retries} lần thử]: {e}. Giữ nguyên tiếng Anh.")
                return instruction, input_text
            sleep_time = initial_delay * (2 ** (attempt - 1))
            time.sleep(sleep_time)

    return instruction, input_text
