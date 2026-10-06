#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 3: cross_relationship_perturbation_vi.py
Mục đích: Thuật toán Đột biến Quan hệ Chéo (Cross-Relationship Perturbation - CRPL) 
          kết hợp cả Hard + Soft Constraints tiếng Việt để sinh các biến thể prompt.

Cơ chế sinh mẫu > 16 mẫu / batch:
- Tiếp nhận 16 mẫu gốc (seed samples) từ Module 2 (results/module_soft_constraints/).
- Mỗi mẫu gốc sinh thêm 3 biến thể đột biến trên Hard Constraints dựa trên 3 quan hệ tập hợp:
    1. Containment (Bao hàm: mở rộng/thu hẹp miền giá trị)
    2. Partial Overlap (Giao thoa: bổ sung/dịch chuyển ràng buộc giao nhau)
    3. Disjoint / Negation (Loại trừ/Đối lập hoàn toàn)
- Giữ nguyên 100% Soft Constraints đã được kiểm định.
- Tái tạo prompt tiếng Việt 2 tầng hoàn chỉnh (Ràng buộc Cấu trúc + Hướng dẫn Ngữ nghĩa).
- Hệ số mở rộng: 16 mẫu gốc * (1 gốc + 3 biến thể) = 64 mẫu / batch (> 16 mẫu).
- Lưu kết quả vào: 'results/module_final/module3_final_batch_{id}_{timestamp}.json'.
- Hỗ trợ các cờ: --id, --start-idx, --end-idx, --resume, --batch-size 16, --split (seen/unseen).
"""

import os
import sys
import json
import random
import re
import copy
import argparse
import datetime
import unicodedata
from typing import List, Dict, Any, Tuple, Optional

WORD_BANK_VI = [
    "công nghệ", "chiến lược", "đổi mới", "bền vững", "giải pháp",
    "phát triển", "hài hòa", "sáng tạo", "thách thức", "cân bằng",
    "hiệu quả", "tiềm năng", "xu hướng", "nền tảng", "tối ưu"
]


def norm_vi(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# ==============================================================================
# 1. PERTURBATION RULES CHO TỪNG LOẠI HARD CONSTRAINT (TIẾNG VIỆT)
# ==============================================================================

def perturb_number_words_vi(kwargs: Dict[str, Any], mode: str) -> Tuple[str, Dict[str, Any]]:
    rel = kwargs.get("relation", "at least")
    n = kwargs.get("num_words", 100)

    if mode == "containment":
        if rel == "at least":
            new_n = max(10, n - random.randint(20, 40))
            return f"Câu trả lời của bạn phải có ít nhất {new_n} từ.", {"num_words": new_n, "relation": "at least"}
        else:
            new_n = n + random.randint(30, 60)
            return f"Câu trả lời của bạn phải có ít hơn {new_n} từ.", {"num_words": new_n, "relation": "less than"}

    elif mode == "partial_overlap":
        if rel == "at least":
            new_low = n + 10
            new_high = new_low + 80
            return f"Câu trả lời của bạn phải có độ dài từ {new_low} đến {new_high} từ.", {
                "num_words": new_high, "relation": "range", "min_words": new_low, "max_words": new_high
            }
        else:
            shift = random.randint(20, 40)
            new_low = max(10, n - shift)
            return f"Câu trả lời của bạn phải có ít nhất {new_low} từ.", {"num_words": new_low, "relation": "at least"}

    else:  # disjoint
        if rel == "at least":
            new_n = max(10, n // 2)
            return f"Câu trả lời của bạn phải có ít hơn {new_n} từ.", {"num_words": new_n, "relation": "less than"}
        else:
            new_n = n + random.randint(50, 100)
            return f"Câu trả lời của bạn phải có ít nhất {new_n} từ.", {"num_words": new_n, "relation": "at least"}


def perturb_number_sentences_vi(kwargs: Dict[str, Any], mode: str) -> Tuple[str, Dict[str, Any]]:
    rel = kwargs.get("relation", "at least")
    n = kwargs.get("num_sentences", 3)

    if mode == "containment":
        new_n = max(1, n - 1)
        return f"Phản hồi phải chứa ít nhất {new_n} câu.", {"num_sentences": new_n, "relation": "at least"}
    elif mode == "partial_overlap":
        new_n = n + 1
        return f"Phản hồi phải chứa chính xác {new_n} câu.", {"num_sentences": new_n, "relation": "exactly"}
    else:  # disjoint
        new_n = n + 4
        return f"Phản hồi phải chứa ít nhất {new_n} câu.", {"num_sentences": new_n, "relation": "at least"}


def perturb_number_paragraphs_vi(kwargs: Dict[str, Any], mode: str) -> Tuple[str, Dict[str, Any]]:
    n = kwargs.get("num_paragraphs", 3)
    if mode == "containment":
        new_n = max(2, n - 1)
        return f"Cấu trúc bài viết thành chính xác {new_n} đoạn văn phân cách bằng dòng trống kép.", {"num_paragraphs": new_n}
    elif mode == "partial_overlap":
        new_n = n + 1
        return f"Cấu trúc bài viết thành chính xác {new_n} đoạn văn phân cách bằng dòng trống kép.", {"num_paragraphs": new_n}
    else:  # disjoint
        new_n = n + 3
        return f"Cấu trúc bài viết thành chính xác {new_n} đoạn văn phân cách bằng dòng trống kép.", {"num_paragraphs": new_n}


def perturb_bullet_lists_vi(kwargs: Dict[str, Any], mode: str) -> Tuple[str, Dict[str, Any]]:
    n = kwargs.get("num_bullets", 3)
    if mode == "containment":
        new_n = max(2, n - 1)
        return f"Phải chứa chính xác {new_n} mục gạch đầu dòng dạng '*' hoặc '-'.", {"num_bullets": new_n}
    elif mode == "partial_overlap":
        new_n = n + 1
        return f"Phải chứa chính xác {new_n} mục gạch đầu dòng dạng '*' hoặc '-'.", {"num_bullets": new_n}
    else:  # disjoint
        new_n = n + 4
        return f"Phải chứa chính xác {new_n} mục gạch đầu dòng dạng '*' hoặc '-'.", {"num_bullets": new_n}


def perturb_keywords_existence_vi(kwargs: Dict[str, Any], mode: str) -> Tuple[str, Dict[str, Any]]:
    kws = kwargs.get("keywords", ["công nghệ"])
    if mode == "containment":
        sub_kws = kws[:max(1, len(kws) - 1)]
        return f"Bắt buộc chứa từ khóa '{', '.join(sub_kws)}' trong câu trả lời.", {"keywords": sub_kws}
    elif mode == "partial_overlap":
        add_w = random.choice([w for w in WORD_BANK_VI if w not in kws])
        overlap_kws = kws + [add_w]
        return f"Bắt buộc chứa từ khóa '{', '.join(overlap_kws)}' trong câu trả lời.", {"keywords": overlap_kws}
    else:  # disjoint (đổi thành từ cấm)
        return f"Tuyệt đối không được chứa các từ sau: {', '.join(kws)}.", {"forbidden_words": kws}


def perturb_keywords_frequency_vi(kwargs: Dict[str, Any], mode: str) -> Tuple[str, Dict[str, Any]]:
    kw = kwargs.get("keyword", "giải pháp")
    freq = kwargs.get("frequency", 2)
    if mode == "containment":
        new_f = max(1, freq - 1)
        return f"Từ khóa '{kw}' phải xuất hiện ít nhất {new_f} lần.", {"keyword": kw, "frequency": new_f, "relation": "at least"}
    elif mode == "partial_overlap":
        new_f = freq + 1
        return f"Từ khóa '{kw}' phải xuất hiện chính xác {new_f} lần.", {"keyword": kw, "frequency": new_f, "relation": "exactly"}
    else:  # disjoint
        return f"Tuyệt đối không được chứa từ khóa '{kw}' trong toàn bộ bài viết.", {"forbidden_words": [kw]}


def perturb_capital_words_vi(kwargs: Dict[str, Any], mode: str) -> Tuple[str, Dict[str, Any]]:
    n = kwargs.get("capital_words", 3)
    if mode == "containment":
        new_n = max(1, n - 1)
        return f"Trong bài viết phải chứa ít nhất {new_n} từ được VIẾT HOA TOÀN BỘ.", {"capital_words": new_n}
    elif mode == "partial_overlap":
        new_n = n + 2
        return f"Trong bài viết phải chứa ít nhất {new_n} từ được VIẾT HOA TOÀN BỘ.", {"capital_words": new_n}
    else:  # disjoint
        return "Toàn bộ phản hồi phải được viết bằng chữ thường (không sử dụng chữ in hoa).", {}


def perturb_generic_vi(inst_id: str, kwargs: Dict[str, Any], mode: str) -> Tuple[str, Dict[str, Any]]:
    """Bộ xử lý đột biến chung cho các constraint khác."""
    if "number" in inst_id or "count" in inst_id:
        for k, v in kwargs.items():
            if isinstance(v, int):
                delta = -1 if mode == "containment" else (1 if mode == "partial_overlap" else 3)
                new_v = max(1, v + delta)
                new_kwargs = copy.deepcopy(kwargs)
                new_kwargs[k] = new_v
                return f"Điều chỉnh yêu cầu số lượng ({k}) thành {new_v}.", new_kwargs

    if mode == "containment":
        return "Nới lỏng phạm vi yêu cầu định dạng.", kwargs
    elif mode == "partial_overlap":
        return "Mở rộng yêu cầu bổ sung liên quan.", kwargs
    else:
        return "Đảo ngược điều kiện ràng buộc đối lập.", kwargs


PERTURB_DISPATCH = {
    "length_constraints:number_words": perturb_number_words_vi,
    "length_constraints:number_sentences": perturb_number_sentences_vi,
    "length_constraints:number_paragraphs": perturb_number_paragraphs_vi,
    "detectable_format:number_bullet_lists": perturb_bullet_lists_vi,
    "format:numbered_list": perturb_bullet_lists_vi,
    "keywords:existence": perturb_keywords_existence_vi,
    "keywords:frequency": perturb_keywords_frequency_vi,
    "change_case:capital_word_frequency": perturb_capital_words_vi,
}


def apply_crpl_perturbation(
    record: Dict[str, Any],
    target_rel: str
) -> Optional[Dict[str, Any]]:
    """Tạo 1 biến thể đột biến từ mẫu gốc theo quan hệ target_rel (containment / partial_overlap / disjoint)."""
    hard_info = record.get("hard_constraints", {})
    inst_ids = hard_info.get("constraint_ids", [])
    desc_list = hard_info.get("constraints_description", [])
    ground_truth = hard_info.get("ground_truth", [{}])[0]
    kwargs_list = ground_truth.get("kwargs", [])

    if not inst_ids or not desc_list:
        return None

    # Chọn 1 constraint ngẫu nhiên để đột biến
    chosen_idx = random.randrange(len(inst_ids))
    cid = inst_ids[chosen_idx]
    orig_kwargs = kwargs_list[chosen_idx] if chosen_idx < len(kwargs_list) else {}
    orig_desc = desc_list[chosen_idx]

    perturb_fn = PERTURB_DISPATCH.get(cid, lambda kw, m: perturb_generic_vi(cid, kw, m))
    new_desc, new_kw = perturb_fn(orig_kwargs, target_rel)

    # Clone và cập nhật
    new_inst_ids = list(inst_ids)
    new_desc_list = list(desc_list)
    new_kwargs_list = [copy.deepcopy(kw) for kw in kwargs_list]

    new_desc_list[chosen_idx] = norm_vi(new_desc)
    new_kwargs_list[chosen_idx] = new_kw

    # Đổi id nếu disjoint chuyển sang forbidden
    if target_rel == "disjoint" and "forbidden_words" in new_kw and cid == "keywords:existence":
        new_inst_ids[chosen_idx] = "keywords:forbidden_words"
    elif target_rel == "disjoint" and cid == "change_case:capital_word_frequency":
        new_inst_ids[chosen_idx] = "change_case:english_lowercase"

    # Rebuild Prompt 2 tầng
    base_text = record.get("base_instruction", "")
    hard_bullets = "\n".join([f"- {d}" for d in new_desc_list])

    soft_constraints = record.get("soft_constraints", {}).get("constraints", [])
    soft_bullets = "\n".join([f"- {sc['description']}" for sc in soft_constraints]) if soft_constraints else ""

    if soft_bullets:
        rebuilt_prompt = (
            f"{base_text}\n\n"
            f"Vui lòng tuân thủ nghiêm ngặt các ràng buộc sau trong câu trả lời:\n"
            f"[Ràng buộc Định dạng & Cấu trúc]\n{hard_bullets}\n\n"
            f"[Hướng dẫn Nội dung & Ngữ nghĩa]\n{soft_bullets}"
        )
    else:
        rebuilt_prompt = (
            f"{base_text}\n\n"
            f"Vui lòng tuân thủ nghiêm ngặt các ràng buộc sau trong câu trả lời:\n"
            f"{hard_bullets}"
        )

    perturbed_key = f"{record.get('key', 'sample')}_crpl_{target_rel}"

    variant = {
        "id": record.get("id"),
        "key": perturbed_key,
        "dataset_source": record.get("dataset_source", ""),
        "constraint_split": record.get("constraint_split", "seen"),
        "base_instruction": base_text,
        "prompt": norm_vi(rebuilt_prompt),
        "num_constraints": {
            "total": len(new_inst_ids) + len(soft_constraints),
            "hard_count": len(new_inst_ids),
            "soft_count": len(soft_constraints)
        },
        "hard_constraints": {
            "verifier_type": "python_rule_engine",
            "constraint_ids": new_inst_ids,
            "constraints_description": new_desc_list,
            "ground_truth": [{
                "instruction_id": new_inst_ids,
                "kwargs": new_kwargs_list
            }]
        },
        "soft_constraints": record.get("soft_constraints", {}),
        "reward_spec": record.get("reward_spec", {
            "aggregation_strategy": "hard_priority_gated",
            "weights": {"w_hard": 0.75, "w_soft": 0.25},
            "formula": "Reward = (0.75 * R_hard + 0.25 * R_soft) if (R_hard > 0 and R_soft >= 0.5) else 0.0"
        }),
        "messages": [{"role": "user", "content": norm_vi(rebuilt_prompt)}],
        "crpl_metadata": {
            "relationship": target_rel,
            "perturbed_constraint_index": chosen_idx,
            "original_constraint_id": cid,
            "perturbed_constraint_id": new_inst_ids[chosen_idx],
            "original_description": orig_desc,
            "perturbed_description": new_desc
        },
        "created_at": datetime.datetime.now().isoformat()
    }
    return variant


def main():
    parser = argparse.ArgumentParser(description="Module 3: Sinh Biến Thể CRPL Final (> 16 mẫu / batch).")
    parser.add_argument("--input-dir", type=str, default="results/module_soft_constraints",
                        help="Thư mục chứa các file batch từ Module 2 (hoặc file json/jsonl)")
    parser.add_argument("--output-dir", type=str, default="results/module_final",
                        help="Thư mục lưu các file batch JSON kết quả Module 3")
    parser.add_argument("--split", type=str, default="seen", choices=["seen", "unseen"],
                        help="Tập ràng buộc: 'seen' hoặc 'unseen'")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Số lượng mẫu gốc trong mỗi batch (mặc định: 16 mẫu gốc)")
    parser.add_argument("--id", type=str, default=None,
                        help="Chạy duy nhất 1 mẫu theo ID số nguyên hoặc key")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Chỉ mục bắt đầu duyệt trong danh sách mẫu")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Chỉ mục kết thúc duyệt trong danh sách mẫu")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Giới hạn số mẫu gốc tối đa cần xử lý")
    parser.add_argument("--resume", action="store_true",
                        help="Tiếp tục từ mẫu/batch chưa xử lý")
    parser.add_argument("--save-unified", action="store_true",
                        help="Đồng thời lưu file gộp vào datasets/seen/crpl_task2_hybrid_vi.jsonl")
    parser.add_argument("--no-crpl", action="store_true",
                        help="LƯỢC BỎ CRPL: Giữ nguyên 1:1 số mẫu (16 mẫu/batch), không sinh 3 biến thể đột biến, đóng gói prompt 2 tầng hoàn chỉnh.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Đọc dữ liệu từ input-dir
    all_input_samples = []
    if os.path.isdir(args.input_dir):
        batch_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".json") and f.startswith("module2_soft_batch_")])
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

    print(f"📖 Đã nạp {len(all_input_samples)} mẫu đầu vào từ '{args.input_dir}'.")

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
    print(f"⚙️ Xử lý {len(selected_samples)} mẫu gốc qua CRPL Perturbation.")

    # Kiểm tra key đã có nếu --resume
    processed_keys = set()
    if args.resume:
        for fname in os.listdir(args.output_dir):
            if fname.endswith(".json") and fname.startswith("module3_final_batch_"):
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
        print(f"🔄 [RESUME] Đã phát hiện {len(processed_keys)} mẫu đã tồn tại trong kết quả final.")

    batch_size = args.batch_size
    current_expanded_batch = []
    seed_count_in_batch = 0
    batch_idx = 1
    total_saved = 0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.no_crpl:
        unified_file = f"datasets/grpo/train_seen.jsonl" if args.split == "seen" else f"datasets/eval/test_unseen.jsonl"
    else:
        unified_file = f"datasets/{args.split}/crpl_task2_hybrid_vi.jsonl" if args.split == "seen" else f"datasets/{args.split}/crpl_task2_eval_unseen_vi.jsonl"

    if args.save_unified:
        os.makedirs(os.path.dirname(unified_file), exist_ok=True)

    for item in selected_samples:
        key = item.get("key", f"sample_{item.get('id', 0)}")
        if args.resume and key in processed_keys:
            continue

        # 1. Mẫu gốc (chuẩn hóa prompt nếu đã có soft)
        base_record = copy.deepcopy(item)
        if "reference_response" in base_record:
            del base_record["reference_response"]  # GRPO không cần đáp án mẫu

        if not args.no_crpl:
            base_record["crpl_metadata"] = {"relationship": "original"}
            current_expanded_batch.append(base_record)

            # 2. Sinh 3 biến thể: Containment, Partial Overlap, Disjoint
            for rel in ["containment", "partial_overlap", "disjoint"]:
                var = apply_crpl_perturbation(item, target_rel=rel)
                if var:
                    current_expanded_batch.append(var)
        else:
            if "crpl_metadata" in base_record:
                del base_record["crpl_metadata"]
            current_expanded_batch.append(base_record)

        seed_count_in_batch += 1

        # Khi gom đủ batch_size mẫu thì lưu file batch
        if seed_count_in_batch >= batch_size:
            prefix = "module3_final_batch" if not args.no_crpl else "module_final_batch"
            batch_filename = f"{prefix}_{batch_idx:03d}_{timestamp}.json"
            batch_filepath = os.path.join(args.output_dir, batch_filename)
            with open(batch_filepath, "w", encoding="utf-8") as f_b:
                json.dump(current_expanded_batch, f_b, ensure_ascii=False, indent=2)

            if args.save_unified:
                with open(unified_file, "a", encoding="utf-8") as f_uni:
                    for rec in current_expanded_batch:
                        f_uni.write(json.dumps(rec, ensure_ascii=False) + "\n")

            tag = "CRPL Mở rộng" if not args.no_crpl else "Đóng gói No-CRPL (16 mẫu)"
            print(f"📦 [Module 3 ({tag}) - Batch {batch_idx:03d}] Đã lưu {len(current_expanded_batch)} mẫu vào '{batch_filepath}'.")
            total_saved += len(current_expanded_batch)
            current_expanded_batch = []
            seed_count_in_batch = 0
            batch_idx += 1

    # Lưu nốt batch cuối
    if current_expanded_batch:
        prefix = "module3_final_batch" if not args.no_crpl else "module_final_batch"
        batch_filename = f"{prefix}_{batch_idx:03d}_{timestamp}.json"
        batch_filepath = os.path.join(args.output_dir, batch_filename)
        with open(batch_filepath, "w", encoding="utf-8") as f_b:
            json.dump(current_expanded_batch, f_b, ensure_ascii=False, indent=2)

        if args.save_unified:
            with open(unified_file, "a", encoding="utf-8") as f_uni:
                for rec in current_expanded_batch:
                    f_uni.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tag = "CRPL Mở rộng" if not args.no_crpl else "Đóng gói No-CRPL (16 mẫu)"
        print(f"📦 [Module 3 ({tag}) - Batch {batch_idx:03d} (Cuối)] Đã lưu {len(current_expanded_batch)} mẫu vào '{batch_filepath}'.")
        total_saved += len(current_expanded_batch)

    print("\n" + "=" * 60)
    title = "HOÀN THÀNH ĐÓNG GÓI DATASET (NO-CRPL - 2 TASK VERIF)" if args.no_crpl else "HOÀN THÀNH MODULE 3 (CRPL Final - Mở rộng biến thể)"
    print(f"🎉 {title}:")
    print(f"   • Tổng số mẫu hoàn chỉnh đã lưu : {total_saved} mẫu")
    print(f"   • Số batch hoàn thành           : {batch_idx}")
    print(f"   • Thư mục kết quả               : {args.output_dir}")
    if args.save_unified:
        print(f"   • File gộp chuẩn                : {unified_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
