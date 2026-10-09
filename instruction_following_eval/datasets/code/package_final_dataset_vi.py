#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 3 (Đóng gói không dùng CRPL): package_final_dataset_vi.py
Mục đích: Đóng gói tập dữ liệu cuối cùng từ Module 2 (Hard + Soft Constraints)
          theo đúng 2 Task Mentor giao, LƯỢC BỎ THUẬT TOÁN ĐỘT BIẾN CRPL.

Tính năng:
- Tiếp nhận các file batch 16 mẫu từ Module 2 (results/module_soft_constraints/).
- Chỉ đóng gói mẫu có rubric được kiểm định; lưu các mục bị loại riêng.
- Ghép prompt hoàn chỉnh 2 tầng:
    Tầng 1: Ràng buộc Cấu trúc & Định dạng (Hard Constraints - 29 Seen / 24 Unseen)
    Tầng 2: Hướng dẫn Phong cách & Ngữ nghĩa (Soft Constraints - 5 Seen / 7 Unseen)
- Xuất dữ liệu:
    + Batch files: 'results/module_final/module_final_batch_{id}_{timestamp}.json' (16 mẫu/file)
    + GRPO train: 'datasets/grpo/train_seen.jsonl' (chỉ chứa prompt + ground_truth verifiers)
    + Eval In-Domain: 'datasets/eval/test_seen.jsonl'
    + Eval Out-of-Domain: 'datasets/eval/test_unseen.jsonl'
- Hỗ trợ các cờ: --id, --start-idx, --end-idx, --num-samples, --resume, --batch-size 16, --split (seen/unseen).
"""

import os
import sys
import json
import copy
import argparse
import datetime
import unicodedata
from typing import List, Dict, Any, Optional
from soft_constraints_policy import select_row_soft_constraints, soft_reward_spec


def norm_vi(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def build_final_prompt_vi(base_text: str, hard_descs: List[str], soft_constraints: List[Dict[str, Any]]) -> str:
    """Ghép prompt tiếng Việt 2 tầng rõ ràng."""
    hard_bullets = "\n".join([f"- {desc}" for desc in hard_descs]) if hard_descs else ""
    soft_bullets = "\n".join([f"- {sc.get('description') or sc.get('description_vi', '')}" for sc in soft_constraints if (sc.get("description") or sc.get("description_vi"))])

    if hard_bullets and soft_bullets:
        final_prompt = (
            f"{base_text}\n\n"
            f"Vui lòng tuân thủ nghiêm ngặt các ràng buộc sau trong câu trả lời:\n"
            f"[RÀNG BUỘC CẤU TRÚC & ĐỊNH DẠNG]\n"
            f"{hard_bullets}\n\n"
            f"[HƯỚNG DẪN PHONG CÁCH & NGỮ NGHĨA]\n"
            f"{soft_bullets}"
        )
    elif hard_bullets:
        final_prompt = (
            f"{base_text}\n\n"
            f"Vui lòng tuân thủ nghiêm ngặt các ràng buộc sau trong câu trả lời:\n"
            f"{hard_bullets}"
        )
    else:
        final_prompt = base_text + ("\n\n[HƯỚNG DẪN PHONG CÁCH & NGỮ NGHĨA]\n" + soft_bullets if soft_bullets else "")

    return norm_vi(final_prompt.strip())


def main():
    parser = argparse.ArgumentParser(description="Đóng gói Final Dataset (No-CRPL, chuẩn 2-Task Mentor).")
    parser.add_argument("--input-dir", type=str, default="results/module_soft_constraints",
                        help="Thư mục chứa kết quả Module 2 (hoặc Module 1)")
    parser.add_argument("--output-dir", type=str, default="results/module_final",
                        help="Thư mục lưu batch JSON (16 mẫu/file)")
    parser.add_argument("--split", type=str, default="seen", choices=["seen", "unseen"],
                        help="'seen' (Train & Test Seen) hoặc 'unseen' (Test OOD)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Số mẫu trong mỗi batch (mặc định: 16)")
    parser.add_argument("--id", type=str, default=None,
                        help="Chạy duy nhất 1 mẫu theo ID hoặc key")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="Chỉ mục bắt đầu duyệt")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Chỉ mục kết thúc duyệt")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Giới hạn số mẫu cần đóng gói")
    parser.add_argument("--resume", action="store_true",
                        help="Tiếp tục từ các mẫu chưa xử lý")
    parser.add_argument("--save-unified", action="store_true",
                        help="Đồng thời xuất file jsonl vào datasets/grpo hoặc datasets/eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Đọc dữ liệu đầu vào
    all_samples = []
    if os.path.isdir(args.input_dir):
        files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".json") and ("module2_soft_batch_" in f or "module1_hard_batch_" in f)])
        for fname in files:
            fpath = os.path.join(args.input_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f_in:
                    d = json.load(f_in)
                    if isinstance(d, list):
                        all_samples.extend(d)
            except Exception as e:
                print(f"⚠️ Lỗi đọc file {fname}: {e}")
    elif os.path.isfile(args.input_dir):
        with open(args.input_dir, "r", encoding="utf-8") as f_in:
            if args.input_dir.endswith(".jsonl"):
                for line in f_in:
                    if line.strip():
                        all_samples.append(json.loads(line))
            else:
                all_samples = json.load(f_in)

    # Tự động lọc trùng lặp theo key
    seen_keys = set()
    deduped_samples = []
    for s in all_samples:
        k = s.get("key") or str(s.get("id"))
        if k not in seen_keys:
            seen_keys.add(k)
            deduped_samples.append(s)
    all_samples = deduped_samples

    print(f"📖 Đã nạp {len(all_samples)} mẫu duy nhất từ '{args.input_dir}'.")

    # 2. Lọc theo --id
    if args.id:
        target = str(args.id).strip()
        all_samples = [s for s in all_samples if str(s.get("id")) == target or s.get("key") == target]
        print(f"🎯 Lọc theo --id '{args.id}': tìm thấy {len(all_samples)} mẫu.")

    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(all_samples)
    if args.num_samples is not None:
        end_idx = min(end_idx, start_idx + args.num_samples)

    selected = all_samples[start_idx:end_idx]
    print(f"⚙️ Bắt đầu đóng gói {len(selected)} mẫu (chế độ No-CRPL, 16 mẫu/batch)...")

    # Resume check
    processed_keys = set()
    if args.resume:
        for f in os.listdir(args.output_dir):
            if f.endswith(".json") and f.startswith("module_final_batch_"):
                fp = os.path.join(args.output_dir, f)
                try:
                    with open(fp, "r", encoding="utf-8") as fb:
                        for it in json.load(fb):
                            if "key" in it:
                                processed_keys.add(it["key"])
                except Exception:
                    pass
        print(f"🔄 [RESUME] Đã phát hiện {len(processed_keys)} mẫu đã tồn tại.")

    # Chuẩn bị file gộp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.split == "seen":
        unified_grpo = "datasets/grpo/train_seen.jsonl"
        unified_eval = "datasets/eval/test_seen.jsonl"
    else:
        unified_grpo = None
        unified_eval = "datasets/eval/test_unseen.jsonl"

    if args.save_unified:
        if unified_grpo:
            os.makedirs(os.path.dirname(unified_grpo), exist_ok=True)
        if unified_eval:
            os.makedirs(os.path.dirname(unified_eval), exist_ok=True)

    current_batch = []
    batch_idx = 1
    total_saved = 0
    quarantined = []

    for item in selected:
        key = item.get("key", f"sample_{item.get('id', 0)}")
        if args.resume and key in processed_keys:
            continue

        base_inst = item.get("base_instruction", item.get("instruction_vi", item.get("instruction", "")))
        hard_data = item.get("hard_constraints", {})
        hard_descs = hard_data.get("constraints_description", [])
        soft_list, excluded_soft = select_row_soft_constraints(item, split=args.split)
        if excluded_soft:
            quarantined.append({"key": key, "excluded": excluded_soft, "record": item,
                                "whole_record_excluded": not soft_list})
        if not soft_list:
            print(f"Quarantine {key}: không có soft constraint được kiểm định hợp lệ.")
            continue

        # Tạo prompt hoàn chỉnh
        full_prompt = build_final_prompt_vi(base_inst, hard_descs, soft_list)

        final_record = copy.deepcopy(item)
        final_record["prompt"] = full_prompt
        final_record["hybrid_prompt"] = full_prompt
        final_record["messages"] = [{"role": "user", "content": full_prompt}]
        final_record["instruction_vi"] = item.get("instruction_vi", "")
        final_record["input_vi"] = item.get("input_vi", "")
        final_record["updated_at"] = datetime.datetime.now().isoformat()
        final_record["soft_constraints"] = {"verifier_type": "llm_reasoning_judge", "constraints": soft_list}
        final_record["reward_spec"] = soft_reward_spec()
        final_record["num_constraints"] = {"hard_count": len(hard_descs), "soft_count": len(soft_list),
                                           "total": len(hard_descs) + len(soft_list)}
        final_record["soft_quality"] = {"excluded": excluded_soft, "ready_for_hybrid_training": True}

        # Chuẩn hóa cấu trúc verifier thành list phẳng [{"id": ..., "kwargs": ...}]
        verifier_list = []
        gt_entries = hard_data.get("ground_truth", [])
        for gt in gt_entries:
            if isinstance(gt, dict) and "instruction_id" in gt and "kwargs" in gt:
                inst_ids = gt.get("instruction_id", [])
                kwargs_list = gt.get("kwargs", [])
                for i_id, kw in zip(inst_ids, kwargs_list):
                    verifier_list.append({"id": i_id, "kwargs": kw})
            elif isinstance(gt, dict) and "id" in gt:
                verifier_list.append(gt)

        # Chuẩn hóa soft constraints giữ đầy đủ id, description, rubric
        soft_clean_list = soft_list  # Giữ category và provenance kiểm định cho GRPO.

        # Gán verifier và soft dạng JSON string cho cả final_record để khớp với GRPO format
        final_record["verifier"] = json.dumps(verifier_list, ensure_ascii=False)
        final_record["soft"] = json.dumps(soft_clean_list, ensure_ascii=False)

        # Dòng định dạng chuẩn cho GRPOTrainer (Zero-SFT)
        grpo_row = {
            "key": key,
            "base_instruction": base_inst,
            "constraint_split": args.split,
            "prompt": [{"role": "user", "content": full_prompt}],
            "verifier": json.dumps(verifier_list, ensure_ascii=False),
            "soft": json.dumps(soft_clean_list, ensure_ascii=False),
            "num_hard": len(verifier_list),
            "reward_spec": soft_reward_spec(),
            "soft_quality": {"ready_for_hybrid_training": True},
            "base_pass_rate": None
        }

        current_batch.append((final_record, grpo_row))

        if len(current_batch) >= args.batch_size:
            b_name = f"module_final_batch_{batch_idx:03d}_{timestamp}.json"
            b_path = os.path.join(args.output_dir, b_name)
            batch_records = [r[0] for r in current_batch]
            with open(b_path, "w", encoding="utf-8") as f_b:
                json.dump(batch_records, f_b, ensure_ascii=False, indent=2)

            if args.save_unified:
                if unified_grpo:
                    with open(unified_grpo, "a", encoding="utf-8") as f_ug:
                        for _, gr in current_batch:
                            f_ug.write(json.dumps(gr, ensure_ascii=False) + "\n")
                if unified_eval:
                    with open(unified_eval, "a", encoding="utf-8") as f_ue:
                        for r, _ in current_batch:
                            f_ue.write(json.dumps(r, ensure_ascii=False) + "\n")

            print(f"📦 [Batch {batch_idx:03d}] Đã lưu {len(current_batch)} mẫu vào '{b_path}'.")
            total_saved += len(current_batch)
            current_batch = []
            batch_idx += 1

    if quarantined:
        with open(os.path.join(args.output_dir, f"soft_quarantine_{timestamp}.json"), "w", encoding="utf-8") as f_q:
            json.dump(quarantined, f_q, ensure_ascii=False, indent=2)

    if current_batch:
        b_name = f"module_final_batch_{batch_idx:03d}_{timestamp}.json"
        b_path = os.path.join(args.output_dir, b_name)
        batch_records = [r[0] for r in current_batch]
        with open(b_path, "w", encoding="utf-8") as f_b:
            json.dump(batch_records, f_b, ensure_ascii=False, indent=2)

        if args.save_unified:
            if unified_grpo:
                with open(unified_grpo, "a", encoding="utf-8") as f_ug:
                    for _, gr in current_batch:
                        f_ug.write(json.dumps(gr, ensure_ascii=False) + "\n")
            if unified_eval:
                with open(unified_eval, "a", encoding="utf-8") as f_ue:
                    for r, _ in current_batch:
                        f_ue.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"📦 [Batch {batch_idx:03d} (Cuối)] Đã lưu {len(current_batch)} mẫu vào '{b_path}'.")
        total_saved += len(current_batch)

    print("\n" + "=" * 60)
    print(f"🎉 HOÀN THÀNH ĐÓNG GÓI FINAL DATASET (NO-CRPL, 2-TASK VERIF):")
    print(f"   • Tổng số mẫu hoàn chỉnh : {total_saved} mẫu")
    print(f"   • Số batch hoàn thành     : {batch_idx}")
    print(f"   • Thư mục kết quả         : {args.output_dir}")
    if args.save_unified:
        if unified_grpo:
            print(f"   • File GRPO train         : {unified_grpo}")
        if unified_eval:
            print(f"   • File Eval benchmark     : {unified_eval}")
    print("=" * 60)


if __name__ == "__main__":
    main()
