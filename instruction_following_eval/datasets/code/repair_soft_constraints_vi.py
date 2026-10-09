#!/usr/bin/env python3
"""Offline migration of the three packaged datasets to soft-v2.

No judge requests. Original bytes and every exclusion/revision are preserved.
REVIEWED_OFFLINE documents editorial review; it is not human judge calibration.
Run with --apply to write; without it only a summary is printed.
"""
import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from soft_constraints_policy import (SOFT_POLICY_VERSION, normalize_soft_constraints,
                                     select_row_soft_constraints, soft_reward_spec)
from package_final_dataset_vi import build_final_prompt_vi

# Reviewed against base tasks and hard descriptions. Replace a redundant category
# instead of adding an extra criterion and changing per-prompt soft weighting.
CURATED_UNSEEN = {
    'alpaca-gpt4_0034': ('critique_and_limitations', 'counterfactual_context:no_new_equipment',
        'Giả sử gia đình không có ngân sách mua thiết bị mới. Kế hoạch tiết kiệm điện chỉ dựa trên thay đổi thói quen và sử dụng thiết bị hiện có; không đề xuất mua hoặc thay thiết bị.'),
    'alpaca-gpt4_0045': ('target_audience', 'counterfactual_context:no_fossil_fuels',
        'Giả sử nhiên liệu hóa thạch không còn khả dụng. Giải thích vai trò của năng lượng tái tạo trong bối cảnh này và đề cập hạn chế khi triển khai; không đề xuất tiếp tục sử dụng nhiên liệu hóa thạch.'),
    'alpaca-gpt4_0082': ('role_play_persona', 'counterfactual_context:protective_magic_bow',
        'Trong thế giới giả tưởng của câu chuyện, cung và tên thần kỳ chỉ tạo lá chắn hoặc bảo vệ người khác, không thể gây thương tích. Các sự kiện trong truyện phải nhất quán với giả định này.'),
    'alpaca-gpt4_0090': ('role_play_persona', 'counterfactual_context:offline_marketing',
        'Giả sử không thể quảng cáo trên Internet. Các chiến lược quảng bá ứng dụng phải sử dụng kênh ngoại tuyến; không đề xuất quảng cáo mạng xã hội, quảng cáo tìm kiếm hay tiếp thị qua email.'),
    'alpaca-gpt4_0042': ('critique_and_limitations', 'safety_and_neutrality:water_shared_responsibility',
        'Phân tích trách nhiệm và khả năng tham gia của chính quyền, doanh nghiệp và hộ gia đình trong việc giảm ô nhiễm nước; không quy kết một nhóm là nguyên nhân duy nhất khi đề bài chưa cung cấp bằng chứng.'),
    'alpaca-gpt4_0079': ('target_audience', 'safety_and_neutrality:fair_customer_resolution',
        'Thừa nhận trải nghiệm của khách hàng và đề xuất cách kiểm tra, hỗ trợ phù hợp; không đổ lỗi cho khách hàng hoặc khẳng định nguyên nhân lỗi sản phẩm khi chưa có thông tin xác minh.'),
    'alpaca-gpt4_0087': ('role_play_persona', 'safety_and_neutrality:balanced_film_review',
        'Bài đánh giá nêu cả điểm mạnh và điểm hạn chế của bộ phim bằng lập luận gắn với nội dung hoặc nghệ thuật; không dùng định kiến hay quy kết chung về một tầng lớp xã hội.'),
}


def aligned_rubric(description):
    return ('Chấm 1 nếu câu trả lời đáp ứng đúng và đầy đủ yêu cầu đã công bố: ' + description +
            ' Chấm 0 nếu bỏ sót yêu cầu bắt buộc, trình bày sai, tự mâu thuẫn hoặc chỉ nhắc từ khóa mà không thực hiện yêu cầu. '
            'Không áp thêm số lượng, định dạng hoặc điều kiện không có trong yêu cầu của đề bài. '
            'Các ví dụ ghi rõ là minh họa không phải danh sách bắt buộc; nếu đề bài không quy định số ví dụ/công cụ, một ví dụ/công cụ cụ thể phù hợp cũng có thể đủ.')


def needs_alignment(item):
    rubric, desc = item['rubric'], item['description']
    # Practical/persona criteria should measure fulfillment rather than lists of
    # keywords or undisclosed minimum counts. Threshold gaps also need a total
    # binary decision (old rubrics often leave the 3-of-4 boundary undefined).
    hidden_count = re.search(r'(?:ít nhất|tối thiểu|>=|≥)\s*(?:[2-9]|hai|ba|bốn|năm|sáu)', rubric, re.I)
    return (item['category'] in ('practical_examples', 'role_play_persona') or
            bool(hidden_count) or ('markdown' in rubric.lower() and 'markdown' not in desc.lower()))


def migrate_eval(rows, split, audit):
    result = []
    for original in rows:
        row = copy.deepcopy(original)
        accepted, excluded = select_row_soft_constraints(row, split=split)
        if excluded:
            audit['quarantine'].append({'split': split, 'key': row['key'], 'record': original,
                                        'excluded': excluded, 'whole_record_excluded': not accepted})
        if not accepted:
            continue
        for i, sc in enumerate(accepted):
            old = copy.deepcopy(sc)
            change = None
            curated = CURATED_UNSEEN.get(row['key']) if split == 'unseen' else None
            if curated and sc['category'] == curated[0]:
                _, ident, description = curated
                sc['id'], sc['category'], sc['description'] = ident, ident.split(':')[0], description
                sc.pop('evaluation_guidance', None)
                change = 'coverage_curation: explicit compatible assumption/neutrality requirement'
            elif needs_alignment(sc):
                change = 'rubric_alignment: judge only the published description; remove hidden minima and undefined binary boundaries'
            if change:
                sc['rubric'] = aligned_rubric(sc['description'])
                sc['verification_status'] = 'REVIEWED_OFFLINE'
                sc['verified_by'] = 'offline_policy_review_soft_v2'
                accepted[i] = normalize_soft_constraints([sc], split=split)[0][0]
                audit['revisions'].append({'split': split, 'key': row['key'], 'reason': change,
                                           'before': old, 'after': accepted[i]})
        row['soft_constraints'] = {'verifier_type': 'llm_reasoning_judge', 'constraints': accepted}
        row['soft'] = json.dumps(accepted, ensure_ascii=False)
        prompt = build_final_prompt_vi(row['base_instruction'], row['hard_constraints']['constraints_description'], accepted)
        if prompt != original['prompt']:
            audit['changed_prompts'].append({'split': split, 'key': row['key'],
                                              'old_sha256': hashlib.sha256(original['prompt'].encode()).hexdigest(),
                                              'new_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
        row['prompt'] = row['hybrid_prompt'] = prompt
        row['messages'] = [{'role': 'user', 'content': prompt}]
        row['reward_spec'] = soft_reward_spec()
        h = len(row['hard_constraints']['constraints_description'])
        row['num_constraints'] = {'hard_count': h, 'soft_count': len(accepted), 'total': h + len(accepted)}
        row['soft_quality'] = {'policy_version': SOFT_POLICY_VERSION, 'ready_for_hybrid_training': True}
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    names = ['datasets/grpo/train_seen.jsonl', 'datasets/eval/test_seen.jsonl', 'datasets/eval/test_unseen.jsonl']
    paths = [args.root / n for n in names]
    blobs = [p.read_bytes() for p in paths]
    originals = [[json.loads(l) for l in b.decode('utf-8').splitlines() if l.strip()] for b in blobs]
    report_path = args.root / 'datasets/soft_audit/repair_report_soft_v2.json'
    # Do not apply again: provenance from original v1 must not be overwritten.
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if all(hashlib.sha256(b).hexdigest() == report['output_sha256'][n] for n,b in zip(names,blobs)):
            print('soft-v2 đã áp dụng; dữ liệu khớp hash báo cáo, không thay đổi thêm.')
            return
        raise ValueError('Dữ liệu đã thay đổi sau migration; cần review riêng, không ghi đè bản audit.')
    audit = {'policy_version': SOFT_POLICY_VERSION, 'quarantine': [], 'revisions': [], 'changed_prompts': []}
    seen = migrate_eval(originals[1], 'seen', audit)
    unseen = migrate_eval(originals[2], 'unseen', audit)
    seen_map = {r['key']: r for r in seen}
    train = []
    for original in originals[0]:
        if original['key'] not in seen_map:
            audit['quarantine'].append({'split': 'train', 'key': original['key'], 'record': original,
                                        'reason': 'matching seen record has no approved soft rubric', 'whole_record_excluded': True})
            continue
        row = copy.deepcopy(original)
        canonical = seen_map[row['key']]
        row.update(prompt=canonical['messages'], soft=canonical['soft'], reward_spec=soft_reward_spec(),
                   base_instruction=canonical['base_instruction'], soft_quality=canonical['soft_quality'])
        train.append(row)
    outputs = [train, seen, unseen]
    data = [(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)).encode() for rows in outputs]
    audit['summary'] = {'rows_before': {n:len(r) for n,r in zip(names, originals)},
                        'rows_after': {n:len(r) for n,r in zip(names, outputs)},
                        'quarantine_records':len(audit['quarantine']), 'revised_criteria':len(audit['revisions']),
                        'changed_eval_prompts':len(audit['changed_prompts']),
                        'categories':{s:dict(Counter(c['category'] for r in rows for c in json.loads(r['soft'])))
                                      for s,rows in [('seen',seen),('unseen',unseen)]}}
    audit['input_sha256'] = {n:hashlib.sha256(b).hexdigest() for n,b in zip(names,blobs)}
    audit['output_sha256'] = {n:hashlib.sha256(b).hexdigest() for n,b in zip(names,data)}
    print(json.dumps(audit['summary'], ensure_ascii=False, indent=2))
    if not args.apply:
        return
    backup = args.root / 'datasets/soft_audit/original_v1'
    backup.mkdir(parents=True, exist_ok=False)
    for n,b in zip(names,blobs):
        (backup / Path(n).name).write_bytes(b)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist original/audit before replacing the three files.
    report_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    for path,b in zip(paths,data):
        temp = path.with_suffix('.jsonl.soft-v2.tmp')
        temp.write_bytes(b)
        temp.replace(path)


if __name__ == '__main__':
    main()
