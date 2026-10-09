"""Offline rules shared by dataset generation, packaging and notebook exports.

Approval means an explicit successful verification or documented offline curation;
it never means a failed API request was accepted. Human judge calibration remains
a separate measurement.
"""
import copy
import json

SOFT_POLICY_VERSION = "soft-v2"
SEEN_SOFT_CATEGORIES = (
    "semantic_completeness", "style_and_tone", "clarity_and_coherence",
    "conciseness_and_efficiency", "practical_examples",
)
UNSEEN_SOFT_CATEGORIES = (
    "target_audience", "reasoning_and_logic", "role_play_persona",
    "counterfactual_context", "safety_and_neutrality", "pedagogical_analogy",
    "critique_and_limitations",
)
ACCEPTED_SOFT_STATUSES = ("PASS", "REVIEWED_OFFLINE")
SOFT_CATEGORY_GUIDANCE = {
    "semantic_completeness": "Các ý bắt buộc phải được giải thích đúng và liên quan tới đề bài; chỉ nhắc từ khóa hoặc đưa thông tin sai không được tính là bao phủ nội dung.",
    "style_and_tone": "Dựa vào cách diễn đạt thực tế: rõ ràng, lịch sự, phù hợp phong cách được yêu cầu. Không đòi mô hình tự nhận là chuyên gia, không thưởng chỉ vì có thuật ngữ.",
    "clarity_and_coherence": "Các ý liên kết và không tự mâu thuẫn; người đọc theo dõi được trình tự. Không bắt buộc câu nối hay hình thức riêng nếu description không yêu cầu.",
    "conciseness_and_efficiency": "Không lặp lại cùng ý, không có lời dẫn/đệm không phục vụ nhiệm vụ. Độ dài cần để đáp ứng luật cứng không tự động bị coi là dài dòng.",
    "practical_examples": "Ví dụ hoặc công cụ phải cụ thể, liên quan và được áp dụng đúng vào nhiệm vụ; danh sách tên không liên quan không đủ đạt.",
    "target_audience": "Giải thích thuật ngữ cần thiết, dùng cách diễn đạt và độ sâu phù hợp đối tượng. Không cấm mọi thuật ngữ chuyên môn khi thuật ngữ được giải thích đúng.",
    "reasoning_and_logic": "Các bước hoặc quan hệ nhân quả phải đúng, nối được dữ kiện với kết luận; không chỉ liệt kê nhãn các bước hoặc thuật ngữ.",
    "role_play_persona": "Vai diễn thể hiện trong cách phân tích, lựa chọn thông tin và cách giao tiếp đúng nhiệm vụ. Không đủ đạt chỉ vì tự xưng vai diễn hay chèn từ chuyên môn.",
    "counterfactual_context": "Tuân thủ giả định được công bố xuyên suốt và suy luận nhất quán trong giả định đó. Không bác bỏ giả định vì khác thực tế và không tự thêm giả định khác.",
    "safety_and_neutrality": "Không bịa dữ kiện, quy kết hoặc khẳng định tuyệt đối thiếu căn cứ; thể hiện các góc nhìn mà description yêu cầu, không ép cân bằng giả giữa thông tin đúng và sai.",
    "pedagogical_analogy": "Ẩn dụ/ví dụ phải có sự tương ứng đúng với khái niệm và giúp giải thích; chỉ nhắc một hình ảnh so sánh không liên quan không đủ đạt.",
    "critique_and_limitations": "Nêu rủi ro hoặc điều kiện biên cụ thể gắn với giải pháp, giải thích vì sao có giới hạn; cảnh báo chung chung không đủ đạt.",
}


def soft_reward_spec(gate="all_hard"):
    if gate not in ("all_hard", "partial_hard"):
        raise ValueError("Soft gate must be all_hard or partial_hard")
    return {
        "version": SOFT_POLICY_VERSION,
        "aggregation_strategy": "hard_priority_gated",
        "hard_partial_weight": 0.70,
        "hard_all_pass_weight": 0.30,
        "soft_bonus_weight": 0.25,
        "soft_gate": gate,
        "maximum_reward": 1.25,
        "soft_aggregation": "mean_binary_per_criterion",
        "formula": "0.70 * hard_partial + 0.30 * all_hard_pass + 0.25 * soft_gate_pass * soft_score",
    }


def normalize_soft_constraints(values, require_verified=True, split=None):
    """Return accepted, validated criteria and rejected entries with reasons.

    Never let a flat JSON field bypass a rejected nested verification record;
    callers must merge that provenance before calling this function.
    """
    values = json.loads(values) if isinstance(values, str) else values
    if not isinstance(values, list) or not all(isinstance(v, dict) for v in values):
        raise ValueError("Soft constraints phải là list object JSON.")
    accepted, excluded, ids = [], [], set()
    allowed = set(SEEN_SOFT_CATEGORIES if split == "seen" else
                  UNSEEN_SOFT_CATEGORIES if split == "unseen" else
                  SEEN_SOFT_CATEGORIES + UNSEEN_SOFT_CATEGORIES)
    for original in values:
        item = copy.deepcopy(original)
        ident = item.get("id")
        if not isinstance(ident, str) or not ident.strip() or ident in ids:
            raise ValueError("Soft constraint id thiếu hoặc trùng.")
        ids.add(ident)
        status = item.get("verification_status", "UNVERIFIED")
        if status == "REJECT" or (require_verified and status not in ACCEPTED_SOFT_STATUSES):
            excluded.append({"constraint": item, "reason": f"verification_status={status}"})
            continue
        category = item.get("category") or ident.split(":", 1)[0]
        if category not in allowed:
            raise ValueError(f"Soft category không hợp lệ cho split={split}: {category}")
        description = item.get("description") or item.get("description_vi")
        rubric = item.get("rubric") or item.get("eval_rubric")
        if not isinstance(description, str) or not description.strip() or not isinstance(rubric, str) or not rubric.strip():
            raise ValueError(f"Description/rubric rỗng: {ident}")
        accepted.append({
            "id": ident, "category": category,
            "description": description.strip(), "rubric": rubric.strip(),
            "evaluation_guidance": item.get("evaluation_guidance") or SOFT_CATEGORY_GUIDANCE[category],
            "verification_status": status,
            "verified_by": item.get("verified_by", "unknown"),
            "policy_version": SOFT_POLICY_VERSION,
        })
    return accepted, excluded


def select_row_soft_constraints(row, require_verified=True, split=None):
    nested = row.get("soft_constraints", {}).get("constraints", [])
    raw = row.get("soft")
    raw = json.loads(raw) if isinstance(raw, str) else raw
    if raw is None:
        raw = nested
    if not isinstance(raw, list) or not all(isinstance(v, dict) for v in raw):
        raise ValueError("Soft constraints phải là list object JSON.")
    # Verification status belongs to the full dataset, not to the stripped field.
    provenance = {c.get("id"): c for c in nested if isinstance(c, dict)}
    merged = []
    for value in raw:
        item = dict(value)
        origin = provenance.get(item.get("id"), {})
        origin_status = origin.get("verification_status")
        if origin_status is not None and origin_status not in ACCEPTED_SOFT_STATUSES:
            item["verification_status"] = origin_status
        else:
            for key in ("verification_status", "verified_by", "category"):
                if key not in item and key in origin:
                    item[key] = origin[key]
        merged.append(item)
    return normalize_soft_constraints(merged, require_verified, split)
