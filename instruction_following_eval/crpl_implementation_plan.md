# 📋 PLAN: Tinh giản CRPL cho Huấn luyện GRPO với Verifiable Reward (Hỗ trợ Hybrid Hard + Soft)

---

## Mục lục
1. [Bản chất CRPL trong bối cảnh GRPO (Giữ gì & Bỏ gì?)](#1-bản-chất-crpl-trong-bối-cảnh-grpo-giữ-gì--bỏ-gì)
2. [Lộ trình Triển khai Tinh giản (Hỗ trợ trực tiếp Hybrid Dataset)](#2-lộ-trình-triển-khai-tinh-giản-hỗ-trợ-trực-tiếp-hybrid-dataset)
3. [Format JSON chuẩn cho môi trường GRPO](#3-format-json-chuẩn-cho-môi-trường-grpo)
4. [Kiến trúc Pipeline GRPO tổng thể](#4-kiến-trúc-pipeline-grpo-tổng-thể)
5. [So sánh: Pipeline ban đầu (DPO) vs Pipeline tinh giản (GRPO)](#5-so-sánh-pipeline-ban-đầu-dpo-vs-pipeline-tinh-giản-grpo)
6. [Kế hoạch thực hiện tiếp theo](#6-kế-hoạch-thực-hiện-tiếp-theo)

---

## 1. Bản chất CRPL trong bối cảnh GRPO (Giữ gì & Bỏ gì?)

### 1.1 Phân tích cốt lõi
* **CRPL trong bài báo gốc:** Được thiết kế chủ yếu cho **DPO (Direct Preference Optimization)** hoặc các thuật toán pairwise offline. Vì DPO không có môi trường tương tác online, nó bắt buộc phải:
  1. Dùng LLM sinh trước $K$ câu trả lời offline.
  2. Dùng rule engine phân loại câu trả lời vào các vùng giao thoa $S_1, S_2, S_3$.
  3. Ghép cặp nhân tạo `(prompt, chosen, rejected)` để tối ưu hóa hàm cross-entropy loss của DPO.
* **Pipeline thực tế của chúng ta:** Sử dụng **GRPO (Group Relative Policy Optimization)** với bộ Verifiable Reward (Hard rule engine + Soft LLM judge).
  * Trong GRPO, policy tự thực hiện **online rollouts** một nhóm $G$ responses $\{o_1, o_2, \dots, o_G\} \sim \pi_\theta(x)$ cho mỗi prompt $x$.
  * Verifier chấm điểm trực tiếp từng response trong nhóm: $r_i = \text{Reward}(x, o_i)$.
  * Advantage được chuẩn hóa tương đối ngay trong nhóm:
    $$\tilde{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$$
  * 👉 **GRPO đã tự động so sánh các ứng viên trong group theo thời gian thực**, do đó **hoàn toàn không cần** việc sinh response trước và chia cặp $S_1/S_2/S_3$ offline!

### 1.2 Bảng quyết định: Giữ gì và Bỏ gì từ CRPL

| Thành phần CRPL | Trạng thái | Đánh giá & Lý do kỹ thuật |
|---|:---:|---|
| **Sinh biến thể constraint (Containment / Overlap / Disjoint)** | ✅ **GIỮ (100%)** | **Cực kỳ giá trị**: Sinh bằng Python rule rẻ, nhanh, không tốn GPU/API. Tạo ra không gian prompt phong phú, đa dạng hóa tham số và ép model học được độ nhạy biên (boundary sensitivity), chống học vẹt. |
| **Sinh $K$ responses offline, phân vùng $S_1/S_2/S_3$, ghép cặp preference** | ❌ **BỎ HOÀN TOÀN** | **Thừa thãi & Lãng phí**: GRPO là online RL, tự sinh group response khi train. Bỏ bước này giúp **tiết kiệm 100% chi phí API/GPU** và tránh tạo ra các cặp dữ liệu offline cồng kềnh. |
| **Tích hợp Soft Constraints (Hybrid Task 2)** | ✅ **GIỮ** | Đóng vai trò là thành phần reward thứ hai trong GRPO: $R = \alpha R_{\text{hard}} + \beta R_{\text{soft}}$, đảm bảo cả tính đúng đắn logic lẫn văn phong. |
| **Ground-Truth Verifier Kwargs** | ✅ **GIỮ** | Mỗi variant constraint sinh ra phải kèm bộ `kwargs` chuẩn xác để `python_rule_engine` chấm điểm tự động cho GRPO. |
| **Reference Response mẫu** | ❌ **LOẠI BỎ** | GRPO chấm điểm trên chính response của agent qua Verifier kwargs, không dùng text mẫu. Loại bỏ giúp giảm dung lượng và tránh nhiễu dữ liệu. |

---

## 2. Lộ trình Triển khai Tinh giản (Hỗ trợ trực tiếp Hybrid Dataset)

Pipeline hỗ trợ **đầu vào trực tiếp là các tập dữ liệu Hybrid Task 2** (như [task2_hybrid_alpaca_gpt4.json](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/seen/task2_hybrid_alpaca_gpt4.json)):

```
┌────────────────────────────────────────────────────────┐
│  Đầu vào: Dataset Hybrid Task 2                        │
│  - Prompt có cả Hard Constraints + Soft Guidelines     │
│  - hard_constraints (rule engine kwargs)               │
│  - soft_constraints (LLM judge rubrics)                │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  CRPL Perturbation Engine                              │
│  (cross_relationship_perturbation.py)                  │
│  - Perturb tham số Hard Constraints                    │
│  - Cập nhật hard_constraints: description & kwargs     │
│  - BẢO TOÀN 100% Soft Constraints                      │
│  - Rebuild Prompt (giữ nguyên cấu trúc 2 tầng)         │
│  - Bổ sung crpl_metadata, Loại bỏ reference_response   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Dataset Hybrid CRPL cho GRPO (159 variants từ 21 mẫu) │
│  - Prompt đa dạng ranh giới                            │
│  - Verifier kwargs chính xác                           │
│  - Soft rubrics nguyên vẹn                             │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  GRPO Online Training với Composite Verifiable Reward  │
│  - Group rollouts G responses tại mỗi step             │
│  - Realtime Verifier: R_hard + R_soft                  │
│  - Chuẩn hóa Group Advantage và cập nhật policy        │
└────────────────────────────────────────────────────────┘
```

### Cơ chế Rebuild Prompt cho Hybrid:
Khi một Hard Constraint bị biến đổi (ví dụ: từ `60–137 từ` thành `98–205 từ`), prompt được tái cấu trúc bảo toàn nguyên vẹn 2 tầng:
```text
{base_instruction}

Please adhere strictly to the following constraints in your response:
[Formatting & Structure Constraints]
- {hard_constraint_1}
- {hard_constraint_2_ĐÃ_PERTURB}

[Content & Semantic Guidelines]
- {soft_constraint_1_GIỮ_NGUYÊN}
- {soft_constraint_2_GIỮ_NGUYÊN}
```

---

## 3. Format JSON chuẩn cho môi trường GRPO

### 3.1 Format Hybrid Sample hoàn chỉnh sau khi qua CRPL

Tham chiếu mẫu thực tế sinh từ script [cross_relationship_perturbation.py](file:///home/admin123/Desktop/dataocubuntu/desktop/google-research/instruction_following_eval/datasets/code/cross_relationship_perturbation.py):

```json
{
  "key": "alpaca-gpt4_0_crpl_partial_overlap_1",
  "dataset_source": "vicgalle/alpaca-gpt4",
  "constraint_split": "seen",
  "base_instruction": "Give three tips for staying healthy.",

  "prompt": "Give three tips for staying healthy.\n\nPlease adhere strictly to the following constraints in your response:\n[Formatting & Structure Constraints]\n- Your entire response must be written in Vietnamese.\n- Include at least 5 placeholders in brackets (e.g. [name], [address], [date]) in your response.\n\n[Content & Semantic Guidelines]\n- Must cover multiple distinct dimensions of health maintenance, specifically addressing nutritional intake, physical movement habits, and rest or recovery.\n- Maintain an informative, encouraging, and clear advisory register suitable for general health guidance.",

  "num_constraints": {
    "total": 4,
    "hard_count": 2,
    "soft_count": 2
  },

  "hard_constraints": {
    "verifier_type": "python_rule_engine",
    "constraint_ids": [
      "language:response_language",
      "detectable_content:number_placeholders"
    ],
    "constraints_description": [
      "Your entire response must be written in Vietnamese.",
      "Include at least 5 placeholders in brackets (e.g. [name], [address], [date]) in your response."
    ],
    "ground_truth": [
      {
        "instruction_id": [
          "language:response_language",
          "detectable_content:number_placeholders"
        ],
        "kwargs": [
          {
            "language": "Vietnamese"
          },
          {
            "num_placeholders": 5
          }
        ]
      }
    ]
  },

  "soft_constraints": {
    "verifier_type": "llm_reasoning_judge",
    "model_judge": "google/gemma-4-31B-it",
    "constraints": [
      {
        "id": "semantic_completeness:health_dimensions",
        "category": "semantic_completeness",
        "description": "Must cover multiple distinct dimensions of health maintenance, specifically addressing nutritional intake, physical movement habits, and rest or recovery.",
        "eval_rubric": "Return 1 if the response covers nutrition, physical activity, and sleep/rest as the three tips. Return 0 if any of these three core dimensions are missing."
      },
      {
        "id": "style_and_tone:professional_advisory",
        "category": "style_and_tone",
        "description": "Maintain an informative, encouraging, and clear advisory register suitable for general health guidance.",
        "eval_rubric": "Return 1 if the tone is consistently informative, direct, and encouraging without using overly casual slang or overly academic jargon. Return 0 otherwise."
      }
    ]
  },

  "reward_spec": {
    "aggregation_strategy": "hard_priority_gated",
    "weights": {
      "w_hard": 0.75,
      "w_soft": 0.25
    },
    "formula": "Reward = (0.75 * R_hard + 0.25 * R_soft) if (R_hard > 0 and R_soft >= 0.5) else 0.0",
    "anti_reward_hacking_note": "Nếu R_hard == 0 (không đạt quy tắc cứng) hoặc R_soft < 0.5 (sai lệch nội dung), toàn bộ reward sẽ bằng 0."
  },

  "messages": [
    {
      "role": "user",
      "content": "Give three tips for staying healthy.\n\nPlease adhere strictly to the following constraints in your response:\n[Formatting & Structure Constraints]\n- Your entire response must be written in Vietnamese.\n- Include at least 5 placeholders in brackets (e.g. [name], [address], [date]) in your response.\n\n[Content & Semantic Guidelines]\n- Must cover multiple distinct dimensions of health maintenance, specifically addressing nutritional intake, physical movement habits, and rest or recovery.\n- Maintain an informative, encouraging, and clear advisory register suitable for general health guidance."
    }
  ],

  "crpl_metadata": {
    "relationship": "partial_overlap",
    "perturbed_constraint_index": 1,
    "original_constraint_id": "detectable_content:number_placeholders",
    "perturbed_constraint_id": "detectable_content:number_placeholders",
    "original_description": "Include at least 3 placeholders in brackets (e.g. [name], [address], [date]) in your response.",
    "perturbed_description": "Include at least 5 placeholders in brackets (e.g. [name], [address], [date]) in your response."
  }
}
```

> [!TIP]
> **Điểm mấu chốt của cấu trúc:**
> - Trường `reference_response` đã được tự động loại bỏ.
> - `hard_constraints` được cập nhật mô tả và `kwargs` mới tương ứng với loại perturbation.
> - `soft_constraints` giữ nguyên 100% không suy suyển.
> - `prompt` và `messages` thể hiện đúng thông số mới, đảm bảo tính nhất quán tuyệt đối giữa đề bài và bộ verifier.
> - `crpl_metadata` lưu lại đầy đủ lịch sử biến đổi (traceability).

---

## 4. Kiến trúc Pipeline GRPO tổng thể

```mermaid
flowchart TD
    subgraph DataPrep ["1. Data Preparation (Rẻ & Nhanh)"]
        HybridInput[21 Mẫu Hybrid Gốc\ntask2_hybrid_alpaca_gpt4.json] --> CRPL[CRPL Perturbation Engine\ncross_relationship_perturbation.py]
        CRPL -->|Containment| VarC[53 Containment Variants]
        CRPL -->|Overlap| VarO[53 Overlap Variants]
        CRPL -->|Disjoint| VarD[53 Disjoint Variants]
        
        HybridInput & VarC & VarO & VarD --> GRPODataset[(Tập Huấn luyện GRPO\n180 Prompts Hybrid Hoàn chỉnh)]
    end

    subgraph OnlineGRPO ["2. Online GRPO Training Loop"]
        GRPODataset -->|Sample Prompt x| Policy[Policy Model π_θ]
        Policy -->|Rollout Group G=4| G1[Response o₁]
        Policy -->|Rollout| G2[Response o₂]
        Policy -->|Rollout| G3[Response o₃]
        Policy -->|Rollout| G4[Response o₄]
        
        subgraph RewardVerification ["3. Real-time Verifier"]
            G1 & G2 & G3 & G4 --> RuleEngine[Python Rule Engine\n(Hard Constraints)]
            G1 & G2 & G3 & G4 --> SoftJudge[LLM Reasoning Judge\n(Soft Guidelines)]
            RuleEngine & SoftJudge --> GatedReward[Gated Composite Reward\nR₁..R₄]
        end
        
        GatedReward --> GroupAdv[Group Advantage\nÃ_i = R_i - mean / std]
        GroupAdv --> LossUpdate[Policy Gradient Update\n(GRPO Loss)]
        LossUpdate -.->|Cập nhật trọng số| Policy
    end
```

---

## 5. So sánh: Pipeline ban đầu (DPO) vs Pipeline tinh giản (GRPO)

| Tiêu chí so sánh | Pipeline DPO cũ (đề xuất ban đầu) | Pipeline GRPO tinh giản (hiện tại) |
|---|---|---|
| **Mục đích của CRPL** | Sinh prompt biến thể + Tạo cặp $(y_w, y_l)$ offline | **Chỉ dùng sinh biến thể prompt** để làm giàu không gian huấn luyện |
| **Hỗ trợ Hybrid (Soft Constraints)** | Phức tạp, phải ghép soft sau khi sinh pairs | **Tự động & Nativet:** Input trực tiếp file Hybrid, giữ nguyên soft constraints |
| **Phân vùng $S_1, S_2, S_3$** | Bắt buộc (tốn công phân loại) | **Không cần thiết** |
| **Chi phí API / GPU sinh response trước** | Rất cao (cần sinh $K \times 2 \times N$ responses) | **0đ (Không tốn bất kỳ chi phí nào)** |
| **So sánh phản hồi** | Tĩnh (Offline Pairwise Preference) | **Động (Online Group Advantage)** |
| **Khả năng khám phá (Exploration)** | Bị giới hạn trong các response đã sinh sẵn | **Tự do**: Model liên tục sinh các phản hồi mới trong quá trình train |
| **Độ phức tạp code** | Phức tạp (thêm module phân vùng, ranking, pair sampling) | **Đơn giản, gọn gàng**: 1 script duy nhất xử lý cả Hard-only lẫn Hybrid |

---

## 6. Kế hoạch thực hiện tiếp theo

1. **Bước 1: Chạy sinh Dataset Hybrid CRPL:**
   ```bash
   python datasets/code/cross_relationship_perturbation.py \
       --input datasets/seen/task2_hybrid_alpaca_gpt4.json \
       --output datasets/seen/crpl_task2_hybrid_alpaca_gpt4.jsonl \
       --mode all
   ```
2. **Bước 2: Chuẩn bị Verifier Reward Manager cho GRPO:**
   * Tích hợp `python_rule_engine` (cho hard constraints) và model judge (cho soft constraints) thành 1 class thống nhất để tính reward cho $G$ rollouts.
3. **Bước 3: Cấu hình và khởi chạy GRPO Training.**
