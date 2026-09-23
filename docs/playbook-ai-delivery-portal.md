# SỔ TAY THAM CHIẾU CỦA CƯỜNG (v4)
## AI Delivery Portal (MLOps/LLMOps)
### Viettel Digital Talent 2026 · Track Cloud

> **Thay đổi lớn nhất so với bản trước**: Mentor yêu cầu tạm gác hướng "GitOps for Model" tích hợp CI/CD generic ngoài phạm vi AI Platform — tập trung hoàn thiện đầy đủ luồng MLOps/LLMOps của AI Delivery Portal trước. Sổ tay này viết lại toàn bộ phần định vị/kiến trúc/scope theo đúng phạm vi mới.

---

## 1. LA BÀN — CÁC CÂU HỎI CỐT LÕI

1. **Việc gì lặp lại thường xuyên nhất?** (Frequency)
2. **Việc gì dễ sai nhất?** (Error-proneness)
3. **Việc gì có rủi ro cao nhất nếu làm sai?** (Impact/Blast radius)
4. **Thiết kế này có dễ expose thành tool cho agent sau này không?** (Agent-readiness)

### Ma trận ưu tiên
```
                    Rủi ro thấp              Rủi ro cao
Tần suất cao   │  Tự động hóa nhẹ nhàng  │  ƯU TIÊN SỐ 1
Tần suất thấp  │  Ưu tiên thấp nhất      │  Stretch goal
```

---

## 2. ĐỊNH VỊ ĐỀ TÀI — AI DELIVERY PORTAL

```
┌─────────────────────────────────────────────────────────┐
│ AI Delivery Portal — MLOps/LLMOps (Cường)                │
│ = LỚP NGHIỆP VỤ cho vòng đời model AI, trên nền Backstage│
│                                                           │
│ - Backstage UI + Adapter Pattern tích hợp 4 sản phẩm     │
│   AI Platform (Registry/Experiment/Inference/Notebook)   │
│ - Golden Path #1: Train → Track → Register               │
│ - Golden Path #2: Evaluate → Deploy (quyết định GÌ được  │
│   phép deploy — Evaluate Gate)                            │
│ - Vòng lặp Monitor → Drift → Retrain                      │
│ - Agent-ready: MCP servers + Skills                       │
└─────────────────────────────────────────────────────────┘
```

### Câu chuyện định vị (dùng khi bảo vệ)

> *"AI Delivery Portal là BỘ NÃO cho vòng đời model AI trong AI Platform — quyết định model nào đủ điều kiện đưa vào sử dụng (Evaluate Gate), theo dõi vòng đời model sau khi deploy (Drift → Retrain). Phạm vi hiện tại là hoàn thiện trọn vẹn luồng MLOps/LLMOps nội bộ: train → track → register → deploy → monitor. Việc tích hợp CI/CD generic với hệ thống ngoài là hướng mở rộng đã xác định trong roadmap, không nằm trong phạm vi hiện tại — đây là quyết định thu hẹp phạm vi có chủ đích theo định hướng của mentor, không phải giới hạn kỹ thuật."*

### Đặc thù vòng đời Model AI

So với một ứng dụng thông thường, vòng đời của model AI có những đặc thù riêng — đây là lý do luồng MLOps/LLMOps cần thiết kế khác, không thể dùng nguyên bộ quy trình deploy ứng dụng thông thường:

| | Ứng dụng thông thường | Model AI |
|---|---|---|
| **Điều kiện "đủ tốt để dùng"** | Vượt qua kiểm thử kỹ thuật (build, test) | Vượt qua đánh giá chất lượng (Evaluate Gate — accuracy/LLM-as-judge) |
| **Truy vết (lineage)** | Commit code → artifact | Code + dữ liệu huấn luyện + tham số → model |
| **Vòng đời sau khi đưa vào dùng** | Ổn định tới khi có bản vá mới | Suy giảm dần theo thời gian (drift) — cần giám sát & huấn luyện lại |
| **Rollback bảo vệ điều gì** | Tính đúng đắn kỹ thuật (không lỗi/crash) | Chất lượng dự đoán thực tế (accuracy) |

---

## 3. GOLDEN PATH — KHÁI NIỆM NỀN TẢNG

> **Golden Path = con đường được trải sẵn, dễ đi nhất, đã chuẩn hóa best-practice — "làm đúng" là lựa chọn dễ nhất, không ép buộc tuyệt đối.**

```
Internal Developer Platform (IDP)   ← triết lý ("dev tự phục vụ")
        └── Golden Path             ← cách hiện thực hóa
                └── Backstage        ← công cụ cụ thể để build
```

### 6 nguyên tắc nền tảng
1. "Paved road, not the only road" — khuyến nghị, không ép buộc tuyệt đối
2. Giảm cognitive load — hệ thống "nhớ hộ" best-practice
3. Self-service — dev tự làm được ngay
4. Bắt đầu từ pain point thật
5. Đo lường được (adoption, thời gian, số lỗi giảm)
6. Là sản phẩm sống, cần bảo trì

### 2 Golden Path hiện tại

Chốt **2 golden path** (#1 Train→Track→Register, #2 Register→Deploy) — AI Notebook và provisioning hạ tầng generic vẫn
là tính năng Portal độc lập, không đóng khung thành golden path.

| | #1 — Train → Track → Register | #2 — Evaluate → Deploy |
|---|---|---|
| **Mục đích** | Huấn luyện (hoặc fine-tune) → log vào MLflow Tracking → đăng ký Model Registry | Model đủ điều kiện (qua Evaluate Gate) mới được deploy lên KServe |
| **Template** | `templates/train-track-register/template.yaml` (repo frontend) | `templates/evaluate-deploy-model/template.yaml` (repo frontend) |
| **Cơ chế chạy** | `ClusterWorkflow` `train-register-golden-path` (OpenChoreo WorkflowRun API), hoặc chế độ fine-tune khi có `baseModelUri` | Orchestration API gọi `evaluations/gate.py` (LLM-as-judge) trước khi gọi `KServeAdapter` |
| **Trạng thái** | Template còn mock (`debug:log`), chưa nối Custom Action thật | Template còn mock (`debug:log`), chưa nối Custom Action thật |

### Kế hoạch tiếp theo — Software Template & Catalog

1. **Orchestration API** (`routers/models.py`, mới) — `/register-model`,
   `/trigger-training`, `/policy-check`. Làm trước, đang chặn các bước sau.
2. **4 Custom Action** (`packages/backend/`, chỉ gọi HTTP sang API trên):
   - `orchestration:trigger-training` → Argo
   - `orchestration:register-model` → MLflow
   - `orchestration:policy-check` → Evaluate Gate
   - `orchestration:deploy-model` → commit Git → ArgoCD tự động sync (không gọi K8s trực tiếp)
3. Nối 4 action vào 2 template (thay `debug:log`).
4. **Catalog**: thêm action `catalog:register` cuối Golden Path #1 → tự sinh entity thật thay vì file demo.
5. **Test**: `docker compose up -d` → verify từng endpoint qua curl/Postman → verify lại qua Scaffolder UI.
6. **IDP tham khảo** (theo yêu cầu mentor) — không nền tảng nào làm đúng golden path này, gần nhất là AWS:
   - [AWS: Backstage + SageMaker AIOps](https://docs.aws.amazon.com/prescriptive-guidance/latest/patterns/accelerate-mlops-with-backstage-and-sagemaker-templates.html) — cùng kiến trúc Backstage+Template+Adapter, nhưng dừng ở provisioning hạ tầng, không có Evaluate Gate/dataset lineage.
   - [Red Hat Developer Hub — AI templates](https://github.com/redhat-developer/red-hat-developer-hub-software-templates) — golden path AI của họ là scaffold app GenAI (chatbot, RAG...), không phải MLOps train/register/deploy.
   - [Port — Blueprints](https://docs.getport.io/build-your-software-catalog/define-your-data-model/setup-blueprint/) & [Scorecards](https://docs.port.io/promote-scorecards/usage/) — không có blueprint ML dựng sẵn, Scorecard là rules engine generic, phải tự cấu hình.
   - [Cortex — Internal Developer Portal](https://www.cortex.io/post/what-is-an-internal-developer-portal) — "ML model" chỉ là 1 loại entity liệt kê ngang Kafka topic/Docker image, không có lifecycle riêng.

### Nguyên tắc thiết kế UI/UX

> **"Một Portal duy nhất — không phải tập hợp các UI rời rạc. Dev chỉ chạm vào UI gốc của hệ thống con (MLflow, Argo Workflows, KServe...) khi chủ động muốn đào sâu chi tiết."**

**Vì sao:** đúng tinh thần Backstage — Portal là *single pane of glass*, không phải cổng trung chuyển. Theo nghiên cứu UX cho IDP (Kirsten Schwarzer, "Designing for Success: UX Principles for IDP", KubeCon), khi vẽ user journey map cho onboarding, ma sát (friction) lớn nhất tập trung đúng ở *touchpoint* — chỗ dev phải nhảy qua lại giữa nhiều UI khác nhau để hoàn thành 1 tác vụ. Gộp về 1 Portal là cách trực tiếp triệt tiêu điểm ma sát này, không chỉ là lựa chọn thẩm mỹ.

- Dev KHÔNG BAO GIỜ cần mở UI gốc của MLflow/Argo Workflows để thao tác (trigger training, theo dõi trạng thái).
- Trạng thái/log **nhúng (embed) trực tiếp** trong Portal — không deep-link đưa Dev rời khỏi Portal.
- Log lỗi hiển thị trong Portal theo nguyên tắc "error là proxy của frustration" (Jared Spool, trích trong cùng talk): nêu rõ **chuyện gì xảy ra, vì sao, và bước tiếp theo cần làm** — không chỉ dump nguyên console gốc.
- Áp dụng: Adapter Pattern (`IWorkflowAdapter.get_workflow_status()`, `IInferenceAdapter.get_inference_status()`) lấy trạng thái/log trực tiếp từ Argo Workflows/KServe, Portal render ngay trong trang Catalog của model/component đó.

---

## 4. KIẾN TRÚC HỆ THỐNG

Chạy trên cụm Kubernetes 3 node (1 master + 2 worker, k3d local —
`scripts/setup-3node-infra.sh`), phân vùng theo plane:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Backstage Portal (repo riêng) — packages/app-backstage               │
└───────────────────────────────┬───────────────────────────────────────┘
                                 │
┌────────────────────────────────┼───────────────────────────────────────┐
│ MASTER — Control plane (OpenChoreo, do OpenChoreo quản lý)             │
│  OpenChoreo API (REST+RBAC) · Controllers (reconcile CRD)              │
│  Thunder (IdP, OAuth2)                                                 │
└────────────────────────────────┼───────────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────┐  ┌───────────────────────┐
│ WORKER 1 — Data plane portal                │  │ WORKER 2              │
│ (do OpenChoreo quản lý)                     │  │                       │
│  Orchestration API (FastAPI) + Adapter layer│  │ AI Platform zone      │
│  - auth/thunder.py, evaluations/gate.py     │  │ (ngoài OpenChoreo —   │
│  - routers/chat.py, prompts.py              │  │  phía Viettel)        │
│  MCP servers: mlops, llmops, golden-path-   │  │  MLflow · Qdrant      │
│  guide, ai-observability                     │  │  MinIO · Feast serve  │
└──────────────────────────────────────────────┘  │  · LiteLLM            │
                                                    │───────────────────────│
                                                    │ Workflow plane        │
                                                    │ (training/fine-tune)  │
                                                    │  Argo Workflows       │
                                                    └───────────────────────┘
  Adapter Layer (adapters/) nói với cả 2 worker qua Service DNS trong cụm
```

agents/mcp-servers/: ai-observability, llmops-golden-paths, mlops-golden-paths,
golden-path-guide (4 server, chạy trên worker1)

### Adapter Pattern — nguyên tắc bất biến

```python
# adapters/interfaces.py — đã implement đúng trong repo
class IModelRegistryAdapter(ABC):
    def register_model(...) -> dict: ...
    def list_models(...) -> list[dict]: ...
    def get_model_metrics(...) -> dict: ...
    def get_dataset_lineage(...) -> list[dict]: ...

class IInferenceAdapter(ABC): ...
class IWorkflowAdapter(ABC): ...
class IVectorStoreAdapter(ABC): ...
class ILLMGatewayAdapter(ABC): ...
class IFeatureStoreAdapter(ABC): ...
class INotebookAdapter(ABC): ...
```

Switching Mock → real backend (MLflow/KServe/Argo/...) means adding one new
class that implements the interface — never touching callers. Không có
adapter nào gọi ra hệ thống ngoài phạm vi AI Platform hiện tại; tích hợp CI/CD
generic (nếu cần) sẽ là 1 adapter mới thêm sau, theo đúng nguyên tắc này.

---

## 5. BỐ CỤC SLIDE BẢO VỆ

```
1. Trang bìa (AI Delivery Portal — MLOps/LLMOps)
2. Bối cảnh AI Platform + vị trí của Portal trong bức tranh tổng thể (sơ đồ mục 2)
3. ⭐ Vấn đề thật (pain point — bổ sung số liệu Qwen/fine-tune nếu có)
4. Giải pháp tổng quan — Backstage + Adapter Pattern + Golden Path
5. Kiến trúc hệ thống (sơ đồ mục 4, có nhánh Agent Interface — MCP)
6. Adapter Pattern — minh chứng bằng code thật (interfaces.py)
7. Golden Path #1 (Argo Workflows)
8. Golden Path #2 (trọng tâm) — Evaluate Gate LLM-as-judge
9. LLMOps active: Vector DB + LLM Gateway + Prompt Registry (case thật: Qwen)
10. Demo (live + video backup)
11. Benchmark thời gian/số bước
12. Dashboard observability
13. (nếu MCP ổn định) Demo agent gọi Golden Path qua MCP
14. Giới hạn hiện tại — nêu rõ phạm vi chưa gồm tích hợp CI/CD generic bên ngoài
15. Roadmap Production (bao gồm hướng mở rộng tích hợp CI/CD sau này)
16. Kết luận
```

---

## 6. CÂU TRẢ LỜI MẪU CHO CÂU HỎI KHÓ

**Q: "Sao chưa tích hợp CI/CD generic (Jenkins) với hệ thống ngoài?"**
> "Đây là quyết định thu hẹp phạm vi có chủ đích theo yêu cầu mentor — ưu tiên hoàn thiện trọn vẹn luồng MLOps/LLMOps nội bộ trước (train → track → register → deploy → monitor), tránh dàn trải sang tích hợp CI/CD generic khi lõi nghiệp vụ MLOps chưa xong. Kiến trúc Adapter Pattern đã tính trước hướng mở rộng này — thêm 1 Adapter mới khi cần, không phải thiết kế lại."

**Q: "Vì sao dùng Adapter Pattern thay vì gọi thẳng MLflow/KServe/Argo?"**
> "Để cô lập Portal khỏi thay đổi ở từng backend cụ thể — đổi Mock sang backend thật chỉ cần thêm 1 class implement interface có sẵn trong `adapters/interfaces.py`, không sửa code gọi (Orchestration API, Custom Action). Đây cũng là nền cho việc mở rộng thêm hệ thống mới sau này mà không phải thiết kế lại kiến trúc."

**Q: "Golden path của em dừng ở Deploy, model xuống cấp thì sao?"**
> "Em đã có `agents/skills/evaluate_drift.py` làm nền cho vòng lặp Retrain — phần trigger tự động nằm trong roadmap, nhưng cơ chế phát hiện drift đã có sẵn."

**Q: "Sao lại thêm MCP/Agent, đề bài đâu có yêu cầu?"**
> "Orchestration API thiết kế theo Facade Pattern — thêm kênh truy cập cho AI Agent gần như không phát sinh rủi ro kiến trúc. Đây là minh chứng cho việc thiết kế Adapter/Facade từ đầu là đúng đắn, đón đầu xu hướng Agentic Platform Engineering."

---

## 7. NGUYÊN TẮC GHI NHỚ TỔNG QUÁT

1. Luôn quay lại **4 câu hỏi cốt lõi** (mục 1)
2. Chiều sâu > chiều rộng — **Golden Path #2 là ưu tiên số 1 hiện tại**, không mở rộng MCP thêm cho tới khi #2 xong
3. Mọi thiết kế phải có lý do — kể cả lý do KHÔNG dùng 1 công nghệ (Temporal/Kusion/Crossplane) cũng cần ghi lại, không chỉ lý do có dùng
4. Chủ động nêu giới hạn — đặc biệt phạm vi hiện tại chưa gồm tích hợp CI/CD generic bên ngoài (roadmap có chủ đích, không phải giới hạn kỹ thuật)
5. **Adapter Pattern là bảo hiểm** — cô lập Portal khỏi thay đổi ở từng backend cụ thể (MLflow/KServe/Argo/...), và là nền sẵn cho việc thêm hệ thống ngoài sau này
6. Nghĩ như Platform Engineer thật — golden path giải quyết pain point thật, đo lường được, có đường tới production
7. Agent-ready là triết lý thiết kế, nhưng **không được lấn át việc hoàn thiện Golden Path lõi** — 3 MCP server hiện tại là tín hiệu cần tự kiểm tra lại ưu tiên.

---

**Cập nhật lần 4, sau khi:**

- (a) mentor yêu cầu tạm gác hướng tích hợp CI/CD generic và framing "GitOps for Model", tập trung hoàn thiện đầy đủ luồng MLOps/LLMOps của AI Delivery Portal trước;
- (b) soát trực tiếp repo thật `AI-delivery-portal` (`adapters/interfaces.py`).
