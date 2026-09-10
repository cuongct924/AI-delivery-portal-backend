# LLMOps Lifecycle — Kế hoạch triển khai

> **Trạng thái: CHỐT.** Bản trước (`llmops-lifecycle-plan-draft.md`) viết sau
> phiên thảo luận, trước khi tác giả verify kỹ phần MLOps. MLOps giờ đã hoàn
> thiện end-to-end (Golden Path #1/#2/#3, DL/NLP/CV/RecSys training, Model
> Monitoring, auth guard trên mọi router, Feast tích hợp — 214 test pass,
> `make check` xanh). Toàn bộ claim trong bản draft đã được re-verify trực
> tiếp trên code hiện tại (không suy đoán) — 1 claim sai (mục 4), phần còn lại
> đúng nhưng có vài chi tiết lệch (đường dẫn `evaluations/`, auth guard trên
> `prompts.py`). 5 câu hỏi mở ở cuối bản draft đã chốt xong, có lý do — xem
> mục 8. Từ đây là kế hoạch triển khai trực tiếp được, không còn phần "tuỳ
> chọn chưa quyết".

## 1. Bối cảnh

`docs/playbook-ai-delivery-portal.md` xác định "Đối tượng thứ nhất" của đề tài
là luồng **MLOps/LLMOps** cho các sản phẩm AI Platform. Phần MLOps (Golden
Path #1 Train→Track→Register, #2 Register→Deploy, #3 Recommend→Track→Register,
Setup Model Monitoring) đã hoàn thiện end-to-end (train thật qua Argo
Workflows trên kind cluster — sklearn/MLP/LSTM/NLP/CV/RecSys, Evaluate Gate
thật, PR deploy thật qua ArgoCD, Drift monitoring qua CronWorkflow, Dashboard
hiển thị kết quả). Phần LLMOps — playbook liệt kê gồm 3 trụ cột "Vector DB +
LLM Gateway + Prompt Registry" — mới có từng mảnh rời rạc (adapter đã viết,
chưa ai gọi), chưa nối thành 1 luồng end-to-end. Kế hoạch này lấp đúng phần
đó, không đụng vào phần MLOps đã xong.

## 2. LLMOps khác MLOps ở điểm nào

| | MLOps | LLMOps |
|---|---|---|
| Đối tượng quản lý | Model tự train (weights mới) | LLM có sẵn (Claude) — không tự train, chỉ dùng lại qua API |
| "Version" nghĩa là gì | Model artifact mới sau khi train | Prompt mới, hoặc tập tài liệu RAG mới — trọng số LLM không đổi |
| Đánh giá chất lượng | Metric số (accuracy/precision/recall) so ngưỡng | Chủ yếu LLM-as-judge (output là văn bản tự do) |
| "Deploy" nghĩa là gì | Đưa model lên serving endpoint (KServe) | Kích hoạt prompt/RAG version làm "đang sống" cho endpoint chat |
| Vòng lặp giám sát | Data drift → Monitor→Drift→Retrain | Phản hồi người dùng / re-run eval set định kỳ, không phải "retrain" |

## 3. Pain point có giống 2 Golden Path của MLOps không?

**Giống về hình dạng** — lý do 2 Golden Path tồn tại vẫn đúng cho LLMOps
(chốt ở mục 8, Q3):
- **Golden Path #1 tương ứng** (Draft/Ingest→Register): prompt/tài liệu RAG
  mới cần version + lineage trước khi ai tin dùng — tần suất thay đổi prompt
  trong thực tế còn cao hơn tần suất train lại model.
- **Golden Path #2 tương ứng** (Evaluate→Deploy): pain point cấp thiết nhất
  của LLMOps thực tế — sửa prompt xong đẩy thẳng production, không ai
  review, không có "build step" nào bắt lỗi.

**Khác về cơ chế vận hành phía sau — không copy y nguyên:**
- "Ingest"/"Draft" không nặng như "Train": sửa prompt không cần Argo Workflow
  chạy job dài; ingest RAG (embed + upsert) cũng đủ nhanh để chạy đồng bộ
  trong 1 HTTP request, không cần batch job kiểu Argo.
- "Deploy" nhẹ hơn nhiều: đổi prompt/RAG version active là đổi 1 con trỏ
  trong 1 file JSON, giống bật/tắt feature flag hơn là "provision hạ tầng".
  Khác với MLOps, ở đây **không có ArgoCD (hay cơ chế tương đương) nào theo
  dõi và tự sync** — nên "PR-gated" cho LLMOps không tái tạo được đúng cơ chế
  MLOps đang có (xem mục 8, Q4). Deploy LLMOps = Instant, không có lựa chọn
  khác.

## 4. Quyết định đã chốt trong phiên thảo luận (re-verify: 1 claim đã sửa)

**Không đưa fine-tune LLM thật (LoRA/Qwen qua Ollama...) vào LLMOps lifecycle
đợt này.**

Bản draft trước ghi nhận file `infra/argo-workflows/fine-tune-template.yaml`
(`fine-tune-golden-path`) — **file này không tồn tại, claim đó sai, đã lỗi
thời.** Verify lại thực tế trong code: fine-tune **không phải 1 template
riêng** — nó là tham số `mode` (`train` | `finetune`) trên đúng 1
`WorkflowTemplate` dùng chung cho cả train lẫn fine-tune,
`infra/argo-workflows/train-register-template.yaml`
(`train-register-golden-path`), chọn bởi việc Golden Path #1
(`examples/templates/train-track-register/template.yaml`) có set
`baseModelUri` hay không (`routers/models.py::trigger_training()`, dòng
`"mode": "finetune" if request.base_model_uri is not None else "train"`).
Fine-tune áp dụng cho nhiều architecture hơn draft mô tả — không chỉ
`LogisticRegression` cổ điển: NLP fine-tune 1 model HuggingFace pretrained
thật (`infra/argo-workflows/training-image/train_nlp.py`, mặc định
`distilbert-base-uncased`) và CV fine-tune `resnet18` transfer learning
(`train_cv.py`) — đều là fine-tune thật trên model classification/vision nhỏ,
**không phải LLM sinh văn bản**. Kết luận không đổi, thậm chí chắc hơn: đây
đúng là năng lực MLOps (task classification/vision, không phải LLMOps), giữ
nguyên không đụng vào.

Lý do giữ nguyên kết luận phạm vi:
- Best practice ngành LLMOps đi theo thứ tự: **Prompting → RAG →
  Fine-tuning** (fine-tuning là lựa chọn cuối cùng, không phải điểm khởi
  đầu). Đa số nhu cầu LLMOps thực tế dừng ở bước 1-2.
- Khả năng "dựng pipeline train/fine-tune" đã chứng minh xong ở MLOps (kể cả
  fine-tune NLP/CV thật) — làm lại y hệt cơ chế đó cho 1 LLM qua LoRA không
  chứng minh thêm năng lực mới.
- Phần thực sự CHƯA được chứng minh trong repo là đúng phần đặc thù LLMOps:
  prompt versioning, RAG ingest→evaluate→deploy, và `routers/chat.py` (xác
  nhận lại: vẫn đúng là stub — xem mục 6).

**Kết luận phạm vi (không đổi): LLMOps lifecycle = prompt + RAG + evaluate +
deploy, không có bước train/fine-tune.**

## 5. Luồng RAG ingest→evaluate→deploy — thiết kế cụ thể

Ánh xạ vào đúng adapter đã có sẵn (`adapters/interfaces.py` + implementation),
cộng 1 adapter mới cho phần chưa có cơ chế lưu trữ (`IVersionRegistryAdapter`
— xem mục 8, Q2 và mục 9):

1. **Ingest + Register** (1 lệnh gọi, đồng bộ — không cần Argo vì embed+upsert
   đủ nhanh): tài liệu nguồn (`docs/`, README từng service/adapter, runbook)
   → chunk theo ký tự (chunk_size/chunk_overlap) → mỗi chunk qua **embedding
   model** (Voyage AI qua LiteLLM Gateway — chốt ở Q1) →
   `QdrantAdapter.upsert()` lưu (vector + text gốc + metadata) → 1 version
   mới được ghi vào `IVersionRegistryAdapter` (kind=`"rag-index"`), **chưa
   active**. Vai trò tương đương "Track→Register" bên MLOps — không tự động
   gate ở bước này, giống `trigger_training()` không tự gọi
   `evaluate_metrics_gate()`.
2. **Evaluate**: chạy 1 eval set câu hỏi mẫu qua RAG pipeline —
   `QdrantAdapter.search()` lấy đoạn liên quan từ đúng index version cần
   evaluate → ghép vào prompt → `LiteLLMGatewayAdapter.chat_completion()` →
   câu trả lời → chấm bằng LLM-as-judge đã có sẵn
   (`services/orchestration-api/evaluations/llm_judge.py` +
   `services/orchestration-api/evaluations/gate.py` — **đường dẫn draft cũ
   ghi `evaluations/llm_judge.py` là sai, 2 file này nằm dưới
   `services/orchestration-api/evaluations/`**, cùng cơ chế MLOps Golden Path
   #2 dùng cho phần metric số). Tỷ lệ pass ≥ 80% trên eval set → coi là đủ
   tốt (`passed: bool` trả về, không tự activate).
3. **Deploy = Activate**: gọi endpoint activate, ghi thẳng
   `active_version` trong `IVersionRegistryAdapter` — tức thời, không qua
   PR (chốt ở Q4). `routers/chat.py` khi phục vụ người dùng thật đọc đúng
   `active_version` hiện tại, không phải bản mới nhất chưa kiểm chứng.

Prompt versioning đi theo đúng khuôn tương tự: Draft prompt mới → Evaluate
(LLM-as-judge trên eval set câu hỏi mẫu, dùng prompt đó làm system prompt) →
Register (đã làm cùng lúc Draft — xem mục 9) → Activate (làm system prompt
đang sống cho `chat.py`).

## 6. Thành phần đã có sẵn — tái dùng, không viết lại (re-verified)

| Thành phần | Trạng thái (verify lại) | Vị trí |
|---|---|---|
| Prompt Registry UI | Chạy được, chỉ đọc (read-only), backend in-memory. Response shape `{id,name,version,persona,content}` — kế hoạch này giữ nguyên shape này, không cần sửa UI | `plugins/prompt-registry/`, `routers/prompts.py` |
| Prompt Registry (code-side) | Xác nhận đúng: hằng số tĩnh trùng lặp thủ công (`MLOPS_ASSISTANT`/`K8S_ASSISTANT`), nội dung đã lệch nhẹ so với `routers/prompts.py` — **và hoàn toàn không có nơi nào trong repo import module này** (không Dockerfile nào COPY `agents/prompts/`, không code nào `import` nó). Xoá hẳn — xem mục 9 | `agents/prompts/system_prompts.py` |
| LLM Gateway (LiteLLM) adapter | Xác nhận đúng: đã implement `chat_completion()`/`list_models()`, `litellm-config.yaml` mới có đúng 1 model (`claude-sonnet-5`). Kế hoạch thêm `embed()` + 1 model entry (`voyage-3`) — không phải 1 model chat thứ 2 (xem Q5, không nhầm với Q1) | `adapters/llm_gateway_adapter.py`, `infra/llm-gateways/litellm-config.yaml` |
| LLM-as-judge / Evaluate Gate | Xác nhận đúng, đi qua LiteLLM adapter đúng chuẩn. Chữ ký chính xác: `judge_response(question: str, answer: str) -> dict` (trả `{safety, correctness, relevance, reasoning}`), `evaluate_gate(judge_result: dict, thresholds: GateThresholds \| None = None) -> dict` (`GateThresholds`: `min_safety=8, min_correctness=7, min_relevance=7`) | `services/orchestration-api/evaluations/llm_judge.py`, `.../gate.py` (**đường dẫn đã sửa so với draft cũ**) |
| Vector DB (Qdrant) adapter | Xác nhận đúng: `upsert()`/`search()`/`ensure_collection()` đã implement, **grep xác nhận không ai gọi trong toàn repo**. `qdrant-client==1.19.0` đã có sẵn trong `adapters/requirements.txt`, image pin khớp `docker-compose.yml`. `QDRANT_URL`/`LITELLM_GATEWAY_URL` đã wire sẵn vào env của service `orchestration-api` trong `docker-compose.yml` dù chưa ai dùng | `adapters/vector_db_adapter.py` |
| Chat endpoint | Xác nhận đúng: **stub thật** — `return ChatResponse(reply=f"[stub] {user['preferred_username']} received: {request.message}")`, có sẵn `Depends(get_current_user)` | `services/orchestration-api/routers/chat.py` |
| MCP servers | Xác nhận đúng: 3 server (`mlops-server`: `list_experiments`/`get_model_metrics` — thật qua MLflow; `k8s-server`: `check_pod_status`/`get_logs` — mock; `metrics-server`: `query_metric`/`check_model_latency` — thật qua Prometheus), không server nào liên quan RAG/embedding | `agents/mcp-servers/*/server.py` |
| Prompt Registry API auth | **Không có trong draft cũ, verify thêm**: `routers/prompts.py` đã có `Depends(get_current_user)` trên cả 2 route GET (thêm trong phiên MLOps vừa xong) — endpoint mới thêm ở kế hoạch này giữ cùng guard, không cần thay đổi gì về auth | `routers/prompts.py` |
| Deploy Strategy interfaces | **Không có trong draft cũ, phát hiện thêm khi verify**: `IDeployTrafficStrategy`/`IReleaseStrategy` (`adapters/interfaces.py`) + 4 class cụ thể (`adapters/deploy_strategies.py`) đã tổng quát hoá tốt cho MLOps Golden Path #2, nhưng 2 class release (`PRGatedStrategy`/`InstantStrategy`) gắn chặt với `IInferenceAdapter`/KServe URI (`models:/{name}/{version}`) — **không tái dùng được nguyên trạng cho LLMOps** (activate LLMOps không "deploy" gì lên KServe). Chốt: không cần Strategy pattern cho LLMOps activate ở scope này — xem Q4 | `adapters/deploy_strategies.py` |
| Embedding step | Xác nhận đúng — chưa có gì cả. `infra/vector-dbs/README.md` đã ghi rõ note "pick an embedding model (Voyage AI, or self-hosted)" — đúng là quyết định còn treo, chưa ai âm thầm chọn sẵn ở đâu khác trong repo | — |

## 7. Không nằm trong phạm vi kế hoạch này

Liệt kê rõ để tránh scope creep — lý do giống hệt tinh thần loại fine-tune ở
mục 4 (mentor ưu tiên "hoàn thiện luồng" trước khi mở rộng):

- **MCP tool-routing trong `chat.py`.** Docstring/TODO của `chat.py` nói tới
  việc route tool-call sang 3 MCP server — đây là 1 năng lực "AI Agent
  copilot" riêng, độc lập với việc quản lý version prompt/RAG (đúng phạm vi
  LLMOps lifecycle mục 4 đã chốt). `mcp==2.0.0` đã có sẵn trong
  `services/orchestration-api/requirements.txt`, TODO comment giữ nguyên,
  làm sau, không block kế hoạch này.
- **`release_strategy=pr-gated` cho activate.** Xem Q4 — không có cơ chế sync
  tương đương ArgoCD, dựng nửa vời (mở PR nhưng merge không tự activate) sẽ
  gây hiểu lầm nguy hiểm hơn là không có. Nếu cần audit trail, giải pháp rẻ
  hơn: mỗi lần activate log ra `services/orchestration-api/.state/` (đã
  persist trong JSON) — không cần PR.
- **Rollback template riêng.** Activate không bắt buộc phải đi sau Evaluate
  (xem mục 9) — rollback về version cũ đã dùng lại được `llm-evaluate-deploy`
  ngay, nhập thẳng version cũ vào bước activate. Không cần thêm template.
- **Catalog entity cho prompt/RAG index.** Prompt/RAG index không phải
  "Component" triển khai như model — track qua Prompt Registry UI +
  registry JSON, không qua Backstage Software Catalog. Không mở PR Catalog
  entry cho 2 Golden Path LLMOps mới (khác MLOps).
- **Model chat thứ 2 (Qwen qua Ollama).** Xem Q5.

## 8. 5 câu hỏi mở — đã chốt

### Q1 — Cách tạo embedding

**Chốt: Voyage AI qua LiteLLM Gateway — không phải self-hosted
sentence-transformers như đề xuất ban đầu của draft.** Lật lại đề xuất cũ vì
2 tiền lệ thật trong chính repo này:

- `torch` (CPU-only, `--extra-index-url .../whl/cpu`) đã được thêm cho Deep
  Learning training — nhưng **chỉ trong
  `infra/argo-workflows/training-image/requirements.txt`**, không lọt vào
  `services/orchestration-api/requirements.txt` hay `adapters/requirements.txt`
  bao giờ. `orchestration-api` hiện có 0 dependency ML/tensor nào
  (`fastapi`, `uvicorn`, `pydantic-settings`, `mcp`, `httpx`, `python-jose`,
  `prometheus-fastapi-instrumentator`, `jinja2`, `mlflow`). Thêm
  `sentence-transformers`+`torch` (~1-2GB) vào đúng service đang phục vụ mọi
  request (kể cả `chat.py` sau này) đi ngược tiền lệ "cách ly dependency ML
  nặng vào đúng service cần nó" đã thiết lập.
- `lightfm` bị bỏ hẳn (không chỉ hạ cấp) vì lỗi build thật với Python 3.12 —
  team đã cho thấy sẵn sàng loại bỏ 1 dependency thay vì chịu đau hạ tầng.
  Thêm 1 dependency GB-scale vào đúng service phục vụ chat real-time là rủi
  ro tương tự cần tránh nếu có lựa chọn nhẹ hơn — và có.
- Repo **đã** yêu cầu `ANTHROPIC_API_KEY` (key trả phí) để toàn bộ luồng
  chat/eval/judge chạy được — "không cần thêm API key trả phí" (lý do chính
  draft cũ đưa ra cho self-hosted) không còn đúng nữa, vì repo vốn đã không
  phải "zero paid-API-key". Thêm `VOYAGE_API_KEY` chỉ là nhất quán với giả
  định vận hành đã có sẵn, không phải gánh nặng mới về chất.
- `LiteLLMGatewayAdapter.chat_completion()` đã có đúng hình dạng cần —
  `POST {base_url}/chat/completions`. Thêm `embed()` cùng dạng, `POST
  {base_url}/embeddings`, hoàn toàn không cần cài gì mới vào
  `orchestration-api` — embedding tính bên ngoài process qua HTTP, y hệt
  cách `chat_completion` đã hoạt động. `litellm` (LiteLLM Proxy) hỗ trợ
  Voyage AI như 1 provider chuẩn (`voyage/voyage-3` trong `model_list`).

### Q2 — Lưu trữ prompt/RAG-index version state ở đâu

**Chốt: file JSON local mới trên `orchestration-api`, không tái dùng
`MlflowAdapter.set_model_version_tag()`.** Đã điều tra kỹ khả năng tái dùng
trước khi quyết định:

- `set_model_version_tag()` (`adapters/mlflow_adapter.py`) đúng là cơ chế
  "version nào đang được đánh dấu gì" đã có sẵn, dùng thật trong
  `routers/models.py` (`gate_passed`, `gate_<metric>`, `deploy_pr_url`) —
  nhưng nó gắn chặt vào MLflow Model Registry **model version**, tạo ra bởi
  `mlflow.register_model(model_uri=artifact_uri, name=name)` — đòi hỏi
  `artifact_uri` là 1 artifact MLflow load được (flavor/pyfunc). Prompt text
  và RAG index pointer không phải model artifact — ép chúng qua API này
  nghĩa là phải fake 1 "run" giả mỗi lần chỉ để có `runs:/...` URI đăng ký,
  hoặc chấp nhận `chat.py` (đường phục vụ request nóng) có thêm phụ thuộc
  cứng vào MLflow đang chạy — điều nó hiện **không** có.
- Bài học đúng cần lấy từ `set_model_version_tag()` là **hình dạng** (tên →
  version → key-value nhỏ, không cần DB riêng), không phải **cơ chế**
  (MLflow cụ thể). `routers/prompts.py` tự docstring đã dự đoán đúng hướng
  này: *"Demo version: in-memory data. For production, replace with a DB
  (Postgres) or Git."*
- Chốt: adapter mới `IVersionRegistryAdapter` (interface) +
  `JsonFileVersionRegistryAdapter` (implementation) — 1 file JSON local
  (`services/orchestration-api/.state/llmops-registry.json`, gitignore),
  active ngay sau khi gọi activate — đơn giản, tức thời, mất dữ liệu khi
  container restart (chấp nhận được, cùng tier với `_PROMPTS` in-memory hiện
  tại — demo/dev, không phải production nhiều instance). Theo đúng Adapter
  Pattern (CLAUDE.md): swap sang Postgres/Git sau này chỉ cần 1 class mới,
  không đụng caller.

### Q3 — 1 template gộp hay 2 template riêng

**Chốt: 2 template riêng, đúng cấu trúc MLOps — nhưng gộp "prompt" và
"rag-index" làm 2 nhánh trong CÙNG 2 template đó** (không phải 4 template),
qua 1 tham số `artifactKind`.

Lý do giữ 2 template (không gộp Draft→Evaluate→Register→Deploy làm 1):
`register-deploy` (Golden Path #2 MLOps) đã chứng minh tách "evaluate" khỏi
"deploy" hữu ích vì quan hệ nhiều-đối-một — 1 version có thể deploy lại nhiều
lần (canary tăng dần %, rollback về version cũ) mà không cần chạy lại
train/evaluate. Lý luận này **chuyển được nguyên vẹn sang LLMOps, thậm chí
mạnh hơn**: đúng câu mục 3 tự nêu ("deploy nhẹ như feature flag") không phải
lý do để gộp — ngược lại, chính vì activate quá nhẹ/rẻ nên nó càng hợp làm 1
action tách riêng, gọi lại nhiều lần độc lập (rollback = activate lại 1
version cũ, không phải chạy lại cả luồng). Gộp làm 1 sẽ mất khả năng "chỉ
evaluate, chưa deploy ngay" — đúng pain point mục 3 nêu (sửa prompt xong đẩy
thẳng production không ai review) là universe kế hoạch này đang cố tránh.

Lý do gộp "prompt" và "rag-index" vào chung 1 cặp template (không tách 4):
đúng tiền lệ `train-track-register/template.yaml` đã dùng cho nhiều
`architecture` (sklearn/mlp/lstm/nlp/cv) trong 1 template qua field
`architecture` + JSON Schema `allOf/if/then` — dùng lại chính xác pattern đó
với field `artifactKind: [prompt, rag-index]`.

→ `examples/templates/llm-draft-register/template.yaml` (Golden Path LLMOps
#1) và `examples/templates/llm-evaluate-deploy/template.yaml` (Golden Path
LLMOps #2) — xem mục 9.

### Q4 — Instant hay PR-gated (gắn với Q2/Q3, chốt cùng lúc)

**Chốt: Instant — duy nhất, không có lựa chọn PR-gated.** Không phải vì
"nhanh hơn nên chọn nhanh" — mà vì PR-gated **không hoạt động đúng** cho
LLMOps với hạ tầng hiện có: MLOps PR-gated hoạt động vì ArgoCD
(nay `infra/argocd/applicationset-inference-services-dev.yaml` — đã tái
cấu trúc multi-env/multi-tenant, xem `infra/argocd/README.md`; tại thời
điểm viết mục này vẫn là `inference-services-app.yaml`) đang theo dõi
`infra/environments/dev/inference-services/` (trước đó:
`infra/inference-services/`) và tự sync — merge PR **chính là** deploy,
không cần bước nào thêm. LLMOps không có cơ chế tương đương nào theo dõi và tự nạp
lại `services/orchestration-api/.state/llmops-registry.json` vào process
`orchestration-api` đang chạy sau khi 1 PR merge — nếu chọn PR-gated làm mặc
định, "merge PR" sẽ âm thầm **không** activate gì cho tới khi có người chạy
thêm 1 bước thủ công khác — đúng kiểu tạo hạ tầng dở dang mà mentor đã nhiều
lần yêu cầu tránh (mục 4). Xác nhận độc lập: `docs/mlops-lifecycle-software-
template.md` mục 7 (viết trước, không phụ thuộc kế hoạch này) đã tự đề xuất
đúng `IReleaseStrategy` mặc định nên là Instant cho LLMOps — cùng kết luận.

Vì Instant là lựa chọn duy nhất, **không cần** dựng lại
`IDeployTrafficStrategy`/`IReleaseStrategy` cho LLMOps activate — 2 endpoint
`POST /rag/activate`/`POST /prompts/{name}/activate` gọi thẳng
`registry_adapter.set_active_version(...)`, không qua Strategy pattern. Nếu
sau này cần A/B prompt version thật (route % request theo
`IDeployTrafficStrategy`, đúng như mục 7 tài liệu MLOps đã gợi ý) — đó là
điểm mở rộng tự nhiên, không phải yêu cầu của kế hoạch này.

### Q5 — Model thứ 2 (Qwen qua Ollama)

**Xác nhận loại khỏi phạm vi** — playbook chỉ nhắc Qwen như ví dụ trên 1
slide ("LLMOps active: Vector DB + LLM Gateway + Prompt Registry (case thật:
Qwen)"), không phải yêu cầu kỹ thuật bắt buộc. Không tìm thấy lý do nào để
lật lại: multi-model Gateway **đã** là năng lực kiến trúc sẵn có
(`litellm-config.yaml` chỉ là 1 list `model_list`, thêm 1 entry không cần
sửa code) — nên bản thân việc "chứng minh Gateway multi-model" không cần lên
kế hoạch, ai cần có thể thêm trong 5 phút không đụng code. Lưu ý tránh nhầm:
Q1 đã thêm `voyage-3` vào `litellm-config.yaml` — đó là model **embedding**,
không phải model chat thứ 2 mà Q5 đang nói tới; sau kế hoạch này
`litellm-config.yaml` có 2 entry (`claude-sonnet-5` + `voyage-3`), không
phải minh chứng "multi-model chat" mà playbook slide đang muốn nói.

## 9. Kế hoạch triển khai — file cụ thể

### 9.1 Adapters

| File | Thay đổi |
|---|---|
| `adapters/interfaces.py` | Thêm `embed(self, model: str, input_texts: list[str]) -> list[list[float]]` (abstract) vào `ILLMGatewayAdapter`. Sửa `IVectorStoreAdapter.upsert()`/`.search()` thêm tham số `collection: str \| None = None` (mặc định dùng `self.collection` — cần vì RAG ingest/evaluate có thể chạy đồng thời trên nhiều collection khác nhau, không an toàn nếu mutate `self.collection` của 1 singleton dùng chung). Thêm class mới `IVersionRegistryAdapter` (abstract: `register_version`, `get_version`, `list_versions`, `get_active_version`, `set_active_version` — chữ ký ở dòng dưới) |
| `adapters/llm_gateway_adapter.py` | Thêm `embed()` vào `LiteLLMGatewayAdapter` — `POST {base_url}/embeddings`, body `{"model": model, "input": input_texts}`, trả `[item["embedding"] for item in response.json()["data"]]` (cùng style `chat_completion()` đã có) |
| `adapters/vector_db_adapter.py` | Sửa `upsert()`/`search()`/`ensure_collection()` nhận thêm `collection: str \| None = None`, dùng `collection or self.collection` nội bộ — không đổi behavior khi không truyền (giữ tương thích, dù hiện chưa ai gọi) |
| `adapters/version_registry_adapter.py` (mới) | `JsonFileVersionRegistryAdapter(IVersionRegistryAdapter)` — xem chữ ký dưới |

`IVersionRegistryAdapter` (đặt trong `adapters/interfaces.py`, cạnh
`IModelRegistryAdapter`):

```python
class IVersionRegistryAdapter(ABC):
    @abstractmethod
    def register_version(self, kind: str, name: str, metadata: dict) -> str: ...

    @abstractmethod
    def get_version(self, kind: str, name: str, version: str) -> dict: ...

    @abstractmethod
    def list_versions(self, kind: str, name: str) -> dict[str, dict]: ...

    @abstractmethod
    def get_active_version(self, kind: str, name: str) -> str | None: ...

    @abstractmethod
    def set_active_version(self, kind: str, name: str, version: str) -> None: ...
```

`JsonFileVersionRegistryAdapter` (`adapters/version_registry_adapter.py`) —
`kind` phân biệt `"prompt"` vs `"rag-index"`, `name` là persona/collection
key, `version` tự tăng dần (`str(len(existing_versions) + 1)`, cùng kiểu
chuỗi số MLflow đang dùng cho model version). File backing:
`services/orchestration-api/.state/llmops-registry.json` (đường dẫn đọc từ
`LLMOPS_REGISTRY_PATH` env, mặc định `.state/llmops-registry.json` — tương
đối, resolve theo cwd của process, cùng quy ước `MLFLOW_TRACKING_URI` mặc
định `localhost` khi chạy ngoài Docker). Dùng `threading.Lock` quanh
read-modify-write (FastAPI có thể xử lý request đồng thời). Tuân thủ layout
class của `.claude/rules/python-standards.md`: attributes → `__init__` →
public methods (`register_version`, `get_version`, `list_versions`,
`get_active_version`, `set_active_version`) → private methods (`_read`,
`_write`) ở cuối.

Thêm dòng vào `.gitignore`: `services/orchestration-api/.state/`.

### 9.2 orchestration-api — router mới `routers/rag.py`

Docstring đầu file theo đúng mẫu `routers/models.py`: nêu rõ business logic
nằm ở đây (CLAUDE.md), route nào có `Depends(get_current_user)` (tất cả — cả
3 route đều gọi từ Backstage Custom Scaffolder Action).

```python
router = APIRouter(prefix="/rag", tags=["rag"])

llm_gateway_adapter = LiteLLMGatewayAdapter()
vector_store_adapter = QdrantAdapter()
registry_adapter = JsonFileVersionRegistryAdapter()

EMBEDDING_MODEL: Final[str] = "voyage-3"


class RagIngestRequest(BaseModel):
    collection: str
    source_paths: list[str]  # repo-relative, vd ["docs/playbook-ai-delivery-portal.md"]
    chunk_size: int = 800
    chunk_overlap: int = 100


class RagIngestResponse(BaseModel):
    collection: str
    index_version: str
    chunks_ingested: int


class RagEvalCase(BaseModel):
    question: str


class RagEvaluateRequest(BaseModel):
    collection: str
    index_version: str
    eval_cases: list[RagEvalCase]
    top_k: int = 5
    # Overridable — Serving LLM Golden Path deploys models onto the same
    # LiteLLM Gateway (add a model_list entry, no code change there), but
    # a hardcoded model= string here would still lock every eval to
    # Claude regardless. Default kept for convenience, not as a floor.
    model: str = "claude-sonnet-5"


class RagEvaluateResponse(BaseModel):
    passed: bool
    pass_rate: float
    results: list[dict[str, object]]


class RagActivateRequest(BaseModel):
    collection: str
    index_version: str


class RagActivateResponse(BaseModel):
    collection: str
    active_version: str
```

- `POST /rag/ingest` — đọc từng file trong `source_paths` (`Path(p).read_text()`),
  chunk theo ký tự (`chunk_size`/`chunk_overlap`, thuật toán cắt cửa sổ
  trượt đơn giản — không cần token-aware ở scope này), `llm_gateway_adapter
  .embed(EMBEDDING_MODEL, chunks)`, `vector_store_adapter.ensure_collection(
  vector_size=len(vectors[0]), collection=request.collection)` rồi
  `.upsert(ids, vectors, payloads, collection=request.collection)` (payload
  mỗi point: `{"text": chunk, "source": source_path}`), rồi
  `registry_adapter.register_version("rag-index", request.collection,
  {"chunks_ingested": len(chunks), "source_paths": request.source_paths})`
  — trả `index_version` mới, **chưa active**.
- `POST /rag/evaluate` — với mỗi `eval_cases`: embed câu hỏi, `search(...,
  collection=request.collection)` lấy `top_k` đoạn, ghép context vào system
  prompt, `chat_completion(model=request.model, ...)` (**không hardcode** —
  `request.model` mặc định `"claude-sonnet-5"` nhưng gọi được bất kỳ
  `model_name` nào đã đăng ký trong `litellm-config.yaml`, kể cả model tự
  host qua Serving LLM Golden Path — tránh vendor lock-in vào Claude),
  `judge_response()` + `evaluate_gate()` (từ `evaluations/llm_judge.py`/
  `gate.py`, import y hệt `routers/models.py` đang import
  `evaluate_metrics_gate`), gom `pass_rate = passed_count /
  len(eval_cases)`, `passed = pass_rate >= 0.8`. **Không tự activate** —
  trả `results` chi tiết từng case để Dev xem trong log bước Scaffolder.
- `POST /rag/activate` — `registry_adapter.set_active_version("rag-index",
  request.collection, request.index_version)`. Không kiểm tra lại đã
  evaluate hay chưa (cho phép rollback về version cũ không cần evaluate lại
  — xem mục 7) — việc gate theo kết quả evaluate nằm ở tầng template
  (`if:` trên bước activate, xem mục 9.4), không ở tầng endpoint, đúng tiền
  lệ `routers/models.py::prepare_deploy_manifest()` cũng không tự kiểm tra
  `policy_check` đã chạy chưa.

### 9.3 orchestration-api — mở rộng `routers/prompts.py`

Xoá `_PROMPTS` hardcode, thay bằng `registry_adapter =
JsonFileVersionRegistryAdapter()` (module-level singleton, cùng instance
logic như `routers/rag.py` — dùng chung 1 file JSON, khác `kind`). Thêm hàm
`_seed_default_prompts()` gọi ở import time: nếu registry chưa có version
nào cho `"mlops"`/`"k8s"`, đăng ký + activate ngay bằng đúng nội dung hiện
tại của `_PROMPTS` (giữ nguyên hành vi mặc định sau khi restart lần đầu).

```python
class DraftPromptRequest(BaseModel):
    name: str  # persona key, vd "mlops"
    persona: str
    content: str


class DraftPromptResponse(BaseModel):
    id: str
    name: str
    version: str
    persona: str
    content: str


class PromptEvalCase(BaseModel):
    question: str


class EvaluatePromptRequest(BaseModel):
    version: str
    eval_cases: list[PromptEvalCase]
    model: str = "claude-sonnet-5"  # overridable — same reasoning as RagEvaluateRequest.model


class EvaluatePromptResponse(BaseModel):
    passed: bool
    pass_rate: float
    results: list[dict[str, object]]


class ActivatePromptRequest(BaseModel):
    version: str


class ActivatePromptResponse(BaseModel):
    name: str
    active_version: str
```

- `GET /prompts` — sửa: trả version **đang active** của mỗi persona đã đăng
  ký (không phải toàn bộ lịch sử) — giữ nguyên shape `PromptVersion` cũ, UI
  không cần sửa gì.
- `GET /prompts/{prompt_id}` — giữ nguyên hành vi (tra theo `id` dạng
  `"{name}-v{version}"`).
- `POST /prompts` (mới) → `draft_prompt()` — `registry_adapter
  .register_version("prompt", request.name, {"persona": request.persona,
  "content": request.content})`, trả version mới, **chưa active**.
- `POST /prompts/{name}/evaluate` (mới) → `evaluate_prompt()` — lấy content
  tại `version` chỉ định (không nhất thiết active) làm system prompt, chạy
  từng `eval_cases` qua `chat_completion(model=request.model, ...)` (cùng
  field overridable, cùng lý do tránh vendor lock-in) + `judge_response` +
  `evaluate_gate`, cùng công thức pass_rate ≥ 0.8 như `/rag/evaluate` (đối
  xứng có chủ đích, dễ hiểu, dễ maintain).
- `POST /prompts/{name}/activate` (mới) → `activate_prompt()` —
  `registry_adapter.set_active_version("prompt", name, request.version)`.
  Không ép phải evaluate trước (lý do giống `/rag/activate`).

### 9.4 orchestration-api — `routers/chat.py` (thay stub bằng thật)

**Phạm vi: gọi LLM thật + dùng đúng prompt version active + RAG retrieval
tuỳ chọn. Không route MCP tool (xem mục 7).**

```python
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    persona: str = "mlops"  # khớp tên trong routers/prompts.py
    use_rag: bool = False
    rag_collection: str | None = None  # bắt buộc khi use_rag=True
    model: str = "claude-sonnet-5"  # overridable — same reasoning as RagEvaluateRequest.model


class ChatResponse(BaseModel):
    reply: str
    persona_version: str
    rag_index_version: str | None = None
```

`send_message()`:
1. `active_version = registry_adapter.get_active_version("prompt", request.persona)`
   — 404 nếu chưa seed (không nên xảy ra sau `_seed_default_prompts()`).
   `system_prompt = registry_adapter.get_version("prompt", request.persona,
   active_version)["content"]`.
2. Nếu `use_rag`: `rag_version = registry_adapter.get_active_version(
   "rag-index", request.rag_collection)` — 400 nếu `None` ("no active RAG
   index for this collection"). Embed `request.message`, `search(...,
   collection=request.rag_collection)`, ghép context vào đầu
   `system_prompt`.
3. `llm_gateway_adapter.chat_completion(model=request.model, messages=[
   {"role": "system", "content": system_prompt}, {"role": "user", "content":
   request.message}])` → `reply`. **Không hardcode** — mặc định
   `"claude-sonnet-5"` nhưng Dev/end-user chọn được bất kỳ `model_name` nào
   đã đăng ký trong `litellm-config.yaml`, kể cả model tự host qua Serving
   LLM Golden Path (thêm 1 entry vào `model_list`, không cần sửa
   `chat.py`) — tránh vendor lock-in vào Claude.
4. Trả `ChatResponse(reply=..., persona_version=active_version,
   rag_index_version=rag_version if use_rag else None)`.

### 9.5 `services/orchestration-api/main.py`

Thêm `from routers import ... rag` và `app.include_router(rag.router)`.

### 9.6 Xoá duplicate — `agents/prompts/system_prompts.py`

Xoá file (và thư mục `agents/prompts/` nếu trống sau đó) — xác nhận không
nơi nào import (`grep -rn "system_prompts\|PROMPT_REGISTRY"` chỉ ra chính
file này). `chat.py` giờ đọc prompt content trực tiếp từ
`registry_adapter` (cùng process, không qua HTTP) — 1 nguồn sự thật duy
nhất, đúng yêu cầu gộp 2 nơi lưu prompt trùng lặp.

### 9.7 Custom Scaffolder Actions — `packages/backend/src/actions/mlopsActions.ts`

Thêm vào **đúng file `mlopsActions.ts` hiện có** — dù tên file chỉ nói
"mlops", file này thực tế đã là nơi chứa mọi Custom Scaffolder Action gọi
orchestration-api (đã có action cho RecSys, Monitoring, không phải chỉ
MLOps thuần) — theo tiền lệ, không tách file riêng theo tên miền LLMOps.
Copy đúng khuôn `createValidateDatasetAction`/`createEnrichDatasetFeaturesAction`
(interface response ở đầu file, `getBaseUrl(config)` + `postJson<T>(...)`,
`ctx.output(...)`):

| Action id | Gọi endpoint | Input chính | Output chính |
|---|---|---|---|
| `orchestration:rag-ingest` | `POST /rag/ingest` | `collection`, `sourcePaths: string[]`, `chunkSize?`, `chunkOverlap?` | `indexVersion`, `chunksIngested` |
| `orchestration:rag-evaluate` | `POST /rag/evaluate` | `collection`, `indexVersion`, `evalCasesJson: string` (JSON array `[{"question": "..."}]` — 1 field JSON, đúng tiền lệ `hyperparametersJson` của `recommend-train-register/template.yaml`, không model nested array trong Backstage picker), `model?: string` (mặc định `"claude-sonnet-5"` phía backend nếu bỏ trống — cùng field `orchestration:evaluate-prompt` có, forward thẳng xuống `RagEvaluateRequest.model`, tránh vendor lock-in) | `passed: boolean`, `passRate: number` |
| `orchestration:rag-activate` | `POST /rag/activate` | `collection`, `indexVersion` | `activeVersion` |
| `orchestration:draft-prompt` | `POST /prompts` | `name`, `persona`, `content` | `version` |
| `orchestration:evaluate-prompt` | `POST /prompts/{name}/evaluate` | `name`, `version`, `evalCasesJson: string`, `model?: string` | `passed: boolean`, `passRate: number` |
| `orchestration:activate-prompt` | `POST /prompts/{name}/activate` | `name`, `version` | `activeVersion` |

Cả 6 action parse `evalCasesJson`/build request body trong `handler()`
trước khi `postJson`, ném lỗi rõ ràng nếu JSON không parse được (cùng style
`throw new Error(...)` của `createValidateDatasetAction` khi có blocking
check).

Đăng ký cả 6 trong `packages/backend/src/actions/index.ts`
(`scaffolder.addActions(...)`), cùng chỗ với các action MLOps hiện có.

### 9.8 Golden Path template mới

**`examples/templates/llm-draft-register/template.yaml`** — "Draft/Ingest →
Register (LLMOps Golden Path #1)". Tham số `artifactKind: [prompt,
rag-index]` (bắt buộc, không default — buộc chọn rõ, đúng kiểu `taskType`
của `train-track-register`), JSON Schema `allOf/if/then` gate field theo
`artifactKind` (đúng pattern có sẵn — field `promptName`/`personaTitle`/
`promptContent` khi `artifactKind=prompt`; `collectionName`/`sourcePaths`/
`chunkSize`/`chunkOverlap` khi `artifactKind=rag-index`). Step: 1 trong 2
(`draft-prompt` hoặc `rag-ingest`) tuỳ `artifactKind`, dùng `if:` như
`register-deploy`'s `publish-pr` step đã dùng cho `releaseStrategy`. Output:
in ra version vừa tạo, nhắc rõ **chưa active** — chạy tiếp Golden Path #2 để
evaluate + activate.

**`examples/templates/llm-evaluate-deploy/template.yaml`** — "Evaluate →
Deploy (LLMOps Golden Path #2)". Cùng `artifactKind` selector, cộng thêm
tham số `model` (optional, default `"claude-sonnet-5"`, mô tả: "Any
model_name registered in litellm-config.yaml — including a self-hosted
model deployed via the Serving LLM Golden Path") forward vào bước
`evaluate` — đây là chỗ Dev thật sự chọn được model, tránh vendor lock-in
đã sửa ở mục 9.2/9.3/9.4. Steps:
1. `evaluate` — `orchestration:rag-evaluate` hoặc `orchestration:evaluate-prompt`
   tuỳ `artifactKind`.
2. `activate` — `orchestration:rag-activate` hoặc `orchestration:activate-prompt`,
   **`if: ${{ steps['evaluate'].output.passed }}`**. Đây là khác biệt có chủ
   đích so với `register-deploy/template.yaml` (bước `prepare-manifest` ở
   đó KHÔNG có `if:` gate theo `steps['policy-check'].output.passed` — chạy
   vô điều kiện) — với LLMOps, gate này chính là câu trả lời trực tiếp cho
   pain point mục 3 ("sửa prompt xong đẩy thẳng production, không ai
   review, không có build step nào bắt lỗi"). Không sửa lại
   `register-deploy` của MLOps — ngoài phạm vi kế hoạch này.

Output: nếu `passed=false`, in rõ pass_rate + gợi ý sửa version rồi chạy lại
Golden Path #1; nếu `passed=true`, xác nhận đã activate.

### 9.9 Đăng ký template + config

- `app-config.yaml` (`catalog.locations`, dạng `../../examples/templates/...`)
  và `app-config.production.yaml` (dạng `./examples/templates/...`) — thêm
  cả `llm-draft-register` và `llm-evaluate-deploy`, đúng quy tắc CLAUDE.md
  "Adding a Catalog entity/template".
- `infra/llm-gateways/litellm-config.yaml` — thêm entry:
  ```yaml
    - model_name: voyage-3
      litellm_params:
        model: voyage/voyage-3
        api_key: os.environ/VOYAGE_API_KEY
  ```
- `docker-compose.yml` — thêm `VOYAGE_API_KEY: ${VOYAGE_API_KEY}` vào
  `environment:` của service `litellm`.
- `.env.example` — thêm dòng `VOYAGE_API_KEY=` cạnh `ANTHROPIC_API_KEY`.

### 9.10 Test — mỗi phần mới có test tương ứng

Quy ước test Python trong repo (xác nhận qua `tests/test_models_router.py`,
`tests/test_prompts_router.py`): 1 file `tests/test_<router_name>.py`,
patch instance adapter module-level bằng `unittest.mock.patch`, gọi thẳng
hàm route (không dùng `TestClient`) — vì `pythonpath` trong
`pyproject.toml` đã có `services/orchestration-api`, import `from
routers.rag import ...` hoạt động trực tiếp trong test.

| File test | Nội dung |
|---|---|
| `tests/test_version_registry_adapter.py` (mới) | Test `JsonFileVersionRegistryAdapter` bằng file thật trên `tmp_path` (không mock — không có SDK ngoài để mock, đúng như các adapter thuần I/O khác không có tiền lệ mock) — `register_version` tự tăng version, `set_active_version` raise `ValueError` nếu version chưa đăng ký, `get_active_version` trả `None` khi chưa có gì |
| `tests/test_rag_router.py` (mới) | Patch `routers.rag.llm_gateway_adapter`, `routers.rag.vector_store_adapter`, `routers.rag.registry_adapter`. Case chính: `ingest()` gọi đúng `embed()`→`upsert()`→`register_version()` theo thứ tự, trả `index_version`; `evaluate()` tính đúng `pass_rate`/`passed` từ nhiều `judge_response` mock khác nhau (patch `routers.rag.judge_response`/`evaluate_gate` theo đúng cách `tests/test_llm_judge.py` patch `evaluations.llm_judge.LiteLLMGatewayAdapter`); `activate()` gọi đúng `set_active_version` |
| `tests/test_prompts_router.py` (mở rộng, không tạo file mới) | Thêm test cho `draft_prompt`/`evaluate_prompt`/`activate_prompt`, patch `routers.prompts.registry_adapter` (thay `_PROMPTS` cũ). Giữ nguyên 3 test hiện có nhưng sửa fixture cho khớp store mới (seed trước khi test `list_prompts`/`get_prompt`) |
| `tests/test_chat_router.py` (mới) | Patch `routers.chat.registry_adapter`, `routers.chat.llm_gateway_adapter`, `routers.chat.vector_store_adapter`. Case: không `use_rag` → không gọi `search()`; có `use_rag` nhưng chưa có active RAG version → raise `HTTPException(400)`; happy path trả đúng `persona_version` |

Không cần file test riêng cho `LiteLLMGatewayAdapter.embed()` hay
`QdrantAdapter`'s tham số `collection` mới — đúng tiền lệ repo hiện tại
(`chat_completion()` không có test adapter riêng, chỉ được exercise gián
tiếp qua nơi gọi nó bị mock nguyên class, vd `test_llm_judge.py`) — logic
mới ở 2 adapter này được exercise gián tiếp qua `test_rag_router.py`.

TypeScript: `packages/backend/src/actions/mlopsActions.test.ts` — thêm
`describe()` cho 6 action mới, dùng đúng helper có sẵn
(`createMockContext<typeof action>(input, workspacePath)`,
`mockFetchResponses([...])`) — theo mẫu `describe('orchestration:register-model', ...)`
đã có.

## 10. Kiểm chứng (matching bar của phiên MLOps vừa xong)

- **`make check`** — ruff + pyright + pytest phải xanh, tổng test tăng từ
  214 lên ≥ 214 + (số test mới ở mục 9.10). Không giảm coverage các module
  cũ.
- **`yarn tsc && yarn lint:all`** — bắt buộc sau khi sửa `mlopsActions.ts`/
  `index.ts` (type Zod schema chặt, không dùng `any`).
- **Smoke test end-to-end thật** (không mock, theo đúng cách Feast/
  register-model đã verify trong phiên trước — hạ tầng local thật, không
  test qua mock):
  1. `docker compose up -d qdrant litellm` (cần `ANTHROPIC_API_KEY` và
     `VOYAGE_API_KEY` thật trong `.env`), `make run-orchestration-api`.
  2. `curl -X POST localhost:8000/rag/ingest -d '{"collection":"smoke-test",
     "source_paths":["docs/playbook-ai-delivery-portal.md"]}'` — check
     response có `index_version="1"`, `chunks_ingested > 0`; mở Qdrant
     dashboard (`localhost:6333/dashboard`) xác nhận collection
     `smoke-test` có points thật, payload chứa `text` đúng nội dung file.
  3. `curl -X POST localhost:8000/rag/evaluate -d '{"collection":
     "smoke-test","index_version":"1","eval_cases":[{"question":"AI
     Delivery Portal quản lý vòng đời model AI như thế nào?"}]}'` — check
     `passed`, đọc `results[0].answer` xác nhận có nhắc nội dung thật từ
     file đã ingest (không phải câu trả lời chung chung không liên quan
     context).
  4. `curl -X POST localhost:8000/rag/activate -d '{"collection":
     "smoke-test","index_version":"1"}'`.
  5. `curl -X POST localhost:8000/chat -d '{"message":"...", "persona":
     "mlops", "use_rag": true, "rag_collection":"smoke-test"}'` — check
     `reply` không còn `[stub]`, `rag_index_version="1"`.
  6. Chạy thử `llm-draft-register` rồi `llm-evaluate-deploy` qua Backstage
     Scaffolder UI thật (`yarn start`), cả 2 nhánh `artifactKind=prompt` và
     `artifactKind=rag-index`, xác nhận step `activate` **không** chạy khi
     cố tình cho `eval_cases` câu hỏi lệch chủ đề để `passed=false`.
