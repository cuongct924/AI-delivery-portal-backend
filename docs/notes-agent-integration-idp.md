# Notes: Tích hợp AI Agent vào AI Delivery Portal

> Ghi chú nghiên cứu — chưa phải kế hoạch triển khai chính thức. Dựa trên
> framework 4 cấp độ phức tạp (traditional code → deterministic workflow →
> RAG chatbot → autonomous agent) áp vào kiến trúc hiện có của repo.

## 1. Framework 4 tầng

| Tầng | Đặc điểm | Khi dùng |
| :--- | :--- | :--- |
| 1. Traditional code | Input dự đoán được, độ trễ thấp, logic minh bạch | Parse log định dạng chuẩn, hệ thống cần audit rõ ràng |
| 2. Deterministic workflow | Các bước & nhánh rẽ hữu hạn, liệt kê được trước | Cần retry/backoff, cần điểm dừng chờ người duyệt |
| 3. RAG chatbot | Hiểu ngôn ngữ tự nhiên + truy xuất tài liệu, **không** tự lên kế hoạch nhiều bước | Hỏi-đáp trên kho tri thức, chi phí bảo trì thấp |
| 4. Autonomous agent | Input phi cấu trúc/biến động cao, tự lên kế hoạch động nhiều bước, tự gọi API, học từ phản hồi | Khi rule cứng và RAG đều bó tay trước dữ liệu quá đa dạng |

Nguyên tắc chọn nhanh: code thuần cho transform cố định → workflow nếu có
vài nhánh rẽ rõ ràng cần kiểm soát lỗi → RAG cho hỏi-đáp tài liệu → chỉ đầu
tư Agent khi hệ thống phải đối mặt với dữ liệu biến động cao, cần tự suy
luận và tự lên kế hoạch hành động liên tục.

## 2. Áp vào kiến trúc hiện tại của repo

| Tầng | Đã có ở đâu trong repo | Vì sao giữ nguyên |
| :--- | :--- | :--- |
| 1 | `catalog_client.py`, `data_quality/checks.py`, `evaluations/gate.py` | Transform cố định, cần test/audit được |
| 2 | `templates/<golden-path-name>/template.yaml` (repo frontend, Golden Paths) chạy qua Scaffolder Action → `orchestration-api` | CLAUDE.md đã quy định: business logic không được nằm trong Backstage, FastAPI sở hữu logic Adapter/Factory/policy |
| 3 | `routers/chat.py` với `use_rag=True` | Hỏi-đáp playbook/runbook, rẻ, không cần planning |
| 4 | **Chưa tồn tại đúng nghĩa** | — |

`chat.py` hiện có `use_tools=True` nhưng bị giới hạn **1 tool call/lượt**,
không có vòng lặp lên kế hoạch — đây là một bước workflow có giới hạn, chưa
phải Tier-4 agent thật sự. Tool phá hoại (`activate_prompt`, `rag_activate`)
luôn dừng lại chờ xác nhận, không tự thực thi.

## 3. Hai trường hợp đúng nghĩa Tier 4 trong bối cảnh IDP này

### 3.1. Chẩn đoán sự cố (incident triage) qua `ai-observability-server`

Khi có cảnh báo bất thường (latency tăng đột biến, model drift, tỷ lệ lỗi
cao), nguyên nhân gốc có thể đến từ nhiều nguồn khác nhau (data drift,
version model lỗi, hạ tầng, downstream chậm...) và tổ hợp giữa chúng không
liệt kê trước được. Agent cần tự quyết định kiểm tra metric nào trước, có
cần log không, so sánh baseline nào, dừng khi nào là đủ bằng chứng.

**Nguyên liệu đã có sẵn (nhưng chưa nối thành agent):**

- `ai-observability-server` (đọc-only, chỉ domain ML/LLM): `check_model_latency`,
  `get_model_metrics`. Tool hạ tầng (`query_metric`, `check_pod_status`,
  `get_logs`) đã chuyển sang MCP server sẵn có của OpenChoreo (Control Plane /
  Observability Plane) — xem `agents/mcp-servers/ai-observability-server/README.md`.
- `routers/monitoring.py` — Golden Path "Setup Model Monitoring": tạo Argo
  CronWorkflow (`monitor-drift-golden-path`) chạy định kỳ, kiểm tra drift,
  rẽ nhánh **cố định** giữa `alert-only` và `auto-retrain`.

**Khoảng trống (chưa implement):** giữa lúc CronWorkflow phát hiện bất
thường và lúc chọn `alert-only`/`auto-retrain`, hiện không có bước điều tra
nguyên nhân — chỉ so ngưỡng rồi rẽ nhánh nhị phân. Một agent Tier 4 có thể
chèn vào đúng chỗ này:

1. CronWorkflow phát hiện bất thường → trigger agent thay vì rẽ nhánh mù quáng.
2. Agent dùng tool đọc của `ai-observability-server` để tự lên kế hoạch
   điều tra (số bước/thứ tự thay đổi tùy triệu chứng ban đầu).
3. Agent kết luận nguyên nhân rồi **đề xuất** Golden Path phù hợp (retrain,
   rollback, scale...) — việc thực thi vẫn phải qua xác nhận, giống cách
   `activate_prompt`/`rag_activate` đang bị chặn trong `chat.py`.

Đây là một **lớp chẩn đoán mới**, không thay thế Golden Path nào — dùng để
chọn đúng Golden Path thay vì chọn nhầm theo ngưỡng cứng.

### 3.2. Agent "gợi ý Golden Path" từ yêu cầu tự do (free-text)

Khi người dùng mô tả nhu cầu bằng ngôn ngữ tự nhiên không theo cấu trúc cố
định, agent phải tự diễn giải ý định, chọn Golden Path nào trong
`templates/<golden-path-name>/template.yaml` (repo frontend) và tham số hoá
nó — input mới lạ mỗi lần, không liệt kê hết nhánh rẽ trước được như trong
Scaffolder Action.

## 4. Ranh giới quan trọng cần giữ

- Ngay cả ở 2 case trên, agent chỉ nên đứng ở giai đoạn **phân tích/quyết
  định**. Giai đoạn **thực thi** (deploy, activate, rollback, retrain) vẫn
  phải đi qua Scaffolder Action / Golden Path xác định trước.
- Toàn bộ luồng thực thi Golden Path (train→track→register,
  register→deploy...) là hữu hạn, biết trước các bước → phải giữ nguyên
  Tier 2, không "agent hoá" chỉ vì có thể.
- IDP tồn tại vì audit trail. Mỗi bước chuyển từ Tier 2 sang Tier 4 là đánh
  đổi audit trail lấy sự linh hoạt — nên giữ blast radius của "agent tự
  quyết" càng nhỏ càng tốt: giới hạn số vòng lặp, log mọi tool call ở phía
  MCP, tool phá hoại luôn cần xác nhận.

## 5. Đánh giá mức độ Agentic-ready hiện tại

### 5.1. MCP — điểm mạnh và khoảng trống

**Điểm mạnh:** dùng MCP SDK thật (`mcp==2.0.0`, transport `streamable-http`
chuẩn hiện đại), discovery qua Backstage Catalog thay vì hardcode
(`catalog_client.py`), tool annotation (`destructive_hint`) gate hành động
phá hoại, service identity riêng qua Thunder client_credentials token
fetch/cache (`llmops-golden-paths-server/thunder_client.py`, hàm `auth_headers()`)
thay vì self-reported header.

**Khoảng trống:**

1. **Lỗ hổng "confused deputy"** — `destructive_hint` chỉ là gợi ý cho
   client (`chat.py` chặn hộ), bản thân MCP server không tự enforce. Ai gọi
   thẳng vào endpoint `streamable-http` sẽ bypass hoàn toàn bước xác nhận.
2. **Chỉ dùng primitive Tools** — chưa dùng Resources (dữ liệu đọc, không
   cần LLM suy luận để quyết định gọi, cache/subscribe được) hay Prompts
   (template runbook dùng lại).
3. **Annotation còn thiếu** — mới có `read_only_hint`/`destructive_hint`,
   chưa có `idempotent_hint`/`open_world_hint`.
4. **Vòng lặp tool-call cứng 1 lần/lượt** trong `chat.py` — chặn khả năng
   làm case chẩn đoán sự cố đa bước ở §3.1.

### 5.2. Skills — khoảng cách giữa lời hứa kiến trúc và triển khai

Playbook ghi `Agent-ready: MCP servers + Skills` là một trụ cột thiết kế,
nhưng `agents/skills/evaluate_drift.py` hiện chỉ là **một hàm Python dùng
chung** giữa MCP tool và FastAPI route (tái sử dụng code, DRY) — không phải
"Agent Skill" đúng nghĩa (không có `SKILL.md`, không có metadata
`name`/`description` để agent tự khám phá/nạp động, không tách khỏi action
layer). Khác biệt cần phân biệt: **MCP Tool** = hành động agent *gọi*,
**Skill** = kiến thức/runbook agent *nạp* để biết cách làm.

### 5.3. Multi-agent readiness

Repo đã có 2 persona khác domain (`mlops`, `k8s` — xem
`.state/llmops-registry.json`), đúng hình dạng ranh giới multi-agent tự
nhiên. Nhưng `chat.py:88-92` gọi `registry.list_tools()` **không lọc theo
persona** — persona `k8s` tuyên bố "read-only" trong prompt text nhưng vẫn
được cấp full tool list, kể cả tool phá hoại. Đây vừa là lỗ hổng, vừa là
điểm rẻ nhất để bắt đầu multi-agent-ready thật: persona-scoped tool access
chính là tiền đề của bất kỳ pattern multi-agent nghiêm túc nào.

**Ranh giới quan trọng cần giữ (mục tiêu đã chốt): IDP làm cho
multi-agent *thân thiện*, IDP không tự xây multi-agent system.** Nghĩa là:

- IDP chịu trách nhiệm expose từng domain thành capability **tách biệt,
  có scope, discover được** — đúng như `llmops-golden-paths-server` (worker hành
  động) và `ai-observability-server` (worker chẩn đoán) đã tách sẵn.
  Đây là điều kiện để **một hệ multi-agent bên ngoài** (framework nào đó,
  do người khác xây, hoặc một hướng mở rộng sau này) có thể gán mỗi domain
  cho một agent riêng, mà IDP không cần biết/quan tâm có bao nhiêu agent
  đang gọi vào hay chúng điều phối nhau ra sao.
- IDP chịu trách nhiệm đảm bảo: dù có bao nhiêu agent bên ngoài gọi vào,
  mỗi agent chỉ thấy đúng tool/resource trong scope của nó (Phase 1), và
  mọi hành động phá hoại đều qua đúng **1 điểm gate duy nhất** bất kể agent
  nào gọi (Phase 0) — đây là "hợp đồng nền tảng" IDP giữ, không phải logic
  điều phối.
- **IDP không tự viết lead agent, không tự quyết định thứ tự worker nào
  chạy trước** — việc phân rã task, điều phối, đó là trách nhiệm của bên
  tiêu thụ (consumer), không phải của platform.

Ví von: IDP đóng vai trò như hạ tầng cloud (AWS) đối với một multi-agent
system — cấp API có scope rõ, auth rõ, discoverable rõ — chứ không tự là
dàn nhạc trưởng điều phối agent.

**Vì sao IDP không nên tự xây cả orchestrator lẫn multi-agent framework:**
nghiên cứu hiện tại (Anthropic's multi-agent research system, Cognition's
"Don't Build Multi-Agents") cho thấy multi-agent tốn 4-15x token, lỗi cộng
dồn qua từng agent, khó audit — đây là chi phí của bên *xây* multi-agent
system, không phải chi phí của việc *làm nền tảng thân thiện* với nó. Giữ
đúng ranh giới này vừa tránh chi phí đó, vừa đúng tinh thần playbook:
"chiều sâu > chiều rộng".

### 5.4. Memory vs Knowledge — Knowledge đã có, Memory thì chưa

Phân biệt quan trọng: **Knowledge** = thông tin thực tế/chuyên môn kéo vào
model (RAG) — đã có, đã dùng tốt. **Memory** = lịch sử tương tác, tool
output, trạng thái của chính agent qua thời gian — **chưa tồn tại**.

**Phát hiện cụ thể:** `ChatRequest.session_id: str | None = None` được
khai báo trong `chat.py`, nhưng **không xuất hiện lại lần nào** trong thân
hàm `send_message()`. Mỗi request build `messages` mới hoàn toàn từ đầu —
có "móc" cho continuity (tên field đặt sẵn) nhưng chưa nối. Người dùng hỏi
tiếp "activate cái version vừa rồi đi" sẽ không hoạt động vì không có gì
được nhớ giữa các lượt.

**Hạ tầng cần cho Memory cơ bản đã có sẵn, không cần thêm công cụ mới:**
`QdrantAdapter` + `llm_gateway_adapter.embed()` hiện chỉ dùng cho RAG tài
liệu — nếu cần Semantic Experience Memory sau này, chỉ cần thêm 1
collection Qdrant khác, tái dùng đúng adapter đã có.

**Việc nên làm nếu quyết định làm Memory (mức đơn giản nhất trước):**
rolling context window theo `session_id` — lưu message history, cắt khi
vượt ngưỡng token (FIFO) — không cần BM25/semantic search/GraphRAG ở bước
đầu.

**Chủ động không làm GraphRAG/Dynamic Knowledge Graph:** RAG hiện tại là
hỏi-đáp trên vài tài liệu playbook/runbook, chưa có bài toán "nối thông
tin rải rác, suy luận multi-hop" mà GraphRAG giải quyết. Đây là lựa chọn
đã cân nhắc và từ chối, không phải bỏ sót — cùng tinh thần từ chối
Graph/multi-agent framework/semantic tool selection ở các mục trên.

**Note-taking + Reflection cho case §3.1:** case chẩn đoán sự cố là bài
toán "cần độ chính xác cao" (giống tinh thần tài chính/y tế) — nên kết hợp
2 kỹ thuật: **note-taking** (agent ghi lại quan sát sau mỗi tool call
trước khi suy luận tiếp) + **Reflection** (tự phê bình/kết luận trước khi
đề xuất Golden Path) — thay vì suy luận trực tiếp từ context thô. Lưu ý:
Evaluate Gate (`evaluations/llm_judge.py` + `gate.py`) đã là một dạng
Reflection, nhưng áp dụng ở tầng *governance của artifact* (đánh giá
version trước khi activate), không phải tầng *suy luận trong 1 lượt chat*
— hai chỗ khác nhau, không nên nhầm lẫn.

## 6. Kế hoạch triển khai từng bước (Roadmap tổng hợp)

> Gộp toàn bộ phân tích ở mục 1-5 thành lộ trình có thứ tự ưu tiên. Mỗi
> phase là bước tối thiểu cần để agentic-ready, không phải build sẵn cho
> tương lai xa chưa có nhu cầu thật.

### Phase 0 — Vá lỗ hổng an ninh (làm ngay, độc lập với roadmap)

- [ ] `llmops-golden-paths-server` tự verify token/scope trước khi thực thi tool
      `destructive_hint=True`, không dựa hoàn toàn vào `chat.py` chặn hộ.
      File: `agents/mcp-servers/llmops-golden-paths-server/server.py`,
      `thunder_client.py`.

### Phase 1 — Single-agent-ready: sửa đúng cái đã có (rẻ, nền tảng cho mọi phase sau)

- [ ] Filter `registry.list_tools()` theo persona trong `chat.py:88-92` —
      để lời hứa "read-only" của persona `k8s` là thật, không chỉ nằm
      trong prompt text.
- [ ] Bổ sung `idempotent_hint`/`open_world_hint` cho tool hiện có.
- [ ] Mở vòng lặp tool-call từ cứng-1-lần thành bounded multi-turn (ví dụ
      max 3-5 vòng) — chỉ cần khi Phase 3 (case chẩn đoán) triển khai thật.

**Đặc tả vòng lặp — dùng đúng thuật ngữ, không nhảy thẳng lên pattern phức tạp:**

- **Agent Type:** §3.1 (chẩn đoán sự cố) = **ReAct Agent** (quan sát → suy
  luận → gọi tool → quan sát → quyết định tiếp) — `chat.py` hiện tại chỉ là
  Reflex Agent (1 quyết định, không lặp). §3.2 (gợi ý Golden Path) =
  **Planner-Executor** — Plan (diễn giải ý định, chọn/tham số hoá Golden
  Path, không gọi tool) tách biệt Execute (gọi đúng 1 lần Scaffolder) —
  khớp thẳng ranh giới đã chốt ở mục 4 ("agent chỉ phân tích/quyết định,
  thực thi qua Scaffolder").
- **Tool Selection:** không cần Semantic/Hierarchical Tool Selection kiểu
  FAISS — chỉ ~12 tool, nằm gọn trong Standard Tool Selection. Persona
  (`operator`/`observer`) đã tự nhiên cho hiệu ứng Hierarchical 2 tầng
  (persona chọn nhóm, LLM chọn tool trong nhóm) mà không cần hạ tầng
  embedding riêng.
- **Tool Topology:** dùng **Chain** cho §3.1 (tuần tự, output bước trước
  → input bước sau), có thể **Parallel** ở bước đầu cho các tool đọc độc
  lập (`check_model_latency` ở `ai-observability-server`; `get_logs`,
  `query_metric` ở MCP server của OpenChoreo — đều
  `read_only_hint=True`, không phụ thuộc nhau) để giảm latency trước khi
  vào phần suy luận tuần tự. §3.2 chỉ cần **Single Tool** mỗi pha. **Không
  dùng Graph/LangGraph** — không case nào có nhánh điều kiện/điểm hợp nhất
  phức tạp đủ để cần nó; đây là lựa chọn có chủ đích, không phải bỏ sót.
- **Context Engineering:** khi vòng lặp nhiều bước thật sự chạy, cần thêm
  bước tóm tắt/cắt bớt tool-output (đặc biệt `get_logs` — hiện là mock
  ngắn, nhưng log K8s thật có thể rất dài) trước khi nhét lại vào
  `messages` cho vòng gọi tiếp theo, tránh phình context qua 3-5 vòng lặp.

### Phase 2 — Resources: agent "nạp" trạng thái mà không tốn lượt LLM suy luận

- [ ] Chuyển tool chỉ-đọc/không cần suy luận tham số từ Tool sang
      Resource (Resource Template nếu có tham số định vị), trong
      `ai-observability-server`:

  | Tool hiện tại | Resource đề xuất |
  |---|---|
  | `list_experiments()` | `mlflow://models` |
  | `get_model_metrics(name, version)` | `mlflow://models/{name}/versions/{version}/metrics` |

  Giữ nguyên Tool: `check_model_latency` (cần LLM soạn tham số/tổng hợp — đúng
  bản chất "cần suy luận"). `query_metric`/`check_pod_status`/`get_logs` giờ
  thuộc MCP server của OpenChoreo, không còn ở đây.
- [ ] Mở rộng `McpToolRegistry` (`mcp_client.py`) gọi thêm
      `session.list_resources()` khi connect (hiện chỉ có
      `session.list_tools()` ở dòng 105), thêm map tương tự
      `_tool_to_server` cho resource.

### Phase 3 — Prompts: KHÔNG xây registry mới, chỉ thêm cổng đọc lên registry đã có

**Sửa lại hiểu nhầm đã có ở bản nháp trước:** MCP Prompts primitive không
phải một hệ thống runbook tách biệt cần viết mới — Prompt Registry
(`routers/prompts.py` + `MlflowPromptRegistryAdapter(kind="prompt")`)
**đã tồn tại đầy đủ** (draft → evaluate-gate → activate), và đã
agent-accessible qua 3 tool có sẵn trong `llmops-golden-paths-server`
(`draft_prompt`, `evaluate_prompt`, `activate_prompt`). Đây là registry
quản lý *system prompt của persona* ("tôi là ai"); MCP Prompts đề xuất
thêm là *task template/runbook* ("gặp tình huống X thì làm theo bước Y") —
khác mục đích, nhưng nên dùng chung một registry, không tách hai hạ tầng.

- [ ] Thêm field phân loại vào metadata registry (`category: "persona"` vs
      `"task-template"`) — không cần bảng/adapter mới, chỉ mở rộng
      `MlflowPromptRegistryAdapter`'s metadata/tags hiện có.
- [ ] Draft/evaluate/activate runbook đầu tiên (ví dụ
      `diagnose_latency_spike`) bằng **đúng 3 tool đã có sẵn** — không viết
      tool mới.
- [ ] Thêm cổng đọc mới: `prompts/list`/`prompts/get` ở phía MCP server,
      lọc registry theo `category="task-template"` và đã active — đây là
      phần thật sự mới, chỉ là lớp đọc, không phải subsystem riêng.
- [ ] Mở rộng `McpToolRegistry` hỗ trợ `prompts/list`/`prompts/get`.

### Phase 4 — Skills → Software Catalog: một skill làm mẫu, chưa làm cả taxonomy

- [ ] Đăng ký `evaluate_drift.py` như Catalog entity: `kind: Resource,
      spec.type: skill`, annotation `skill/path` — theo đúng convention
      tự-đặt-type đã dùng cho `mcp` (`examples/entities.yaml:88-90`).
- [ ] Thêm `discover_skills()` vào `catalog_client.py` — đơn giản hơn MCP
      discovery vì skill là file tĩnh cùng repo, không cần giữ session
      sống.
- [ ] Wire vào `chat.py` (hoặc agent loader riêng): khi persona/topic
      khớp, nạp nội dung skill vào context.

### Phase 5 — Multi-agent-*friendly*: IDP expose scope rõ, KHÔNG tự xây orchestrator

> Mục tiêu đã chốt: IDP làm nền tảng thân thiện với multi-agent, không tự
> là một multi-agent system. IDP không viết lead agent, không tự điều
> phối worker — chỉ đảm bảo mọi agent-caller bên ngoài đều được scope
> đúng, discover đúng, và đi qua đúng 1 gate xác nhận.

- [ ] Tiền đề bắt buộc: Phase 1 (persona → tool scope) phải xong trước —
      điều kiện để mỗi identity/agent-caller bên ngoài thật sự bị giới hạn
      đúng phạm vi của nó, không chỉ khác system prompt.
- [ ] Đảm bảo `llmops-golden-paths-server` (capability hành động) và
      `ai-observability-server` (capability chẩn đoán) **tiếp tục là 2
      capability tách biệt, độc lập scope/auth** — để một hệ multi-agent
      bên ngoài (bất kỳ ai xây, bất kỳ framework nào) có thể gán mỗi domain
      cho một agent riêng mà không cần đụng vào code của IDP.
- [ ] Dù có bao nhiêu agent bên ngoài gọi vào cùng lúc, mọi đường đến tool
      `destructive_hint=True` vẫn phải qua đúng 1 điểm chờ xác nhận ở phía
      IDP (Phase 0) — đây là hợp đồng nền tảng IDP giữ, không phải logic
      điều phối multi-agent.
- [ ] Ghi rõ trong playbook ranh giới này (IDP thân thiện với multi-agent,
      không tự xây multi-agent system) và lý do — dùng để trả lời phản
      biện khi bảo vệ đề tài, tránh hội đồng hiểu nhầm thành "đề tài tự
      nhận build multi-agent framework".

### Phase 6 — Nhận biết xu hướng, chưa cần implement (kiến thức nền bảo vệ đề tài)

- [ ] A2A (Agent2Agent, Google) — bổ sung cho MCP: MCP là agent-to-tool, A2A
      là agent-to-agent. Chỉ cần biết thuật ngữ — hệ thống hiện chưa có
      agent-to-agent thật (mới 1 client duy nhất: `orchestration-api`).
- [ ] Tracing cho chuỗi tool-call đa bước (OpenTelemetry GenAI semantic
      conventions) — chỉ cần khi Phase 3/5 thật sự chạy multi-step trong
      production; Prometheus/Grafana hiện đã đủ cho metric tổng quát.

### Bảng tổng hợp

| Phase | Mục tiêu | Effort | Điều kiện tiên quyết |
|---|---|---|---|
| 0 | Vá an ninh MCP server | Thấp | Không |
| 1 | Persona scoping + annotation đầy đủ | Thấp | Không |
| 2 | Resources cho dữ liệu đọc | Trung bình | Phase 1 |
| 3 | Prompts = cổng đọc mới lên registry đã có | Trung bình | Phase 1, 2 |
| 4 | Skill đầu tiên vào Catalog | Trung bình | Không phụ thuộc 1-3 |
| 5 | Multi-agent-*friendly* (IDP không tự xây orchestrator) | Trung bình | Phase 1 bắt buộc |
| 6 | Kiến thức A2A/tracing | — | Chỉ đọc, chưa implement |

## 7. Câu hỏi mở cần nghiên cứu thêm

- [ ] Vòng lặp planning nhiều bước nên bounded bằng gì — số lần lặp cố
      định, budget token, hay timeout?
- [ ] Agent chẩn đoán sự cố nên chạy đồng bộ (chặn CronWorkflow chờ kết
      quả) hay bất đồng bộ (agent chạy song song, người vận hành xem kết
      quả sau)?
- [ ] Kết quả đề xuất của agent nên hiển thị ở đâu — Backstage UI, Slack,
      hay tạo PR/ticket tự động?
- [ ] `category: "task-template"` nên là field riêng trong metadata, hay
      tách hẳn `kind="runbook"` mới trong registry (Qdrant)
      để không lẫn với danh sách persona ở `GET /prompts`?
