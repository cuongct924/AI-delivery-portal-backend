# Hướng phát triển tiếp theo: chuyển hẳn sang OpenChoreo

Ghi lại lộ trình đã thiết kế (nhưng chưa thực hiện, tạm hoãn theo quyết
định của người dùng ngày 2026-09-15) để đưa toàn bộ nền tảng dùng
OpenChoreo làm control plane chính, thay vì chỉ dùng cho auth (Thunder)
như hiện tại.

## Hiện trạng (đã xác nhận thật, không suy đoán)

**Đã là OpenChoreo-native:**
- Auth: Thunder IdP (`auth/thunder.py`, `core/config.py`).
- `Project`/`Environment`/`DeploymentPipeline` CRD chạy thật trên cluster
  `k3d-openchoreo-quick-start`.
- `orchestration-api` và `fraud-detection-training` (CronJob) đang chạy
  như OpenChoreo `Component`/`Workload` thật.

**Vẫn còn bespoke/gọi thẳng K8s, cần chuyển:**
- `adapters/argo_adapter.py` — gọi thẳng Argo Server REST API, bỏ qua
  `ClusterWorkflowPlane` của OpenChoreo (đã có sẵn, đang dùng cho build
  image, chưa dùng cho training).
- `adapters/kserve_adapter.py` — gọi thẳng K8s `CustomObjectsApi` cho
  `InferenceService`; KServe **chưa cài** trên cluster (0 CRD).
- MLflow/Qdrant/LiteLLM/Feast — chạy plain docker-compose, chưa khai báo
  thành OpenChoreo `Resource`/`ResourceType`.
- `fraud-detection-serving` Component/Workload — placeholder
  (`nginxdemos/hello`), chưa nối với model thật.
- 8 template Golden Path (repo frontend) gọi HTTP thẳng tới
  orchestration-api — **không cần đổi**, vì nghiệp vụ đúng chỗ (backend);
  việc "OpenChoreo hoá" nằm ở tầng adapter bên dưới, không phải ở
  Scaffolder actions.

## Nguyên tắc sắp xếp thứ tự

Theo rủi ro/phụ thuộc thực tế, không theo repo:

```
Phase 0 (khôi phục tài liệu)      ─┐
Phase 1 (Resource wrap, 0 rủi ro) ─┼─ độc lập, làm trước, không cần hạ tầng mới
Phase 6 (frontend config/catalog) ─┘
        │
Phase 2 (Training → OpenChoreoWorkflowAdapter)   — không cần cài gì mới
        ▼
Phase 3 (cài KServe CRD thật, chỉ CPU)           — hạ tầng mới, rủi ro cao hơn
        ▼
Phase 4 (ClusterComponentType + OpenChoreoInferenceAdapter cho serving)
        ▼
Phase 5 (GPU/quota Trait — chỉ thiết kế, không verify)
```

## Phase 0 — Khôi phục tài liệu trạng thái OpenChoreo

`infra/openchoreo/README.md` + `2.3-notes.md` đã bị xoá ở commit `15b2fc0`
dù các YAML khác vẫn trỏ tới chúng theo path. Khôi phục từ
`git show 6c76083:infra/openchoreo/README.md`, viết lại theo đúng tiến độ
thật của từng Phase dưới đây — không ghi trước điều chưa xong. Cập nhật
luôn sơ đồ component trong `docs/playbook-ai-delivery-portal.md`.

**Kiểm chứng:** mọi tham chiếu `infra/openchoreo/README.md`/`2.3-notes.md`
trong comment YAML/Python đều trỏ tới file thật tồn tại.

## Phase 1 — Bọc MLflow/Qdrant/LiteLLM/Feast thành OpenChoreo `Resource`/`ResourceType`

Chỉ là khai báo metadata cho Catalog OpenChoreo thấy — không đổi hành vi,
không đổi port, không đổi code adapter.

- Mới: `infra/openchoreo/platform/resourcetype-external-service.yaml` — 1
  `ClusterResourceType` chung. **Trước khi viết, đọc**
  `kubectl get clustercomponenttype service -o yaml` trên cluster thật để
  theo đúng schema CEL-template, tránh đoán mò.
- Mới: `resource-mlflow.yaml`, `resource-qdrant.yaml`, `resource-litellm.yaml`,
  `resource-feast.yaml` dưới Project `platform`, trỏ URL
  `host.docker.internal:<port>` giống `workload-orchestration-api.yaml`.
  Chỉ áp `development`.
- Không đổi `adapters/mlflow_adapter.py`/`vector_db_adapter.py`/
  `llm_gateway_adapter.py`/`feature_store_adapter.py`.

**Kiểm chứng:** `kubectl apply --dry-run=server` rồi `kubectl get resource -n default`.

**Rủi ro:** thấp nhất toàn kế hoạch.

## Phase 2 — Training: `OpenChoreoWorkflowAdapter` song song `ArgoAdapter`

Dùng REST API generic của OpenChoreo (`POST/GET /api/v1/namespaces/{ns}/workflowruns`,
xác nhận có thật qua `openchoreo-workflows-backend`'s `GenericWorkflowService.ts`
ở repo frontend). Theo đúng nguyên tắc sẵn có của repo: thêm 1 class mới,
không đụng caller (`routers/models.py`, `routers/recommendations.py` vẫn
gọi `get_workflow_adapter()` y hệt).

**3 điều cần xử lý đúng, không phải đổi tên suông:**
1. `trigger_workflow`/`get_workflow_status` khớp tốt với `workflowruns` API
   — nhưng cần viết mới 1 `ClusterWorkflow` chạy container training thật
   (4 `ClusterWorkflow` hiện có chỉ để build image).
2. Mapping trạng thái: OpenChoreo suy `Pending/Running/Succeeded/Completed/Failed`
   từ `conditions[]`. `Completed` ngụ ý "đã tạo Workload mới" — chỉ đúng
   cho workflow *build*. Training không đụng Workload sẽ dừng ở `Succeeded`
   — phải **verify bằng 1 lần trigger thật**, không giả định.
3. `create_cron_workflow` (`adapters/argo_adapter.py`, gọi trực tiếp bởi
   `routers/monitoring.py`, không qua interface) — OpenChoreo **không có**
   primitive cron-workflow tương đương. Giải pháp: implement
   `create_cron_workflow` trong adapter mới bằng cách tạo
   `Component(cronjob/scheduled-task)` + `Workload` (đúng pattern
   `component-training.yaml` đã chứng minh sống), giữ nguyên chữ ký Python.

**File thay đổi:**
- Mới: `adapters/openchoreo_workflow_adapter.py` — dùng `httpx`, auth qua
  Thunder client-credentials (`auth/thunder.py`).
- `adapters/factory.py`: thêm cờ `USE_OPENCHOREO_WORKFLOW`.
- `services/orchestration-api/core/config.py`: thêm `openchoreo_api_url`.
- Mới: `infra/openchoreo/fraud-detection/clusterworkflow-training.yaml`.
- Mới: `component-monitoring.yaml` + `workload-monitoring.yaml` (thay
  `create_cron_workflow`).

**Kiểm chứng:** `tests/test_openchoreo_workflow_adapter.py` (mock httpx,
style giống `tests/test_argo_adapter.py`), `kubectl get workflowruns`, 1
lần chạy Golden Path #1 thật, `make check`.

## Phase 3 — Cài KServe control-plane thật (chỉ CPU, không claim GPU)

Dùng **RawDeployment mode**, không phải Serverless — cluster có sẵn
`cert-manager` nhưng không có `istio`/`knative-serving`.

- Ghi lệnh cài vào `infra/openchoreo/README.md` (Phase 0).
- Mới: `infra/kserve/test-inferenceservice-cpu.yaml` (vd mẫu `sklearn-iris`).

**Kiểm chứng:** `kubectl get crds | grep kserve.io` (hiện tại rỗng), pod
controller `Running`, apply test → `Ready=True` → `curl` predict ra kết
quả thật. **Không** test GPU — Pod xin GPU `Pending` là kết quả đúng.

**Rủi ro:** dễ đụng bug reconciler đã biết (xem mục cuối file).

## Phase 4 — `ClusterComponentType` cho `InferenceService` + `OpenChoreoInferenceAdapter`

- Mới: `clustercomponenttype-inference-service.yaml` — đọc 1
  `ClusterComponentType` thật trên cluster trước khi viết.
- Sửa `component-serving.yaml`/`workload-serving.yaml`: đổi placeholder
  `nginxdemos/hello` thành `storageUri: models:/fraud-detection/<version>`.
- Mới: `adapters/openchoreo_inference_adapter.py` — patch Component/Workload
  qua `kubernetes.client.CustomObjectsApi` (giữ convention test hiện có),
  để reconciler OpenChoreo tự sinh `InferenceService`.
- **Vấn đề namespace**: `get_kserve_adapter()` hiện cứng
  `f"ai-delivery-portal-dev-{tenant}"`, nhưng namespace dataplane thật của
  OpenChoreo tự sinh khác hẳn (vd `dp-default-platform-development-8260e221`)
  — không được giả định cố định, để OpenChoreo tự resolve.

**Kiểm chứng:** mock `CustomObjectsApi` (style `tests/test_kserve_adapter.py`),
`kubectl get inferenceservice -o yaml` xác nhận do reconciler sinh ra, 1
lần chạy Golden Path #2 thật, `curl` endpoint, `make check`.

**Rủi ro:** cộng dồn rủi ro reconciler từ Phase 3. GPU/vLLM
(`deploy_llm_model`) không nằm trong phạm vi verify thật của Phase này.

## Phase 5 — GPU/quota Trait: chỉ thiết kế, không verify

OpenChoreo không có sẵn field GPU nào (`Container` chỉ có
image/command/args/env; `resources` chỉ cpu/memory mờ) — Trait này hoàn
toàn mới, tự viết.

- Mới: `infra/openchoreo/design/clustertrait-gpu-request.yaml` (thư mục
  `design/` riêng, không lẫn YAML apply thật) — schema khớp field
  `KServeAdapter.deploy_llm_model` đã nhận (`gpu_count`,
  `vllm_quantization`, `max_context_length`).
- Mới: `design/README-gpu-trait-design.md` — dòng đầu ghi rõ **"CHƯA
  VERIFY TRÊN GPU THẬT"**.

**Kiểm chứng:** chỉ `kubectl apply --dry-run=server`; nếu apply thật, chỉ
xác nhận Pod dừng ở `Pending`/`Insufficient nvidia.com/gpu` — không hơn.

## Phase 6 — Phần riêng frontend (repo `AI-delivery-portal-frontend`)

- **6a**: `app-config.yaml` thêm `orchestrationApi: { baseUrl: ${ORCHESTRATION_API_URL} }`
  — không cần sửa `mlopsActions.ts`.
- **6b**: Catalog plugin mới cho MLflow/model-registry, theo pattern
  `CostInsights` (`plugins/openchoreo-observability/`) — plugin thật sự
  mới, không phải chỉnh nhỏ.
- **6c**: đã nằm trong Phase 0.

## Rủi ro vận hành cần theo dõi, không phải "fix"

Bug reconciler đã biết của OpenChoreo:
`controller-manager`'s `DeploymentPipeline`/`ReleaseBinding→RenderedRelease`
từng bị kẹt (1 `Project` mới không tự tạo `ProjectReleaseBinding`),
workaround bằng `kubectl rollout restart deployment/controller-manager`.
Nhiều khả năng tái diễn ở Phase 3/4 (ComponentType/Component/Workload/
ReleaseBinding mới, đúng đường đi từng kẹt). Đây là bug thượng nguồn —
chấp nhận restart như bước biết trước, không phải sự cố chặn tiến độ.

## Không nằm trong phạm vi

Không MCP server nào của dự án (`golden-paths-server`,
`golden-path-guide-server`, `observability-server`) được đề xuất làm
"agent điều phối" thay thế Perch/RCA Agent/FinOps Agent có sẵn của
OpenChoreo — chúng vẫn là năng lực bổ sung, đứng song song.

## Ước lượng chi phí

| Phase | Effort | Rủi ro chính |
|---|---|---|
| 0 | 1–2h | Không |
| 1 | 2–4h | Thấp — đoán sai schema CEL |
| 2 | 8–16h | Trung bình-cao — mapping trạng thái cần verify thật |
| 3 | 4–8h (có thể vọt cao) | **Cao nhất** — bug reconciler đã biết |
| 4 | 8–16h | Cộng dồn rủi ro Phase 3 + bài toán namespace |
| 5 | 2–3h | Không (chỉ thiết kế) |
| 6a | 0.5–1h | Cần quyết định kiến trúc trước (URL production) |
| 6b | 8–16h | Plugin mới hoàn toàn |

**Tổng: ~34–66 giờ.** Khuyến nghị timebox Phase 3 (vd 1 buổi) trước khi
cam kết Phase 4, vì Phase 4 phụ thuộc hoàn toàn vào Phase 3 thành công.

## Bắt đầu lại từ đâu

Phase 0 + Phase 1 trước — không rủi ro, không cần hạ tầng mới, làm được
ngay bất cứ lúc nào.
