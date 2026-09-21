# Plan: DORA cho MLOps/LLMOps — Delivery Insights

> Kế hoạch áp dụng 4 insight DORA-for-ML/LLM vào dashboard Delivery Insights.
> Trạng thái: đề xuất, chưa triển khai. Nguồn insight: bài DORA for ML Model
> Deployment (xem mục Tham khảo).

## 1. Bối cảnh

Dashboard Delivery Insights hiện tại là DORA generic cho software:

| Thành phần | Vị trí |
| :--- | :--- |
| API contract (mock observer) | `services/orchestration-api/routers/mock_observer.py` |
| UI (4 tile, 4 chart, breakdown) | `plugins/openchoreo-observability/src/components/DeliveryInsights/` |
| Nguồn dữ liệu | synthetic + captured Golden Path runs |

4 metric đang có: Deployment Frequency, Lead Time, Change Failure Rate, MTTR.
Phân hạng Elite/High/Medium/Low theo ngưỡng generic, chưa có chiều ML/LLM.

## 2. Khoảng trống

| Insight | Đã có | Còn thiếu |
| :--- | :--- | :--- |
| Tách 2 luồng deploy (infra vs model) | 1 luồng chung | `changeType`, tách tile |
| Tần suất cao ≠ tốt | — | `evalCoverage`, `driftTriggered` |
| Lead time tách phase | `steps` (train, register) | phase data/eval/deploy, breakdown |
| Semantic failure | `failedBy`, `failureReason` | `failureClass`, `semanticType`, eval/drift score |
| Silent failure | — | badge + continuous eval |
| Recovery strategy | — | `recoveryStrategy` (rollback/fallback/guardrail/retrain) |

## 3. Nguyên tắc thiết kế

- Giữ 4 metric, chỉ thêm dimension và phân loại — không thay cấu trúc.
- Sửa contract backend (`mock_observer.py`) và `types.ts` song song.
- Prototype trên mock observer trước khi có observability plane thật.

## 4. Thiết kế theo từng metric

### 4.1 Delivery Frequency

- Backend: thêm `changeType: infra|model|rag_index|prompt`, `driftTriggered`, `evalCoverage`.
- Frontend: tách tile 2 luồng hoặc stacked bar; panel "drift-driven cadence".

### 4.2 Lead Time

- Backend: capture phase `data_prep|train|eval|deploy`; thêm `leadTimeBreakdown`, `evalBottleneck`.
- Frontend: waterfall chart; badge "human-in-the-loop".

### 4.3 Change Failure Rate

- Backend: `failureClass: infra|semantic`; `semanticType: accuracy_drop|drift|hallucination|guardrail|prompt_injection`; `evalScore`, `baselineScore`, `driftScore`; tách `infraCfr`/`semanticCfr`.
- Frontend: tách tile infra vs semantic; badge "silent failure".

### 4.4 MTTR

- Backend: `recoveryStrategy: rollback|fallback|guardrail|retrain`.
- Frontend: subtext strategy chiếm ưu thế; cảnh báo khi retrain kéo dài MTTR.

## 5. Cross-cutting

- `DoraDataAvailability`: thêm `evalPipeline`, `driftMonitor`, `guardrails`.
- Ngưỡng phân hạng riêng cho ML/LLM (hiện generic ở `mock_observer.py:250-280`).
- Thêm `modelVersion`, `promptVersion`, `ragIndexVersion` vào `DoraDeployment`.

## 6. Lộ trình

| Phase | Việc | Phạm vi |
| :--- | :--- | :--- |
| 1 | Mở rộng schema + mock generator | backend |
| 2 | Cập nhật `types.ts` + tách tile/chart | frontend |
| 3 | Mở rộng capture script (eval/drift/guardrail) | script |
| 4 | Nối real observer | backend |

## 7. Rủi ro & trade-off

- Thêm dimension làm phức tạp contract mock/observer.
- Ngưỡng ML/LLM cần dữ liệu thật để hiệu chỉnh.
- Semantic failure cần nguồn eval/drift chưa có trên cluster.

## 8. Câu hỏi mở

- Ngưỡng phân hạng cho semantic CFR?
- Nguồn drift/guardrail lấy từ đâu?
- Phạm vi MVP: chỉ backend hay cả frontend?

## Tham khảo

- [DORA Metrics for ML Model Deployment](https://medium.com/womenintechnology/dora-metrics-for-ml-model-deployment-how-software-delivery-performance-applies-to-mlops-pipelines-d7500b3c914d)
- `services/orchestration-api/routers/mock_observer.py`
- `scripts/capture-delivery-insights-runs.sh`
- `plugins/openchoreo-observability/src/types.ts`