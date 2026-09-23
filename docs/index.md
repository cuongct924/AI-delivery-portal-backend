# AI Delivery Portal — Documentation

Internal Developer Platform (IDP) cho vòng đời MLOps/LLMOps, xây trên
Backstage + OpenChoreo. Trang này là điểm vào cho tài liệu kiến trúc, vận hành
và troubleshooting của nền tảng.

## Bắt đầu

- [Architecture Overview](architecture-overview.md) — ba plane (control /
  portal / workflow), Golden Paths, Adapter Pattern, chat assistant.
- [User Flow Diagrams](diagram.md) — sơ đồ luồng nghiệp vụ.
- [Glossary](glossary-openchoreo.md) — thuật ngữ OpenChoreo/MLOps/LLMOps.

## Vận hành & xử lý sự cố

- [Troubleshooting FAQ](troubleshooting-faq.md) — lỗi thường gặp và cách xử lý.
- [Agent Integration Notes](notes-agent-integration-idp.md) — tích hợp agent/MCP.

## Ghi chú

- [Defense Playbook (personal notes)](playbook-ai-delivery-portal.md) — ghi chú
  bảo vệ đồ án, không phải tài liệu vận hành.

## Golden Paths

| Golden Path | Template | Mục đích |
|---|---|---|
| Train → Track → Register | `train-track-register` | Huấn luyện/fine-tune, log MLflow, đăng ký model version |
| Evaluate → Deploy | `evaluate-deploy-model` | Evaluate Gate rồi deploy/rollback/promote |
| Set Up Model Monitoring | `setup-model-monitoring` | Lịch drift check định kỳ |
| Draft Prompt / Ingest RAG / Draft Eval Set | `llm-draft-ingest` | Tạo version prompt/RAG/eval-set |
| Evaluate & Activate Prompt / RAG | `llm-evaluate-activate` | LLM-as-judge gate rồi activate/rollback |
| Serve LLM (Self-hosted) | `llm-serve-deploy` | Deploy vLLM qua KServe |
