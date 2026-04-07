REVA — Automated Medical Rash Image Processing Pipeline
REVA is a production-grade, cloud-native pipeline for automated ingestion, anonymisation, and classification of medical rash images and associated clinical metadata. Built on AWS, it eliminates manual review bottlenecks and enables scalable, reproducible processing of multimodal clinical datasets.

Overview
Medical imaging workflows often involve large volumes of unstructured, sensitive data that require consistent preprocessing, anonymisation, and triage before clinical or research use. REVA automates this end-to-end, from raw image ingestion to classified, audit-ready outputs — reducing manual review effort by 90%.

Architecture
Raw Images & Metadata
        │
        ▼
   AWS S3 (Ingestion)
        │
        ▼
  AWS Lambda (Orchestration)
        │
   ┌────┴────┐
   ▼         ▼
Anonymisation  Classification Model
   │               │
   └────┬──────────┘
        ▼
   Processed Output (S3)
        │
        ▼
 MLflow (Experiment Tracking & Model Versioning)
        │
        ▼
 Real-Time Monitoring Dashboard

Key Features

Automated ingestion — Handles multimodal inputs (images + clinical metadata) at scale via S3 event triggers
Anonymisation — Strips patient-identifiable information before any downstream processing
Classification — ML model pipeline for automated rash image triage and categorisation
Reproducible MLOps — Full experiment tracking, model versioning, and CI/CD via MLflow and Docker
Real-time monitoring — Dashboard for transparent, research-grade data validation across the pipeline


Tech Stack
LayerToolsCloud & OrchestrationAWS Lambda, AWS S3ContainerisationDockerML & Experiment TrackingMLflowCI/CDGitHub ActionsLanguagePython

Outcomes

90% reduction in manual clinical image review effort
Fully reproducible pipeline with end-to-end experiment logging
Scalable to large multimodal clinical datasets


Author
Ranjani Venkatesan
M.S. Data Science, University of Connecticut
ranjuvenkat19@gmail.com | GitHub
