# Theory-to-Practice Project Flow Diagrams

This document summarizes the end-to-end journey of the **NOTDOG YESDOG** project from concept to deployment.

---

## 1) From Theory to Practical Product

```mermaid
flowchart TD
    A[Problem Definition<br/>Dog vs Not-Dog + Breed Classification] --> B[Research and Theory<br/>CNNs, Transfer Learning, Metrics]
    B --> C[Success Criteria<br/>Accuracy, Latency, Robustness]
    C --> D[Dataset Strategy<br/>Stanford Dogs + Non-Dog Samples]
    D --> E[Data Engineering<br/>Preprocess, Balance, Augmentation]
    E --> F[Modeling<br/>Binary Model + Breed Model]
    F --> G[Validation<br/>Cross-Validation, Class Metrics, Bias Checks]
    G --> H[System Design<br/>Hierarchical Inference API]
    H --> I[User Experience<br/>Frontend Upload + Predictions]
    I --> J[Deployment Readiness<br/>Artifacts, Config, Monitoring]
    J --> K[Production Deployment<br/>API + Frontend + Model Hosting]
    K --> L[Continuous Improvement<br/>Error Analysis and Retraining]
```

---

## 2) ML Pipeline (Data to Trained Models)

```mermaid
flowchart LR
    A[Raw Images] --> B[Data Preprocessing]
    B --> C[Label Mapping and Splits]
    C --> D[Balancing Strategy]
    D --> E[Targeted Augmentation]
    E --> F[Train Binary Classifier<br/>EfficientNet-B3]
    E --> G[Train Breed Classifier<br/>ResNet50 - 119 classes]
    F --> H[Binary Evaluation]
    G --> I[Breed Evaluation]
    H --> J{Quality Gate}
    I --> J
    J -- Pass --> K[Export Artifacts<br/>.pth / ONNX]
    J -- Fail --> L[Hyperparameter Tuning]
    L --> F
    L --> G
```

---

## 3) Full Product Flow (User Request to Response)

```mermaid
flowchart TD
    A[User Uploads Image<br/>React or Static Frontend] --> B[FastAPI Endpoint]
    B --> C[Stage 1: Dog Detection<br/>DOG vs NOT-DOG]
    C --> D{Is Dog?}
    D -- No --> E[Return NOT-DOG Response]
    D -- Yes --> F[Stage 2: Breed Classification]
    F --> G[Top-k Predictions + Confidence]
    G --> H[Response Formatting]
    H --> I[Frontend Visualization]
    I --> J[Optional Logging / Analytics]
```

---

## 4) Engineering Loop (Versioning to Release)

```mermaid
flowchart TD
    A[Define Iteration Goal] --> B[Implement Code Changes]
    B --> C[Run Local Tests and Validation]
    C --> D[Commit to Git]
    D --> E[Update Docs and Evidence]
    E --> F[Push to GitHub]
    F --> G[Review Metrics and Feedback]
    G --> H{Need Improvements?}
    H -- Yes --> A
    H -- No --> I[Tag Release / Deploy]
```

---

## Suggested Use in Presentations

- Diagram 1: Executive overview of the complete project journey.
- Diagram 2: Technical MLOps/training narrative.
- Diagram 3: Product behavior from end-user perspective.
- Diagram 4: Team workflow and delivery maturity.
