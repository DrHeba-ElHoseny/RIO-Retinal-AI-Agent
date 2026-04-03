flowchart TB
  A([Start: Launch RIOAgent UI]) --> B{Select Tab}

  B -->|Fine-Tuning Agent| C[Select Dataset Folder]
  C --> D[Select Output Folder (optional)]
  D --> E[Prepare: Datastore + Class Order + Split Train/Val/Test + Augment]
  E --> F[Build CNN + Compute Class Weights]
  F --> G[Stage-A Training (Adam, LR_A, schedule, checkpoints)]
  G --> H[Stage-B Fine-Tuning (LR_B, fewer epochs) or Fallback]
  H --> I[Save model: retina_classification_model.mat]
  I --> J[Evaluation: Accuracy + Confusion (row-norm+counts) + ROC/AUC + Metrics]
  J --> K[Export CSV: StageA/StageB + Comparison]
  K --> Z([End: Training/Evaluation])

  B -->|Single Image Inference| L[Load Model (.mat)]
  L --> M[Select New Image + Resize to InputSize]
  M --> N[Quality Gate: brightness/contrast/sharpness]
  N --> O[Apply SAME preprocessing as training (if enabled)]
  O --> P[Predict: classify -> scores + top-1]
  P --> Q[Policy Rules: ACCEPT/REVIEW/RETAKE]
  Q --> R[UI Output: probs table + notes + decision]
  R --> S[Log to CSV]
  S --> Y([End: Inference])
