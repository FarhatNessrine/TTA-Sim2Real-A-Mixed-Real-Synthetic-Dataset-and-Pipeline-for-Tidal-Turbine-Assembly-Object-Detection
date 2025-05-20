import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from ultralytics import YOLO

# === Paths ===
DEVICE = 0
PROJECT_NAME = "runs_yolov8_100"
BEST_MODEL_PATH = "yolov8_finetune/best_model_8_100.pt"
PLOTS_DIR = "analysis_plots_yolov8_100"
DATA_YAML = "data_yolo_test/data.yaml"  # <-- includes test: entry

os.makedirs(PLOTS_DIR, exist_ok=True)

# === Load best run name
with open("best_run_8.txt") as f:
    best_run_name = f.read().strip()

print(f"\nEvaluating best model from run: {best_run_name}")
model = YOLO(BEST_MODEL_PATH)

# === Evaluation on Annotated Test Set ===
val_metrics = model.val(data=DATA_YAML, split="Test")

mAP50 = val_metrics.box.map50
mAP5095 = val_metrics.box.map
precision = val_metrics.box.mp
recall = val_metrics.box.mr

print("\nTest Set Evaluation Results:")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"mAP@50      : {mAP50:.4f}")
print(f"mAP@50-95   : {mAP5095:.4f}")

# === Plot Training Curves ===
csv_path = os.path.join(PROJECT_NAME, best_run_name, "results.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    epochs = df.index + 1

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, df['train/box_loss'], label='Train Box Loss')
    plt.plot(epochs, df['val/box_loss'], label='Val Box Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Box Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, df['metrics/precision(B)'], label='Precision')
    plt.plot(epochs, df['metrics/recall(B)'], label='Recall')
    plt.plot(epochs, df['metrics/mAP50(B)'], label='mAP@50')
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{best_run_name}_metrics.png"))
    plt.close()
else:
    print(f"Could not find results.csv for: {best_run_name}")

# === Confusion Matrix
try:
    preds = [int(p.cls[0]) for p in val_metrics.pred]
    labels = [int(l.cls[0]) for l in val_metrics.labels]

    print("\nClassification Report:")
    print(classification_report(labels, preds))

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(PLOTS_DIR, f"{best_run_name}_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f" Could not generate classification report or confusion matrix: {e}")
