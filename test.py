import os
import numpy as np
import os
import numpy as np

def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Box format: [x_center, y_center, width, height] (normalized).
    """
    # Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Calculate intersection
    x1_inter = max(box1_x1, box2_x1)
    y1_inter = max(box1_y1, box2_y1)
    x2_inter = min(box1_x2, box2_x2)
    y2_inter = min(box1_y2, box2_y2)
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # Calculate union
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    # IoU is intersection area divided by union area
    return inter_area / union_area

def load_labels(label_file):
    """
    Load labels from a YOLO format label file.
    Returns a list of [class_id, x_center, y_center, width, height, confidence].
    If confidence is not present, defaults to 1.0.
    """
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            # Check if confidence score is provided, otherwise set to 1.0
            if len(values) == 5:  # [class_id, x_center, y_center, width, height]
                values.append(1.0)  # Default confidence
            labels.append(values)  # [class_id, x_center, y_center, width, height, confidence]
    return labels

def compare_predictions(validation_folder, predicted_folder, iou_threshold=0.4, confidence_threshold=0.3):
    """
    Compare predictions against ground truth and calculate TP, FP, FN, TN, and IoU.
    Includes confidence score filtering.
    """
    tp, fp, fn, tn = 0, 0, 0, 0
    total_iou = 0
    total_boxes = 0

    # Get all label files
    validation_files = sorted(os.listdir(validation_folder))
    predicted_files = sorted(os.listdir(predicted_folder))

    for val_file, pred_file in zip(validation_files, predicted_files):
        # Load the ground truth and predicted labels
        val_labels = load_labels(os.path.join(validation_folder, val_file))
        pred_labels = load_labels(os.path.join(predicted_folder, pred_file))

        matched = set()  # Keep track of matched predictions

        # Compare each prediction to every ground truth box
        for pred_box in pred_labels:
            pred_confidence = pred_box[5]  # Get confidence score

            # Filter out low-confidence predictions
            if pred_confidence < confidence_threshold:
                continue  # Skip low-confidence predictions

            best_iou = 0
            best_gt_idx = -1

            for i, gt_box in enumerate(val_labels):
                current_iou = iou(pred_box[1:5], gt_box[1:5])  # IoU with x,y,w,h
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = i

            # Evaluate based on IoU threshold
            if best_iou >= iou_threshold:
                tp += 1  # True positive (correct detection)
                total_iou += best_iou
                total_boxes += 1
                matched.add(best_gt_idx)
            elif 0.3 <= best_iou < iou_threshold:
                # For IoU in range 0.3–0.5, treat it with more caution
                fp += 1
            else:
                fp += 1  # False positive (wrong detection)

        # Any ground truth boxes not matched with a prediction are considered False Negatives
        fn += len(val_labels) - len(matched)

        # TN: Any space without a prediction and without ground truth is a true negative
        tn += max(0, len(val_labels) - len(pred_labels))

    # Calculate overall metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + fp + fn) > 0 else 0
    avg_iou = total_iou / total_boxes if total_boxes > 0 else 0

    return tp, fp, fn, tn, accuracy, precision, recall, f1_score, avg_iou


def load_labels(label_file):
    """
    Load labels from a YOLO format label file.
    Returns a list of [class_id, x_center, y_center, width, height, confidence].
    If confidence is not present, defaults to 1.0.
    """
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            # Check if confidence score is provided, otherwise set to 1.0
            if len(values) == 5:  # [class_id, x_center, y_center, width, height]
                values.append(1.0)  # Default confidence
            labels.append(values)  # [class_id, x_center, y_center, width, height, confidence]
    return labels


def normalize_metrics(tp, fp, fn, tn):
    """
    Normalize TP, FP, FN, TN values based on total ground truth boxes.
    """
    total_samples = tp + fp + fn + tn  # Total samples including TN
    total_ground_truth = tp + fn  # Total ground truth boxes

    normalized_TP = tp / total_ground_truth if total_ground_truth != 0 else 0
    normalized_FP = fp / total_ground_truth if total_ground_truth != 0 else 0
    normalized_FN = fn / total_ground_truth if total_ground_truth != 0 else 0
    normalized_TN = tn / total_samples if total_samples != 0 else 0  # Normalize TN

    return normalized_TP, normalized_FP, normalized_FN, normalized_TN

def save_metrics_to_file(tp, fp, fn, tn, accuracy, precision, recall, f1_score, avg_iou, output_file):
    """
    Save the confusion matrix metrics and performance metrics to a .txt file.
    """
    # Normalize metrics
    normalized_TP, normalized_FP, normalized_FN, normalized_TN = normalize_metrics(tp, fp, fn, tn)

    # Write metrics to a file
    with open(output_file, 'w') as f:
        f.write(f"True Positives (TP): {tp}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
        f.write(f"True Negatives (TN): {tn}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1_score:.4f}\n")
        f.write(f"Average IoU: {avg_iou:.4f}\n\n")

        # Normalized confusion matrix values
        f.write("Confusion Matrix Metrics (Normalized):\n")
        f.write(f"Normalized TP: {normalized_TP:.4f}\n")
        f.write(f"Normalized FP: {normalized_FP:.4f}\n")
        f.write(f"Normalized FN: {normalized_FN:.4f}\n")
        f.write(f"Normalized TN: {normalized_TN:.4f}\n\n")  # Add Normalized TN

def main():
    # Paths to your folders
    validation_folder = 'C:/Users/Neha KB/Desktop/friday/test/labels'
    predicted_folder = 'C:/Users/Neha KB/Desktop/friday/predicted/labels'
    output_metrics_file = 'C:/Users/Neha KB/Desktop/friday/test/metrics.txt'

    # Compare the predictions and get metrics with confidence filtering
    tp, fp, fn, tn, accuracy, precision, recall, f1_score, avg_iou = compare_predictions(
        validation_folder, predicted_folder, iou_threshold=0.5, confidence_threshold=0.5
    )

    # Save the metrics to a text file
    save_metrics_to_file(tp, fp, fn, tn, accuracy, precision, recall, f1_score, avg_iou, output_metrics_file)

    print(f"Metrics saved to {output_metrics_file}")

if __name__ == "__main__":
    main()
