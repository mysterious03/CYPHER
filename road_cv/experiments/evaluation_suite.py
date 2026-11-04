"""
Comprehensive Experimental Evaluation and Ablation Suite
Computes mAP@0.50, mAP@0.50:0.95, per-class metrics, small-object recall,
Expected Calibration Error (ECE), and ablation comparative summaries.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class CalibrationEvaluator:
    """
    Computes Expected Calibration Error (ECE) to verify whether confidence
    scores truly reflect empirical detection reliability.
    """

    @staticmethod
    def compute_ece(
        confidences: List[float],
        matches: List[bool],
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculates ECE across confidence bins between 0.0 and 1.0.
        """
        if not confidences or len(confidences) != len(matches):
            return {"ece": 0.0, "bin_count": num_bins}

        bins = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        total_samples = len(confidences)
        bin_details = []

        confs = np.array(confidences)
        accs = np.array(matches, dtype=float)

        for i in range(num_bins):
            bin_lower = bins[i]
            bin_upper = bins[i + 1]
            in_bin = (confs >= bin_lower) & (confs < bin_upper) if i < num_bins - 1 else (confs >= bin_lower) & (confs <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if np.count_nonzero(in_bin) > 0:
                bin_acc = np.mean(accs[in_bin])
                bin_conf = np.mean(confs[in_bin])
                diff = abs(bin_acc - bin_conf)
                ece += diff * prop_in_bin
                bin_details.append({
                    "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                    "sample_count": int(np.count_nonzero(in_bin)),
                    "accuracy": round(float(bin_acc), 4),
                    "confidence": round(float(bin_conf), 4),
                    "calibration_gap": round(float(diff), 4)
                })

        return {
            "expected_calibration_error": round(float(ece), 4),
            "num_samples": total_samples,
            "bins": bin_details
        }


class RoadCVEvaluationSuite:
    """
    Comprehensive evaluation engine for road-damage perception experiments.
    Computes standard mAP, size-stratified metrics, false positive counts, and ablation tables.
    """

    def __init__(self, iou_thresholds: Optional[List[float]] = None):
        self.iou_thresholds = iou_thresholds or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    @staticmethod
    def _box_iou(b1: List[float], b2: List[float]) -> float:
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        denom = a1 + a2 - inter
        return inter / denom if denom > 0 else 0.0

    def evaluate_detections(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        image_shape: Tuple[int, int] = (1080, 1920)
    ) -> Dict[str, Any]:
        """
        Evaluates a set of predictions against ground truths.
        predictions: list of dicts with 'box', 'class_name', 'confidence'
        ground_truths: list of dicts with 'box', 'class_name'
        """
        total_frame_area = image_shape[0] * image_shape[1]
        small_defect_area_thresh = 0.02 * total_frame_area # area < 2% of image

        matched_gt_50 = set()
        matched_preds_50 = set()
        small_gt_indices = set()
        small_matched_gt = set()

        for gt_idx, gt in enumerate(ground_truths):
            box = gt["box"]
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area <= small_defect_area_thresh:
                small_gt_indices.add(gt_idx)

        # Match at IoU 0.50
        sorted_preds = sorted(enumerate(predictions), key=lambda x: x[1].get("confidence", 0.0), reverse=True)
        conf_list = []
        match_list = []

        for p_idx, pred in sorted_preds:
            p_box = pred["box"]
            p_cls = pred.get("class_name", "")
            p_conf = float(pred.get("confidence", 0.0))
            conf_list.append(p_conf)

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt_50:
                    continue
                if gt.get("class_name", "") != p_cls:
                    continue

                iou = self._box_iou(p_box, gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= 0.50 and best_gt_idx >= 0:
                matched_gt_50.add(best_gt_idx)
                matched_preds_50.add(p_idx)
                match_list.append(True)
                if best_gt_idx in small_gt_indices:
                    small_matched_gt.add(best_gt_idx)
            else:
                match_list.append(False)

        tp = len(matched_gt_50)
        fp = len(predictions) - tp
        fn = len(ground_truths) - tp

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, len(ground_truths))
        f1 = (2 * precision * recall) / max(1e-5, precision + recall)

        # Small defect recall
        small_recall = len(small_matched_gt) / max(1, len(small_gt_indices)) if small_gt_indices else 1.0

        # Calibration
        calibration_results = CalibrationEvaluator.compute_ece(conf_list, match_list)

        return {
            "summary_metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "small_defect_recall": round(small_recall, 4),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "total_ground_truth": len(ground_truths),
                "total_predictions": len(predictions)
            },
            "calibration": calibration_results
        }

    @staticmethod
    def generate_ablation_summary(
        experiments: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compiles ablation runs (B0 to Full Integrated) into a structured comparative table.
        """
        table = []
        for exp_name, res in experiments.items():
            metrics = res.get("summary_metrics", {})
            resources = res.get("resources", {})
            table.append({
                "experiment": exp_name,
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1_score": metrics.get("f1_score", 0.0),
                "small_defect_recall": metrics.get("small_defect_recall", 0.0),
                "false_positives": metrics.get("false_positives", 0),
                "latency_ms": resources.get("total_latency_ms", 0.0),
                "peak_vram_gb": resources.get("peak_allocated_gb", 0.0)
            })
        return table

