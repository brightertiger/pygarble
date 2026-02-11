#!/usr/bin/env python
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from pygarble import GarbleDetector, Strategy, EnsembleDetector


STRATEGIES = [
    # New strategies (v0.3.0)
    Strategy.MARKOV_CHAIN,
    Strategy.NGRAM_FREQUENCY,
    Strategy.WORD_LOOKUP,
    Strategy.SYMBOL_RATIO,
    Strategy.REPETITION,
    Strategy.HEX_STRING,
    # New strategies (v0.4.0)
    Strategy.COMPRESSION_RATIO,
    Strategy.MOJIBAKE,
    Strategy.PRONOUNCEABILITY,
    Strategy.UNICODE_SCRIPT,
    # New high-precision strategies (v0.5.0)
    Strategy.BIGRAM_PROBABILITY,
    Strategy.LETTER_POSITION,
    Strategy.CONSONANT_SEQUENCE,
    Strategy.VOWEL_PATTERN,
    Strategy.LETTER_FREQUENCY,
    Strategy.RARE_TRIGRAM,
    # New word-level strategies (v0.6.0)
    Strategy.FUNCTION_WORD_DENSITY,
    Strategy.AFFIX_DETECTION,
    Strategy.ZIPF_CONFORMITY,
    Strategy.WORD_COLLOCATION,
    # Existing strategies
    Strategy.CHARACTER_FREQUENCY,
    Strategy.WORD_LENGTH,
    Strategy.PATTERN_MATCHING,
    Strategy.STATISTICAL_ANALYSIS,
    Strategy.ENTROPY_BASED,
    Strategy.VOWEL_RATIO,
    Strategy.KEYBOARD_PATTERN,
]

# Strategies that require optional dependencies (excluded by default)
OPTIONAL_STRATEGIES = [
    Strategy.ENGLISH_WORD_VALIDATION,  # Requires pyspellchecker
]


def load_test_cases(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_cases = []
    for category_data in data["test_cases"]:
        category = category_data["category"]
        source = category_data.get("source", "internal")
        for case in category_data["cases"]:
            all_cases.append({
                "category": category,
                "source": source,
                "text": case["text"],
                "expected": case["expected_garbled"]
            })
    return all_cases


def run_benchmark(test_cases: List[Dict[str, Any]], threshold: float = 0.5, include_optional: bool = False) -> Dict[str, Any]:
    results = {}

    strategies_to_run = STRATEGIES.copy()
    if include_optional:
        strategies_to_run.extend(OPTIONAL_STRATEGIES)

    for strategy in strategies_to_run:
        strategy_name = strategy.value
        detector = GarbleDetector(strategy, threshold=threshold)
        
        predictions = []
        start_time = time.perf_counter()
        
        for case in test_cases:
            pred = detector.predict(case["text"])
            predictions.append({
                "text": case["text"][:50] + "..." if len(case["text"]) > 50 else case["text"],
                "category": case["category"],
                "source": case["source"],
                "expected": case["expected"],
                "predicted": pred,
                "correct": pred == case["expected"]
            })
        
        elapsed_time = time.perf_counter() - start_time
        
        correct = sum(1 for p in predictions if p["correct"])
        total = len(predictions)
        
        true_positives = sum(1 for p in predictions if p["expected"] and p["predicted"])
        false_positives = sum(1 for p in predictions if not p["expected"] and p["predicted"])
        true_negatives = sum(1 for p in predictions if not p["expected"] and not p["predicted"])
        false_negatives = sum(1 for p in predictions if p["expected"] and not p["predicted"])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[strategy_name] = {
            "accuracy": correct / total,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "total_cases": total,
            "time_seconds": elapsed_time,
            "predictions": predictions
        }
    
    ensemble = EnsembleDetector(threshold=threshold)
    predictions = []
    start_time = time.perf_counter()
    
    for case in test_cases:
        pred = ensemble.predict(case["text"])
        predictions.append({
            "text": case["text"][:50] + "..." if len(case["text"]) > 50 else case["text"],
            "category": case["category"],
            "source": case["source"],
            "expected": case["expected"],
            "predicted": pred,
            "correct": pred == case["expected"]
        })
    
    elapsed_time = time.perf_counter() - start_time
    correct = sum(1 for p in predictions if p["correct"])
    total = len(predictions)
    
    true_positives = sum(1 for p in predictions if p["expected"] and p["predicted"])
    false_positives = sum(1 for p in predictions if not p["expected"] and p["predicted"])
    true_negatives = sum(1 for p in predictions if not p["expected"] and not p["predicted"])
    false_negatives = sum(1 for p in predictions if p["expected"] and not p["predicted"])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results["ensemble"] = {
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "total_cases": total,
        "time_seconds": elapsed_time,
        "predictions": predictions
    }
    
    return results


def analyze_by_category(results: Dict[str, Any], test_cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    categories = set(case["category"] for case in test_cases)
    category_analysis = {}

    for strategy_name, strategy_results in results.items():
        category_analysis[strategy_name] = {}
        for category in categories:
            category_preds = [p for p in strategy_results["predictions"] if p["category"] == category]
            if category_preds:
                accuracy = sum(1 for p in category_preds if p["correct"]) / len(category_preds)
                category_analysis[strategy_name][category] = accuracy

    return category_analysis


def analyze_by_source(results: Dict[str, Any], test_cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Analyze results by source (internal vs external)."""
    sources = set(case["source"] for case in test_cases)
    source_analysis = {}

    for strategy_name, strategy_results in results.items():
        source_analysis[strategy_name] = {}
        for source in sources:
            source_preds = [p for p in strategy_results["predictions"] if p.get("source") == source]
            if source_preds:
                correct = sum(1 for p in source_preds if p["correct"])
                total = len(source_preds)

                tp = sum(1 for p in source_preds if p["expected"] and p["predicted"])
                fp = sum(1 for p in source_preds if not p["expected"] and p["predicted"])
                fn = sum(1 for p in source_preds if p["expected"] and not p["predicted"])

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                source_analysis[strategy_name][source] = {
                    "accuracy": correct / total,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "total": total,
                    "fp": fp,
                    "fn": fn,
                }

    return source_analysis


def format_results(
    results: Dict[str, Any],
    category_analysis: Dict[str, Dict[str, float]],
    source_analysis: Dict[str, Dict[str, Dict[str, float]]] = None,
    test_cases: List[Dict[str, Any]] = None,
) -> str:
    output = []
    output.append("=" * 80)
    output.append("PYGARBLE BENCHMARK RESULTS")
    output.append(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append("=" * 80)

    # Dataset summary
    if test_cases:
        internal_count = sum(1 for c in test_cases if c.get("source") == "internal")
        external_count = len(test_cases) - internal_count
        output.append(f"\nDataset: {len(test_cases)} total cases ({internal_count} internal, {external_count} external)")

    output.append("\n### OVERALL METRICS ###\n")
    output.append(f"{'Strategy':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} {'Time (s)':>10}")
    output.append("-" * 80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["f1_score"], reverse=True)

    for strategy_name, metrics in sorted_results:
        output.append(f"{strategy_name:<25} {metrics['accuracy']:>10.2%} {metrics['precision']:>10.2%} "
              f"{metrics['recall']:>10.2%} {metrics['f1_score']:>10.2%} {metrics['time_seconds']:>10.4f}")

    output.append("\n### CONFUSION MATRIX SUMMARY ###\n")
    output.append(f"{'Strategy':<25} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}")
    output.append("-" * 55)

    for strategy_name, metrics in sorted_results:
        output.append(f"{strategy_name:<25} {metrics['true_positives']:>6} {metrics['false_positives']:>6} "
              f"{metrics['true_negatives']:>6} {metrics['false_negatives']:>6}")

    # Source-based analysis (Internal vs External)
    if source_analysis:
        output.append("\n### METRICS BY DATA SOURCE ###\n")
        output.append("Comparing performance on internal (developed alongside strategies) vs external (unbiased) data:\n")

        sources = sorted(set(src for sa in source_analysis.values() for src in sa))

        for source in sources:
            source_label = source.replace("_", " ").title()
            output.append(f"\n--- {source_label} ---")
            output.append(f"{'Strategy':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FP':>6} {'FN':>6}")
            output.append("-" * 85)

            for strategy_name, _ in sorted_results:
                if source in source_analysis.get(strategy_name, {}):
                    m = source_analysis[strategy_name][source]
                    output.append(
                        f"{strategy_name:<25} {m['accuracy']:>10.2%} {m['precision']:>10.2%} "
                        f"{m['recall']:>10.2%} {m['f1']:>10.2%} {m['fp']:>6} {m['fn']:>6}"
                    )

        # Overfitting detection
        output.append("\n### OVERFITTING ANALYSIS ###\n")
        output.append("Comparing internal vs external performance (large gaps suggest overfitting):\n")
        output.append(f"{'Strategy':<25} {'Internal F1':>12} {'External F1':>12} {'Gap':>10}")
        output.append("-" * 65)

        for strategy_name, _ in sorted_results[:10]:  # Top 10 strategies
            internal_f1 = source_analysis.get(strategy_name, {}).get("internal", {}).get("f1", 0)

            # Calculate average external F1
            external_sources = [s for s in sources if s != "internal"]
            external_f1s = [
                source_analysis.get(strategy_name, {}).get(s, {}).get("f1", 0)
                for s in external_sources
            ]
            external_f1 = sum(external_f1s) / len(external_f1s) if external_f1s else 0

            gap = internal_f1 - external_f1
            gap_indicator = "⚠" if gap > 0.15 else "✓" if gap < 0.05 else ""
            output.append(f"{strategy_name:<25} {internal_f1:>12.2%} {external_f1:>12.2%} {gap:>+10.2%} {gap_indicator}")

    output.append("\n### ACCURACY BY CATEGORY ###\n")

    categories = sorted(set(cat for cats in category_analysis.values() for cat in cats))

    header = f"{'Strategy':<25}" + "".join(f"{cat[:12]:>14}" for cat in categories)
    output.append(header)
    output.append("-" * len(header))

    for strategy_name, _ in sorted_results:
        row = f"{strategy_name:<25}"
        for cat in categories:
            acc = category_analysis[strategy_name].get(cat, 0)
            row += f"{acc:>13.0%} "
        output.append(row)

    output.append("\n### MISCLASSIFIED EXAMPLES (Top 5 per strategy) ###\n")

    for strategy_name, metrics in sorted_results[:3]:
        misclassified = [p for p in metrics["predictions"] if not p["correct"]][:5]
        if misclassified:
            output.append(f"\n{strategy_name}:")
            for p in misclassified:
                expected = "garbled" if p["expected"] else "normal"
                predicted = "garbled" if p["predicted"] else "normal"
                source_tag = f"[{p.get('source', 'unknown')}]" if p.get('source') != 'internal' else ""
                output.append(f"  [{p['category']}]{source_tag} \"{p['text']}\"")
                output.append(f"    Expected: {expected}, Predicted: {predicted}")

    return "\n".join(output)


def main():
    script_dir = Path(__file__).parent
    json_path = script_dir / "benchmark_data.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return

    print("Loading test cases...")
    test_cases = load_test_cases(str(json_path))

    # Count by source
    source_counts = {}
    for case in test_cases:
        src = case.get("source", "internal")
        source_counts[src] = source_counts.get(src, 0) + 1

    print(f"Loaded {len(test_cases)} test cases:")
    for src, count in sorted(source_counts.items()):
        print(f"  - {src}: {count}")

    print("\nRunning benchmark...")
    results = run_benchmark(test_cases, threshold=0.5)

    print("Analyzing by category...")
    category_analysis = analyze_by_category(results, test_cases)

    print("Analyzing by source...")
    source_analysis = analyze_by_source(results, test_cases)

    formatted_output = format_results(results, category_analysis, source_analysis, test_cases)
    print(formatted_output)

    output_json_path = script_dir / "benchmark_results.json"
    with open(output_json_path, "w") as f:
        output_data = {
            "run_date": datetime.now().isoformat(),
            "threshold": 0.5,
            "total_test_cases": len(test_cases),
            "source_counts": source_counts,
            "strategies": {
                strategy: {k: v for k, v in metrics.items() if k != "predictions"}
                for strategy, metrics in results.items()
            },
            "category_analysis": category_analysis,
            "source_analysis": source_analysis,
        }
        json.dump(output_data, f, indent=2)
    print(f"\nJSON results saved to: {output_json_path}")

    output_txt_path = script_dir / "benchmark_results.txt"
    with open(output_txt_path, "w") as f:
        f.write(formatted_output)
    print(f"Text results saved to: {output_txt_path}")


if __name__ == "__main__":
    main()

