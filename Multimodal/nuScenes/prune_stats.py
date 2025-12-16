import json
import os
import pandas as pd

def analyze_prune_results(json_path="prune_comparison_results.json", output_csv="prune_analysis.csv"):
    """
    Loads prune_comparison_results.json, computes per-image ratios, and saves to CSV.
    Ratios:
      - lost_ratio = missed_baseline / baseline_boxes
      - match_ratio = matched_baseline / baseline_boxes
      - new_ratio   = new_pruned / pruned_boxes
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    results = []
    for entry in data:
        image_index = entry["image_index"]
        b_base   = entry["baseline_boxes"]
        b_pruned = entry["pruned_boxes"]
        m_base   = entry["matched_baseline"]
        miss_base= entry["missed_baseline"]
        m_pruned = entry["matched_pruned"]
        new_prun = entry["new_pruned"]

        # Compute ratios carefully, checking for zero denominators
        lost_ratio = None
        match_ratio = None
        new_ratio = None

        if b_base > 0:
            lost_ratio = round(miss_base / b_base, 3)
            match_ratio = round(m_base / b_base, 3)
        if b_pruned > 0:
            new_ratio = round(new_prun / b_pruned, 3)

        results.append({
            "image_index": image_index,
            "baseline_boxes": b_base,
            "pruned_boxes": b_pruned,
            "matched_baseline": m_base,
            "missed_baseline": miss_base,
            "matched_pruned": m_pruned,
            "new_pruned": new_prun,
            "lost_ratio": lost_ratio,
            "match_ratio": match_ratio,
            "new_ratio": new_ratio
        })

    df = pd.DataFrame(results)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved per-image prune analysis to {output_csv}")

    # Basic stats
    print("\n=== Summary Stats (Ratios) ===")
    # Only compute stats if those columns are not all NaN
    if df["lost_ratio"].notna().any():
        print("Lost Ratio:", df["lost_ratio"].describe())
    if df["match_ratio"].notna().any():
        print("Match Ratio:", df["match_ratio"].describe())
    if df["new_ratio"].notna().any():
        print("New Ratio:", df["new_ratio"].describe())

if __name__ == "__main__":
    analyze_prune_results(
        json_path="prune_comparison_results.json",       # Path to your file
        output_csv="prune_analysis.csv"                  # Output CSV
    )
