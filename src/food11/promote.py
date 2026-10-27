"""Quality gate: move the champion alias only if the candidate is clearly better.

Usage:
    uv run python src/food11/promote.py --version 8
"""

import argparse
import math
import os

from mlflow import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "food11"
METRIC = "test_accuracy"
# size of the test split (evaluation folder of food11_processed_mini)
N_TEST = 1096


def main() -> None:
    p = argparse.ArgumentParser(description="Promote a model version if it passes the gate")
    p.add_argument("--version", required=True, help="candidate model version number")
    p.add_argument("--n-test", type=int, default=N_TEST, help="size of the test split")
    args = p.parse_args()

    client = MlflowClient(tracking_uri=TRACKING_URI)

    candidate = client.get_model_version(MODEL_NAME, args.version)
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")

    acc_candidate = client.get_run(candidate.run_id).data.metrics[METRIC]
    acc_champion = client.get_run(champion.run_id).data.metrics[METRIC]

    se = math.sqrt(acc_champion * (1 - acc_champion) / args.n_test)
    threshold = acc_champion + 2 * se

    if acc_candidate >= threshold:
        client.set_registered_model_alias(MODEL_NAME, "champion", candidate.version)
        decision = "promoted"
    else:
        client.set_registered_model_alias(MODEL_NAME, "challenger", candidate.version)
        decision = "rejected"

    tags = {
        "gate_decision": decision,
        "candidate_test_accuracy": f"{acc_candidate:.4f}",
        "champion_test_accuracy": f"{acc_champion:.4f}",
        "champion_version": champion.version,
        "gate_threshold": f"{threshold:.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, candidate.version, key, value)

    print(f"candidate v{candidate.version}: {acc_candidate:.4f}")
    print(f"champion  v{champion.version}: {acc_champion:.4f}  (se {se:.4f}, 2*se {2 * se:.4f})")
    print(f"threshold {threshold:.4f} -> {decision}")


if __name__ == "__main__":
    main()
