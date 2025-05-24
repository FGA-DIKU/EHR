import pandas as pd
from corebehrt.functional.setup.args import get_args
from corebehrt.modules.setup.config import load_config

CONFIG_PATH = "./corebehrt/configs/performance_tests/test_performance.yaml"


def main_evaluate_performance(config_path):
    # Setup directories
    cfg = load_config(config_path)
    bad_on_bad_metrics_path = cfg.paths.bad_metrics_path
    good_metrics_path = cfg.paths.good_metrics_path

    bad_metrics = pd.read_csv(bad_on_bad_metrics_path)
    good_metrics = pd.read_csv(good_metrics_path)

    good_rocs = good_metrics["roc_auc"].tolist()
    bad_on_bad_rocs = bad_metrics["roc_auc"].tolist()

    acceptable_good_rocs = all(roc >= 0.7 for roc in good_rocs) and all(
        roc <= 0.9 for roc in good_rocs
    )
    acceptable_bad_rocs = all(roc >= 0.99 for roc in bad_on_bad_rocs)

    print(good_rocs)
    print(bad_on_bad_rocs)

    if not acceptable_good_rocs:
        raise ValueError(
            "Performance metrics are not acceptable for model with proper censoring"
        )
    if not acceptable_bad_rocs:
        raise ValueError(
            "Performance metrics are not acceptable for model with no censoring"
        )

    print("Performance metrics are acceptable")


if __name__ == "__main__":
    args = get_args(CONFIG_PATH)
    main_evaluate_performance(args.config_path)
