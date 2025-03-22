from corebehrt.azure import util
from corebehrt.azure.util.config import load_config


def add_parser(subparsers) -> None:
    """
    Add the test subparser
    """
    parser = subparsers.add_parser("test", help="Run a test job.")
    parser.add_argument("NAME", choices={"small", "full"})
    parser.set_defaults(func=create_and_run_test)


def create_and_run_test(args) -> None:
    """
    Run a pipeline test from the given args.
    """
    name = args.NAME

    test_cfg_file = f"corebehrt/azure/configs/{name}/test.yaml"
    test_cfg = load_config(test_cfg_file)

    pl = util.pipeline.create(
        "E2E",
        test_cfg["data"],
        test_cfg.get("computes", {}),
        config_dir=f"corebehrt/azure/configs/{name}",
        register_output={},
        log_system_metrics=True,
        test_cfg_file=test_cfg_file,
    )

    util.pipeline.run(pl, "corebehrt_pipeline_tests")
