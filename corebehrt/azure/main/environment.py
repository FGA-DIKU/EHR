from corebehrt.azure import environment


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("build_env", help="Build the CoreBEHRT environment")
    parser.set_defaults(func=lambda _: environment.build())
