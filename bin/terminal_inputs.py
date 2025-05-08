import argparse


def terminal_inputs():
    """Parse the terminal inputs and return the arguments"""

    parser = argparse.ArgumentParser(
        prog="camera_relsense",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log-level",
        type=int,
        default=30,
        help="Log level 10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL 0=NOTSET",
    )

    parser.add_argument(
        "--connect",
        action="append",
        type=str,
        help="Endpoints to connect to, in case of struggeling to find router. ex. tcp/localhost:7447",
    )

    parser.add_argument(
        "-r",
        "--realm",
        default="rise",
        type=str,
        help="Unique id for a realm/domain to connect ex. rise",
    )

    parser.add_argument(
        "-e",
        "--entity-id",
        type=str,
        required=True,
        help="Entity being a unique id representing an entity within the realm ex, landkrabban",
    )

    parser.add_argument(
        "-s",
        "--source-id",
        type=str,
        required=True,
        help="Lidar source id ex. camera/0",
    )

    parser.add_argument(
        "-f", "--frame-id", type=str, default=None, help="Frame id for foxglow"
    )

    #####################################

    parser.add_argument(
        "--publish",
        action="append",
        type=str,
        choices=["point_cloud","raw_color", "raw_depth"],
        help="publish message type"
    )

  
  
    ## Parse arguments and start doing our thing
    args = parser.parse_args()

    return args
