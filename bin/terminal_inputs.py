import argparse


def terminal_inputs():
    """Parse the terminal inputs and return the arguments"""

    parser = argparse.ArgumentParser(
        prog="camera_luxonis",
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

    ## IP address of the camera
    parser.add_argument(
        "--ip-address",
        type=str,
        default="192.168.3.12",
        help="IP address of the camera",
    )

    parser.add_argument(
        "--camera-thermal",
        action="store_true",
        help="Set to true if using a thermal camera",
    )

    parser.add_argument(
        "--camera-stereo",
        action="store_true",
        help="Set to true if using a thermal camera",
    )

    # FPS
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the camera",
    )
  

  
  
    ## Parse arguments and start doing our thing
    args = parser.parse_args()

    return args
