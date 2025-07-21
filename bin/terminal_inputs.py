import argparse
from main import run_thermal, run_stereo

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
        "--mode",
        "-m",
        dest="mode",
        choices=["peer", "client"],
        type=str,
        help="The zenoh session mode.",
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

    # Subcommands
    subparsers = parser.add_subparsers(required=True)

    # Thermal camera options
    parser_thermal = subparsers.add_parser("run_thermal", help="Thermal camera options")
    parser_thermal.set_defaults(func=run_thermal)

    # FPS
    parser_thermal.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the camera",
    )

    # Enable IMU
    parser_thermal.add_argument(
        "--enable-imu",
        action="store_true",
        help="Enable IMU, this consumes a lot of CPU from the camera",
    )

    # Stereo camera options
    parser_stereo = subparsers.add_parser("run_stereo", help="Stereo camera options")
    parser_stereo.set_defaults(func=run_stereo)

    # Max range
    parser_stereo.add_argument(
        "--max-range-meters",
        type=int,
        default=50,
        help="Max range in meters for the stereo camera",
    )

    # confidence_threshold
    parser_stereo.add_argument(
        "--confidence-threshold",
        type=int,
        default=155,
        help="Confidence threshold for point cloud generation, adjust as needed, 255 accepts all, 0 execludes all, does introduce more noise?",
    )
    
    # fps_stereo 
    parser_stereo.add_argument(
        "--fps-stereo",
        type=int,
        default=12,
        help="Frames per second for the stereo camera",
    )

    # fps_rgb
    parser_stereo.add_argument(
        "--fps-rgb",
        type=int,
        default=12,
        help="RGB camera runs at 2*fps, so if fps is 5, RGB camera runs at 10 FPS",
    )    

    # stereo_resolution
    parser_stereo.add_argument(
        "--stereo-resolution",
        type=str,
        default="400p",
        choices=["400p", "720p", "800p" ],
        help="options: '800p', '720p', '400p'; camera resolution for the rgb image is no affected by this setting",
    )

    # Enable IMU
    parser_stereo.add_argument(
        "--enable-imu",
        action="store_true",
        help="Enable IMU, this consumes a lot of CPU from the LR camera",
    )

  
    ## Parse arguments and start doing our thing
    args = parser.parse_args()

    return args
