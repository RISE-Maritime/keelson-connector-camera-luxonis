#!/usr/bin/env python3

import cv2
import depthai as dai
import time
import numpy as np
import sys

# MiS - Martin Sanfridson, RISE, March 2025
import logging
import zenoh
import warnings
import json
import numpy as np

import cv2

import terminal_inputs
import pipelines
import keelson
from keelson.payloads.foxglove.RawImage_pb2 import RawImage
from keelson.payloads.Primitives_pb2 import TimestampedBytes, TimestampedDuration
from keelson.payloads.Decomposed3DVector_pb2 import Decomposed3DVector


def main():
    # Parse terminal inputs
    args = terminal_inputs.terminal_inputs()

    # Setup logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(lineno)d]: %(message)s",
        level=args.log_level,
    )
    logging.captureWarnings(True)
    warnings.filterwarnings("once")

    ## Construct session
    logging.info("Opening Zenoh session...")
    conf = zenoh.Config()
    if args.mode is not None:
        conf.insert_json5("mode", json.dumps(args.mode))
    if args.connect is not None:
        conf.insert_json5("connect/endpoints", json.dumps(args.connect))

    with zenoh.open(conf) as session:
        # Dispatch to correct function
        try:
            args.func(session, args)
        except KeyboardInterrupt:
            logging.info("Closing down on user request!")
            sys.exit(0)


def run_thermal(session, args):
    logging.info("Running thermal camera...")

    # IMU Acceleration
    key_imu_acc = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="linear_acceleration_mpss",
        source_id=args.source_id + "/imu",
    )
    publisher_imu_acc = session.declare_publisher(key_imu_acc)
    logging.info(f"Publisher for image at {key_imu_acc}")

    # IMU Angular velocity
    key_imu_vel = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="angular_velocity_radps",
        source_id=args.source_id + "/imu",
    )
    publisher_imu_vel = session.declare_publisher(key_imu_vel)
    logging.info(f"Publisher for image at {key_imu_vel}")

    # Thermal image (Raw bytes)
    key_image_thermal_raw = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="raw",
        source_id=args.source_id + "/temperatures/np/celsius",
    )
    publisher_image_thermal_raw = session.declare_publisher(
        key_image_thermal_raw
    )
    logging.info(f"Publisher for image at {key_image_thermal_raw}")

    # Thermal image (Raw bytes, but with timestamp, and normalized)
    key_image_thermal = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="image_raw",
        source_id=args.source_id + "/normalized",
    )
    publisher_image_thermal = session.declare_publisher(
        key_image_thermal
    )
    logging.info(f"Publisher for image at {key_image_thermal}")

    # Image RAW CMOS
    key_image_raw_cmos = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="image_raw",
        source_id=args.source_id + "/cmos",
    )
    publisher_image_raw_cmos = session.declare_publisher(
        key_image_raw_cmos,
        # congestion_control=zenoh.CongestionControl.BLOCK
    )
    logging.info(f"Publisher for image at {key_image_raw_cmos}")

    # IMU  Timestamp sens start

    key_imu_acc_timestamp_start = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="device_uptime_duration",
        source_id=args.source_id + "/imu/linear_acceleration",
    )
    publisher_imu_acc_timestamp_start = session.declare_publisher(
        key_imu_acc_timestamp_start
    )
    logging.info(f"Publisher for IMU timestamp at {key_imu_acc_timestamp_start}")

    key_imu_vel_timestamp_start = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="device_uptime_duration",
        source_id=args.source_id + "/imu/angular_velocity",
    )
    publisher_imu_timestamp_start = session.declare_publisher(
        key_imu_vel_timestamp_start
    )
    logging.info(f"Publisher for IMU timestamp at {key_imu_vel_timestamp_start}")


    # Thermal Timestamp since start 
    key_timestamp_thermal_start = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="device_uptime_duration",
        source_id=args.source_id + "/thermal",
    )
    publisher_timestamp_thermal_start = session.declare_publisher(
        key_timestamp_thermal_start
    )   
    logging.info(f"Publisher for thermal timestamp at {key_timestamp_thermal_start}")

    # CMOS Timestamp since start
    key_timestamp_cmos_start = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="device_uptime_duration",
        source_id=args.source_id + "/cmos",
    )
    publisher_timestamp_cmos_start = session.declare_publisher(
        key_timestamp_cmos_start
    )
    logging.info(f"Publisher for CMOS timestamp at {key_timestamp_cmos_start}")

    logging.info("Pipeline setup THERMAL...")
    pipeline_OAK_T = pipelines.setup_pipeline_OAK_T(args.fps)

    logging.info("Device setup... this may take a while...")
    device_info = dai.DeviceInfo(args.ip_address)

    with dai.Device(deviceInfo=device_info, pipeline=pipeline_OAK_T) as device:

        # device.setLogLevel(dai.LogLevel.INFO)
        # device.setLogOutputLevel(dai.LogLevel.INFO)
        logging.info(
            f"MxID: {device.getMxId()}, name: {device.getDeviceName()}, platform: {device.getDeviceInfo()}"
        )

        videoQueue = device.getOutputQueue(name="video", maxSize=1, blocking=False)
        thermoQueue = device.getOutputQueue(name="thermo", maxSize=1, blocking=False)
        if args.enable_imu:
            imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)

        # Initialize FPS counters (output every 5 seconds)
        fpsCounter_rgb = FPSCounter(output_interval_seconds=5.0)
        fpsCounter_thermal = FPSCounter(output_interval_seconds=5.0)
        fpsCounter_imu_acc = FPSCounter(output_interval_seconds=5.0)
        fpsCounter_imu_vel = FPSCounter(output_interval_seconds=5.0)

        while True:
            videoIn = videoQueue.get()
            thermoIn = thermoQueue.get()
            if args.enable_imu:
                imuIn = imuQueue.get()  # TODO: could put imu queue in a separate thread

            ingress_timestamp = time.time_ns()

            if args.enable_imu:
                for imuPacket in imuIn.packets:
                    # Get individual timestamps for each sensor reading
                    acceleroValues = imuPacket.acceleroMeter
                    acceleroTd = acceleroValues.getTimestampDevice()
                    
                    # Create unique timestamp using device time + current system time offset
                    # This preserves the device's precise timing while anchoring to system time
                    current_time = time.time_ns()
                    acc_device_time_ns = int(acceleroTd.total_seconds() * 1_000_000_000)
                    acc_unique_timestamp = current_time - (current_time % 1_000_000) + (acc_device_time_ns % 1_000_000)

                    payload_acc_duration = TimestampedDuration()
                    payload_acc_duration.timestamp.FromNanoseconds(current_time)
                    payload_acc_duration.value.FromTimedelta(acceleroTd)
                    serialized_payload = payload_acc_duration.SerializeToString()
                    envelope = keelson.enclose(serialized_payload)
                    publisher_imu_acc_timestamp_start.put(envelope)
                    logging.debug(f"...published on {key_imu_acc_timestamp_start}")

                    payload_acc = Decomposed3DVector()
                    payload_acc.timestamp.FromNanoseconds(acc_unique_timestamp)
                    payload_acc.vector.x = acceleroValues.x
                    payload_acc.vector.y = acceleroValues.y
                    payload_acc.vector.z = acceleroValues.z
                    serialized_payload = payload_acc.SerializeToString()
                    envelope = keelson.enclose(serialized_payload)
                    publisher_imu_acc.put(envelope)
                    logging.debug(f"...published on {key_imu_acc}")
                    
                    # Update IMU acceleration FPS counter and log every 5 seconds
                    current_fps_acc = fpsCounter_imu_acc.tick()
                    if fpsCounter_imu_acc.frameCount == 0:  # This means we just reset, so we calculated new FPS
                        logging.info(f"IMU acceleration FPS: {current_fps_acc:.2f} Hz")

                    gyroValues = imuPacket.gyroscope
                    gyroTs = gyroValues.getTimestampDevice()
                    
                    # Create unique timestamp for gyroscope using the same method
                    gyro_device_time_ns = int(gyroTs.total_seconds() * 1_000_000_000)
                    gyro_unique_timestamp = current_time - (current_time % 1_000_000) + (gyro_device_time_ns % 1_000_000)
                    
                    payload_vel_duration = TimestampedDuration()
                    payload_vel_duration.timestamp.FromNanoseconds(current_time)
                    payload_vel_duration.value.FromTimedelta(gyroTs)
                    serialized_payload = payload_vel_duration.SerializeToString()
                    envelope = keelson.enclose(serialized_payload)
                    publisher_imu_timestamp_start.put(envelope)
                    logging.debug(f"...published on {key_imu_vel_timestamp_start}")
                    
                    payload_vel = Decomposed3DVector()
                    payload_vel.timestamp.FromNanoseconds(gyro_unique_timestamp)
                    payload_vel.vector.x = gyroValues.x
                    payload_vel.vector.y = gyroValues.y
                    payload_vel.vector.z = gyroValues.z
                    serialized_payload = payload_vel.SerializeToString()
                    envelope = keelson.enclose(serialized_payload)
                    publisher_imu_vel.put(envelope)
                    logging.debug(f"...published on {key_imu_vel}")
                    
                    # Update IMU angular velocity FPS counter and log every 5 seconds
                    current_fps_vel = fpsCounter_imu_vel.tick()
                    if fpsCounter_imu_vel.frameCount == 0:  # This means we just reset, so we calculated new FPS
                        logging.info(f"IMU angular velocity FPS: {current_fps_vel:.2f} Hz")


            # CMOS        
            time_rbg = videoQueue.get().getTimestamp()

            payload_cmos_duration = TimestampedDuration()
            payload_cmos_duration.timestamp.FromNanoseconds(ingress_timestamp)
            payload_cmos_duration.value.FromTimedelta(time_rbg)
            serialized_payload = payload_cmos_duration.SerializeToString()
            envelope = keelson.enclose(serialized_payload)
            publisher_timestamp_cmos_start.put(envelope)
            logging.debug(f"...published on {key_timestamp_cmos_start}")

            # Get BGR frame from NV12 encoded video frame to show with opencv
            cmos_fram_bytes = videoIn.getCvFrame()  # numpy array
            payload_raw = RawImage()
            payload_raw.timestamp.FromNanoseconds(ingress_timestamp)
            if args.frame_id is not None:
                payload_raw.frame_id = args.frame_id
            payload_raw.width = cmos_fram_bytes.shape[1]
            payload_raw.height = cmos_fram_bytes.shape[0]
            payload_raw.encoding = "bgr8"  # Default in OpenCV
            payload_raw.step = cmos_fram_bytes.strides[0]
            payload_raw.data = cmos_fram_bytes.tobytes()
            serialized_payload = payload_raw.SerializeToString()
            envelope = keelson.enclose(serialized_payload)
            publisher_image_raw_cmos.put(envelope)
            logging.debug(f"...published on {key_image_raw_cmos}")
          
            # # Encode the BGR frame to a JPEG file 
            # ret, jpeg_bytes = cv2.imencode('.jpg', cmos_fram_bytes)
            # if not ret:
            #     logging.error("Failed to encode frame as JPEG")
            #     return
            # cv2.imwrite("bgr.jpg",  cmos_fram_bytes)

            # Get frame from thermal camera, then normalize and colorize it


            # THERMAL 

            time_therm = thermoQueue.get().getTimestamp()  # total_seconds since start

            payload_thermal_duration = TimestampedDuration()
            payload_thermal_duration.timestamp.FromNanoseconds(ingress_timestamp)
            payload_thermal_duration.value.FromTimedelta(time_therm)
            serialized_payload = payload_thermal_duration.SerializeToString()
            envelope = keelson.enclose(serialized_payload)
            publisher_timestamp_thermal_start.put(envelope)
            logging.debug(f"...published on {key_timestamp_thermal_start}")

            thermo_frame = thermoIn.getCvFrame().astype(np.float32)

            # logging.debug(f"Thermal frame: {thermo_frame}")
            # logging.debug(f"Min Temperature: {np.min(thermo_frame)} C")
            # logging.debug(f"Max Temperature: {np.max(thermo_frame)} C")

            # Frame of temperatures is in celsius 
            payload_bytes = TimestampedBytes()
            payload_bytes.timestamp.FromNanoseconds(ingress_timestamp)
            payload_bytes.value = thermo_frame.tobytes()
            serialized_payload = payload_bytes.SerializeToString()
            envelope = keelson.enclose(serialized_payload)
            publisher_image_thermal_raw.put(envelope)
            logging.debug(f"...published on {key_image_thermal_raw}")

            # TODO: Maybe insted of raw bytes, we could publish the raw thermal frame as a RawImage?
            # payload_therm_raw = RawImage()
            # payload_therm_raw.timestamp.FromNanoseconds(ingress_timestamp)
            # if args.frame_id is not None:
            #     payload_therm_raw.frame_id = args.frame_id
            # payload_therm_raw.width = thermo_frame.shape[1]
            # payload_therm_raw.height = thermo_frame.shape[0]
            # payload_therm_raw.encoding = "mono16"  # 32-bit float, single channel # 32FC1
            # payload_therm_raw.step = thermo_frame.strides[0]
            # payload_therm_raw.data = thermo_frame.tobytes()
            # serialized_payload = payload_therm_raw.SerializeToString()
            # envelope = keelson.enclose(serialized_payload)
            # key_raw_image_thermal_raw = keelson.construct_pubsub_key(
            #     base_path=args.realm,
            #     entity_id=args.entity_id,
            #     subject="image_raw",
            #     source_id=args.source_id + "/thermal_raw",
            # )
            # session.put(key_raw_image_thermal_raw, envelope)
            # logging.debug(f"...published on {key_raw_image_thermal_raw}")   

            # Normalize the thermal frame to 0-255 range for visualization
            thermo_frame = cv2.normalize(
                thermo_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            thermo_frame = cv2.applyColorMap(thermo_frame, cv2.COLORMAP_MAGMA)
            # cv2.imwrite("thermo.jpg", thermo_frame)

            payload_thermo = RawImage()
            payload_thermo.timestamp.FromNanoseconds(ingress_timestamp)
            if args.frame_id is not None:
                payload_thermo.frame_id = args.frame_id
            payload_thermo.width = thermo_frame.shape[1]
            payload_thermo.height = thermo_frame.shape[0]
            payload_thermo.encoding = "bgr8"
            payload_thermo.step = thermo_frame.strides[0]
            payload_thermo.data = thermo_frame.tobytes()
            serialized_payload = payload_thermo.SerializeToString()
            envelope = keelson.enclose(serialized_payload)
            publisher_image_thermal.put(envelope)
            logging.debug(f"...published on {key_image_thermal}")

            # Update thermal FPS counter and log every 5 seconds
            current_fps = fpsCounter_thermal.tick()
            if fpsCounter_thermal.frameCount == 0:  # This means we just reset, so we calculated new FPS
                logging.info(f"Thermal camera FPS: {current_fps:.2f} Hz")



def run_stereo(session, args):

    logging.info("Running stereo camera...")

    decimation_factor = 1  

    # Stereo image
    key_image_stereo = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="image_raw",
        source_id=args.source_id + "/rgb",
    )
    publisher_image_stereo = session.declare_publisher(
        key_image_stereo, congestion_control=zenoh.CongestionControl.BLOCK
    )
    logging.info(f"Publisher for image at {key_image_stereo}")

    # Disparity RAW
    key_disp_stereo = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="raw",
        source_id=args.source_id + "/disparity",
    )
    publisher_disp_stereo = session.declare_publisher(
        key_disp_stereo, congestion_control=zenoh.CongestionControl.BLOCK
    )
    logging.info(f"Publisher for disparity image at {key_disp_stereo}")

    # Disparity image
    key_disp_stereo_img = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="image_raw",
        source_id=args.source_id + "/disparity_image",
    )
    publisher_disp_stereo_img = session.declare_publisher(
        key_disp_stereo_img, congestion_control=zenoh.CongestionControl.BLOCK
    )
    logging.info(f"Publisher for disparity image at {key_disp_stereo_img}")

    # Depth image
    key_depth_stereo = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="image_raw",
        source_id=args.source_id + "/camera/stereo/depth",
    )
    publisher_depth_stereo = session.declare_publisher(
        key_depth_stereo, congestion_control=zenoh.CongestionControl.BLOCK
    )
    logging.info(f"Publisher for disparity image at {key_depth_stereo}")

    # IMU data
    key_imu = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="imu_reading",
        source_id=args.source_id + "/camera/stereo",
    )
    publisher_imu = session.declare_publisher(key_imu)
    logging.info(f"Publisher for image at {key_imu}")

    # IMU Acceleration
    key_imu_acc = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="linear_acceleration_mpss",
        source_id=args.source_id + "/imu",
    )
    publisher_imu_acc = session.declare_publisher(key_imu_acc)
    logging.info(f"Publisher for IMU acceleration at {key_imu_acc}")

    # IMU Angular velocity
    key_imu_vel = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="angular_velocity_radps",
        source_id=args.source_id + "/imu",
    )
    publisher_imu_vel = session.declare_publisher(key_imu_vel)
    logging.info(f"Publisher for IMU angular velocity at {key_imu_vel}")

    # IMU Acceleration Timestamp since start
    key_imu_acc_timestamp_start = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="device_uptime_duration",
        source_id=args.source_id + "/imu/linear_acceleration",
    )
    publisher_imu_acc_timestamp_start = session.declare_publisher(
        key_imu_acc_timestamp_start
    )
    logging.info(f"Publisher for IMU acceleration timestamp at {key_imu_acc_timestamp_start}")

    # IMU Angular velocity Timestamp since start
    key_imu_vel_timestamp_start = keelson.construct_pubsub_key(
        base_path=args.realm,
        entity_id=args.entity_id,
        subject="device_uptime_duration",
        source_id=args.source_id + "/imu/angular_velocity",
    )
    publisher_imu_timestamp_start = session.declare_publisher(
        key_imu_vel_timestamp_start
    )
    logging.info(f"Publisher for IMU angular velocity timestamp at {key_imu_vel_timestamp_start}")

    logging.info("Pipeline setup STEREO...")
    # --- Connect, configure and run the pipeline
    deviceInfo = dai.DeviceInfo(args.ip_address)

    if deviceInfo is not None:
        pipeline, stereo, queueNames = pipelines.setup_pipeline_OAK_LR(
            fps_stereo=args.fps_stereo,
            fps_rgb=args.fps_rgb,
            enable_imu=args.enable_imu,
            stereo_resolution=args.stereo_resolution,
            confidence_threshold=args.confidence_threshold,
        )
        pipelines.setup_postprocessing(
            stereo, maxRange=args.max_range_meters * 1000
        )  # convert to [mm]
        print(f"Running pipeline for OAK-LR at {args.ip_address}.")

        with dai.Device(pipeline, deviceInfo) as device:

            calibData = device.readCalibration()

            Q_matrix, disparityMultiplier = define_Q_matrix(
                calibData, args.stereo_resolution, stereo
            )

            # Debug Q matrix and disparity multiplier
            logging.debug(f"Q matrix:\n{Q_matrix}")
            logging.debug(f"Disparity multiplier: {disparityMultiplier}")

            fpsCounter_rgb = FPSCounter(output_interval_seconds=5.0)
            fpsCounter_disp = FPSCounter(output_interval_seconds=5.0)
            fpsCounter_imu_acc = FPSCounter(output_interval_seconds=5.0)
            fpsCounter_imu_vel = FPSCounter(output_interval_seconds=5.0)

            baseTs = None
            count = 0
            while not device.isClosed():
                count += 1
                if count >= 10:
                    try:
                        temp_oak = round(device.getChipTemperature().average)
                        cpu_css_oak = round(device.getLeonCssCpuUsage().average * 100)
                        cpu_mss_oak = round(device.getLeonMssCpuUsage().average * 100)
                        logging.debug(
                            f"Temp {temp_oak} C, CPU load CSS {cpu_css_oak}%, MSS {cpu_mss_oak}%"
                        )
                    except RuntimeError:
                        logging.error(
                            "Could not get temperature or CPU usage from device."
                        )
                    count = 0

                latestPacket = {}
                latestPacket["rgb"] = None
                latestPacket["confid"] = None
                latestPacket["disp"] = None
                latestPacket["imu"] = None

                # Get latest packets from all queues, this way admits receiving data with multiple periods
                queueEvents = device.getQueueEvents(queueNames)  # plc- exclude
                for queueName in queueEvents:
                    packets = device.getOutputQueue(
                        queueName
                    ).tryGetAll()  # is reading all at the same time better than reading one at a time?
                    if len(packets) > 0:
                        latestPacket[queueName] = packets[-1]

                # if rgbIn is not None:
                if latestPacket["rgb"] is not None:

                    timestamp = time.time_ns()
                    rgbIn = latestPacket["rgb"]

                    current_fps_rbg = fpsCounter_rgb.tick()
                    if fpsCounter_rgb.frameCount == 0:  # This means we just reset, so we calculated new FPS
                        logging.info(f"RBG: {current_fps_rbg:.2f} FPS") 
                    # Get the BGR frame directly from the camera
                    img_bgr = rgbIn.getCvFrame()

                    payload_rgb = RawImage()
                    # TODO: Look into --> is better rgbIn.getTimestamp().total_seconds()*1e9
                    payload_rgb.timestamp.FromNanoseconds(timestamp)
                    if args.frame_id is not None:
                        payload_rgb.frame_id = args.frame_id
                    payload_rgb.width = rgbIn.getWidth()
                    payload_rgb.height = rgbIn.getHeight()
                    payload_rgb.encoding = "bgr8"  # Default in OpenCV
                    payload_rgb.step = rgbIn.getWidth() * 3  # 3 bytes per pixel (BGR)
                    payload_rgb.data = img_bgr.tobytes()
                    serialized_payload = payload_rgb.SerializeToString()
                    envelope = keelson.enclose(serialized_payload)
                    publisher_image_stereo.put(envelope)
                    logging.debug(f"...published on {key_image_stereo}")

                # if dispIn is not None: #and confIn is not None:
                if latestPacket["disp"] is not None:  # and confIn is not None:

                    timestamp = time.time_ns()

                    dispIn = latestPacket["disp"]
                    disp_timestamp_raw = dispIn.getTimestamp()
                    disp_timestamp = (
                        disp_timestamp_raw.seconds
                        + disp_timestamp_raw.microseconds * 1e-6
                    )
                    logging.debug(f"Disparity timestamp: {disp_timestamp} s")

                    current_fps_disp = fpsCounter_disp.tick()
                    if fpsCounter_disp.frameCount == 0:  # This means we just reset, so we calculated new FPS
                        logging.info(f"Disparity FPS: {current_fps_disp:.2f} FPS") 

                    disp_frame_raw = dispIn.getFrame()  # outputs 16 bit

                    # logging.debug(f"Disparity frame shape: {disp_frame_raw.shape}, dtype: {disp_frame_raw.dtype}")
                    # unique, counts = np.unique(disp_frame_raw, return_counts=True)
                    # unique_counts = dict(zip(unique, counts))
                    # logging.debug(f"Disparity frame unique values: {unique_counts}")
                    # logging.debug(f"Number of zero values in disparity frame: {unique_counts.get(0, 0)}")

                    # payload_disp = TimestampedBytes()
                    # payload_disp.timestamp.FromNanoseconds(timestamp)
                    # payload_disp.value = disp_frame_raw.tobytes() #disp_frame_raw.tobytes() #disp_frame_scaled.tobytes()
                    # serialized_payload = payload_disp.SerializeToString()
                    # envelope = keelson.enclose(serialized_payload)
                    # publisher_disp_stereo.put(envelope)
                    # logging.debug(f"...published on {key_disp_stereo}")

                    payload_disp_img = RawImage()
                    payload_disp_img.timestamp.FromNanoseconds(timestamp)
                    if args.frame_id is not None:
                        payload_disp_img.frame_id = args.frame_id
                    payload_disp_img.width = disp_frame_raw.shape[1]
                    payload_disp_img.height = disp_frame_raw.shape[0]
                    payload_disp_img.encoding = (
                        "16UC1"  # 16-bit unsigned single channel
                    )
                    payload_disp_img.step = (
                        disp_frame_raw.shape[1] * 2
                    )  # 2 bytes per pixel for 16-bit unsigned
                    payload_disp_img.data = disp_frame_raw.tobytes()
                    serialized_payload = payload_disp_img.SerializeToString()
                    envelope = keelson.enclose(serialized_payload)
                    publisher_disp_stereo_img.put(envelope)

                    disp_frame_scaled = (
                        np.ascontiguousarray(disp_frame_raw).astype(np.float32)
                        * disparityMultiplier
                    )  # to be called cv function needs float32

                    # Debug disparity data
                    logging.debug(
                        f"Raw disparity frame - shape: {disp_frame_raw.shape}, dtype: {disp_frame_raw.dtype}"
                    )
                    logging.debug(
                        f"Raw disparity values - min: {np.min(disp_frame_raw)}, max: {np.max(disp_frame_raw)}, mean: {np.mean(disp_frame_raw)}"
                    )
                    logging.debug(
                        f"Non-zero raw disparity values: {np.count_nonzero(disp_frame_raw)}/{disp_frame_raw.size}"
                    )
                    logging.debug(f"Disparity multiplier: {disparityMultiplier}")
                    logging.debug(
                        f"Scaled disparity frame - shape: {disp_frame_scaled.shape}, dtype: {disp_frame_scaled.dtype}"
                    )
                    logging.debug(
                        f"Scaled disparity values - min: {np.min(disp_frame_scaled)}, max: {np.max(disp_frame_scaled)}, mean: {np.mean(disp_frame_scaled)}"
                    )

                    # Check for potential stereo issues
                    if np.max(disp_frame_raw) == 0:
                        logging.warning(
                            "All disparity values are zero! Possible issues:"
                        )
                        logging.warning("1. Stereo cameras not properly configured")
                        logging.warning(
                            "2. No objects in scene or insufficient texture"
                        )
                        logging.warning("3. Confidence threshold too restrictive")
                        logging.warning("4. Lighting conditions poor")
                        logging.warning(
                            f"Current confidence threshold: {args.confidence_threshold}"
                        )

                    # point cloud and depth map
                    # Q_matrix needs to be sent,
                    point_cloud, depth_map = disparity_to_point_cloud(
                        disp_frame_scaled, Q_matrix, decimation_factor=1
                    )

                    # Debug depth map
                    logging.debug(
                        f"Depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}"
                    )
                    logging.debug(
                        f"Depth map min: {np.min(depth_map)}, max: {np.max(depth_map)}, mean: {np.mean(depth_map)}"
                    )
                    logging.debug(
                        f"Non-zero depth values: {np.count_nonzero(depth_map)}/{depth_map.size}"
                    )

                    payload_depth_image = RawImage()
                    payload_depth_image.timestamp.FromNanoseconds(timestamp)
                    if args.frame_id is not None:
                        payload_depth_image.frame_id = args.frame_id
                    payload_depth_image.width = depth_map.shape[1]
                    payload_depth_image.height = depth_map.shape[0]
                    payload_depth_image.encoding = (
                        "32FC1"  # 32-bit float, single channel
                    )
                    payload_depth_image.step = (
                        depth_map.shape[1] * 4
                    )  # 4 bytes per pixel for 32-bit float
                    payload_depth_image.data = depth_map.tobytes()
                    serialized_payload = payload_depth_image.SerializeToString()
                    envelope = keelson.enclose(serialized_payload)
                    publisher_depth_stereo.put(envelope)
                    logging.debug(f"...published on {key_depth_stereo}")

                    # POINT CLOUD
                    # rr.log("depthai/disp_pcl", rr.Points3D(point_cloud/100)) #convert from [cm] to [m]
                    # rr.log("depthai/statistics/pcl_size", rr.Scalars(len(point_cloud))) #number of points in point cloud

                    # Depth map
                    # rr.log("depthai/disp_depth_image_full", rr.Image(depth_map)) #bw, float32

                    # debug: save pointcloud to file
                    # if False:
                    #     pcd = o3d.geometry.PointCloud()
                    #     pcl = point_cloud[::2//decimation_factor,:]/100 #covert to [m]
                    #     pcd.points = o3d.utility.Vector3dVector(pcl)
                    #     o3d.io.write_point_cloud("./testdata.ply", pcd)

                if latestPacket["confid"] is not None:
                    confIn = latestPacket["confid"]
                    conf_frame_raw = confIn.getCvFrame()  # 8 bits, already decimated
                    conf_downsampled = conf_frame_raw[
                        ::decimation_factor, ::decimation_factor
                    ]  # does not make sense to have higher resolution than disparity
                    # rr.log("depthai/confidence", rr.Image(conf_downsampled)) #8 bits, 0-255

                # if imuIn is not None:
                if latestPacket["imu"] is not None:
                    imuIn = latestPacket["imu"]
                    
                    for imuPacket in imuIn.packets:
                        # Get individual timestamps for each sensor reading
                        acceleroValues = imuPacket.acceleroMeter
                        acceleroTd = acceleroValues.getTimestampDevice()
                        
                        # Create unique timestamp using device time + current system time offset
                        # This preserves the device's precise timing while anchoring to system time
                        current_time = time.time_ns()
                        acc_device_time_ns = int(acceleroTd.total_seconds() * 1_000_000_000)
                        acc_unique_timestamp = current_time - (current_time % 1_000_000) + (acc_device_time_ns % 1_000_000)

                        payload_acc_duration = TimestampedDuration()
                        payload_acc_duration.timestamp.FromNanoseconds(current_time)
                        payload_acc_duration.value.FromTimedelta(acceleroTd)
                        serialized_payload = payload_acc_duration.SerializeToString()
                        envelope = keelson.enclose(serialized_payload)
                        publisher_imu_acc_timestamp_start.put(envelope)
                        logging.debug(f"...published on {key_imu_acc_timestamp_start}")

                        payload_acc = Decomposed3DVector()
                        payload_acc.timestamp.FromNanoseconds(acc_unique_timestamp)
                        payload_acc.vector.x = acceleroValues.x
                        payload_acc.vector.y = acceleroValues.y
                        payload_acc.vector.z = acceleroValues.z
                        serialized_payload = payload_acc.SerializeToString()
                        envelope = keelson.enclose(serialized_payload)
                        publisher_imu_acc.put(envelope)
                        logging.debug(f"...published on {key_imu_acc}")

                        # Update IMU acceleration FPS counter and log every 5 seconds
                        current_fps_acc = fpsCounter_imu_acc.tick()
                        if fpsCounter_imu_acc.frameCount == 0:  # This means we just reset, so we calculated new FPS
                            logging.info(f"IMU acceleration FPS: {current_fps_acc:.2f} Hz")

                        gyroValues = imuPacket.gyroscope
                        gyroTs = gyroValues.getTimestampDevice()
                        
                        # Create unique timestamp for gyroscope using the same method
                        gyro_device_time_ns = int(gyroTs.total_seconds() * 1_000_000_000)
                        gyro_unique_timestamp = current_time - (current_time % 1_000_000) + (gyro_device_time_ns % 1_000_000)

                        payload_vel_duration = TimestampedDuration()
                        payload_vel_duration.timestamp.FromNanoseconds(current_time)
                        payload_vel_duration.value.FromTimedelta(gyroTs)
                        serialized_payload = payload_vel_duration.SerializeToString()
                        envelope = keelson.enclose(serialized_payload)
                        publisher_imu_timestamp_start.put(envelope)
                        logging.debug(f"...published on {key_imu_vel_timestamp_start}")

                        payload_vel = Decomposed3DVector()
                        payload_vel.timestamp.FromNanoseconds(gyro_unique_timestamp)
                        payload_vel.vector.x = gyroValues.x
                        payload_vel.vector.y = gyroValues.y
                        payload_vel.vector.z = gyroValues.z
                        serialized_payload = payload_vel.SerializeToString()
                        envelope = keelson.enclose(serialized_payload)
                        publisher_imu_vel.put(envelope)
                        logging.debug(f"...published on {key_imu_vel}")

                        # Update IMU angular velocity FPS counter and log every 5 seconds
                        current_fps_vel = fpsCounter_imu_vel.tick()
                        if fpsCounter_imu_vel.frameCount == 0:  # This means we just reset, so we calculated new FPS
                            logging.info(f"IMU angular velocity FPS: {current_fps_vel:.2f} Hz")
    else:
        print(f"Could not find device at {args.ip_address}.")


def timeDeltaToMilliS(delta) -> float:
    return delta.total_seconds() * 1000


class FPSCounter:
    def __init__(self, output_interval_seconds=None):
        self.frameCount = 0
        self.fps = 0
        self.startTime = time.time()
        self.lastOutputTime = time.time()
        self.output_interval = output_interval_seconds  # If None, use frame-based output (every 10 frames)

    def tick(self):
        self.frameCount += 1
        current_time = time.time()
        
        # Determine if we should output FPS
        should_output = False
        if self.output_interval is not None:
            # Time-based output
            if current_time - self.lastOutputTime >= self.output_interval:
                should_output = True
        else:
            # Frame-based output (original behavior)
            if self.frameCount % 10 == 0:
                should_output = True
        
        if should_output:
            elapsedTime = current_time - self.startTime
            self.fps = self.frameCount / elapsedTime
            self.frameCount = 0
            self.startTime = current_time
            if self.output_interval is not None:
                self.lastOutputTime = current_time
        return self.fps


def disparity_to_point_cloud(disp_frame, Q, decimation_factor):
    """
    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1bc1152bd57d63bc524204f21fde6e02
    Modifed to turn coordinates to x pointing forward, y to the left and z up, from original:

    Use cv to find Q matrix by "stereRectify" and popultate Q matrix from this example:
    https://docs.luxonis.com/software/depthai/examples/calibration_reader/
    """
    # print(
    #     f"Input disparity frame - shape: {disp_frame.shape}, dtype: {disp_frame.dtype}"
    # )
    # print(
    #     f"Disparity values - min: {np.min(disp_frame)}, max: {np.max(disp_frame)}, mean: {np.mean(disp_frame)}"
    # )
    # print(
    #     f"Non-zero disparity values: {np.count_nonzero(disp_frame)}/{disp_frame.size}"
    # )

    points_3D = cv2.reprojectImageTo3D(disp_frame, Q, handleMissingValues=False)
    # print(f"3D points shape: {points_3D.shape}, dtype: {points_3D.dtype}")

    # Extract depth values (Z-coordinates) and convert to positive values
    depth_map = points_3D[:, :, 2].copy()  # Z-coordinates

    # Handle invalid values - set them to 0
    valid_mask = np.isfinite(depth_map) & (disp_frame > 0) & (depth_map > 0)
    depth_map[~valid_mask] = 0

    # print(
    #     f"Depth map after processing - min: {np.min(depth_map)}, max: {np.max(depth_map)}, mean: {np.mean(depth_map)}"
    # )
    # print(f"Valid depth values: {np.count_nonzero(valid_mask)}/{depth_map.size}")

    # For point cloud, only keep valid points
    valid_points_mask = np.isfinite(points_3D).all(axis=2) & (disp_frame > 0)
    points_3D_valid = points_3D[valid_points_mask]  # nb: this also flattens the array
    points_3D_valid = np.stack(
        [-points_3D_valid[:, 2], points_3D_valid[:, 0], points_3D_valid[:, 1]], axis=1
    )  # rotate

    # NOTE: decimation factor > 1 does not currently work, it warps the point cloud

    return (
        points_3D_valid[::decimation_factor, :],
        depth_map[::decimation_factor, ::decimation_factor],
    )


def define_Q_matrix(calibData, stereo_resolution, stereo):
    """
    See opencv documentation for reprojectImageTo3D. Alternativ is to use stereoRectify to get Q matrix.
    Q matrix can be logged to allow for post runtime calculation of point cloud.
    define_Q_matrix(focal_length, cx, cy, baseline):
    """

    intrinsics = calibData.getCameraIntrinsics(calibData.getStereoLeftCameraId())
    reported_base_line = calibData.getBaselineDistance()
    reported_FoV = calibData.getFov(calibData.getStereoLeftCameraId())
    extrinsics = calibData.getCameraExtrinsics(
        calibData.getStereoLeftCameraId(), calibData.getStereoRightCameraId()
    )  # not used
    reported_focal_length = intrinsics[0][0]
    HFOV_LR = reported_FoV
    # print(
    #     f"Reported baseline: {reported_base_line} cm, reported FoV: {reported_FoV} degrees, reported focal length {reported_focal_length}"
    # )

    # Get max disparity to normalize the disparity images
    max_disparity = stereo.initialConfig.getMaxDisparity()
    disparityMultiplier = 95.0 / max_disparity  # uncertain if perfectly correct
    # print(f"Max disparity: {max_disparity}")
    # print(f"Disparity multiplier: {disparityMultiplier}")

    # https://oak-web.readthedocs.io/en/latest/components/nodes/stereo_depth/
    # Dm = (baseline/2) * tan((90 - HFOV / HPixels)*pi/180)
    # Dm_400P = (reported_base_line / 2) * np.tan((90 - reported_FoV / 640) * np.pi / 180)
    # Dm_800P = (reported_base_line / 2) * np.tan((90 - reported_FoV / 1280) * np.pi / 180)
    # print(f"Max stereo distance for 400P: {Dm_400P} cm, and for 800P: {Dm_800P} cm")

    if stereo_resolution == "400p":
        width, height = 640, 400
        cy, cx = (
            width // 2,
            height // 2,
        )  # disp_frame_raw.shape[1] // 2, disp_frame_raw.shape[0] // 2  # Put principal point in the middle of the image
        focal_length = (
            width * 0.5 / np.tan(HFOV_LR * 0.5 * np.pi / 180)
        )  # focal length in pixels
        baseline = reported_base_line
    elif stereo_resolution == "720p":
        raise ValueError(f"Untested resolution: {stereo_resolution}.")
    elif stereo_resolution == "800p":
        width, height = 1280, 800
        cy, cx = width // 2, height // 2
        focal_length = width * 0.5 / np.tan(HFOV_LR * 0.5 * np.pi / 180)
        baseline = reported_base_line
    else:
        raise ValueError(
            f"Unsupported resolution: {stereo_resolution}. Supported values are '800p'(, '720p'), and '400p'."
        )

    Q = np.array(
        [
            [1, 0, 0, -cx],
            [0, 1, 0, -cy],
            [0, 0, 0, focal_length],  # MiS: debug
            [0, 0, -1 / baseline, 0],
        ],
        dtype=np.float32,
    )
    return Q, disparityMultiplier


if __name__ == "__main__":
    main()
