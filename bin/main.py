#!/usr/bin/env python3

import cv2
import depthai as dai
import time
import numpy as np

#MiS - Martin Sanfridson, RISE, March 2025
import logging
import zenoh
import warnings
import json
import numpy as np
from collections import deque
from threading import Thread, Event

import cv2

import terminal_inputs
import pipelines
import keelson
from keelson.payloads.foxglove.RawImage_pb2 import RawImage
from keelson.payloads.foxglove.PointCloud_pb2 import PointCloud
from keelson.payloads.foxglove.PackedElementField_pb2 import PackedElementField
from keelson.payloads.ImuReading_pb2 import ImuReading
from keelson.payloads.Primitives_pb2 import TimestampedBytes

def main():

    args = terminal_inputs.terminal_inputs()
    
    # Setup logger      
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(lineno)d]: %(message)s", level=args.log_level
    )
    logging.captureWarnings(True)
    warnings.filterwarnings("once")

    ## Construct session
    logging.info("Opening Zenoh session...")
    conf = zenoh.Config()

    if args.connect is not None:
        conf.insert_json5(zenoh.config.CONNECT_KEY, json.dumps(args.connect))

    with zenoh.open(conf) as session:
  
        # PUBLSIHERS
        key_point_cloud = keelson.construct_pubsub_key(
            base_path=args.realm,
            entity_id=args.entity_id,
            subject="point_cloud",
            source_id=args.source_id,
        )
        publisher_point_cloud = session.declare_publisher(
            key_point_cloud
        )
        logging.info(f"Publisher for point cloud at {key_point_cloud}")


        key_image_color = keelson.construct_pubsub_key(
            base_path=args.realm,
            entity_id=args.entity_id,
            subject="image_raw",
            source_id=args.source_id + "/jpeg",
        )
        publisher_image_color = session.declare_publisher(
            key_image_color
        )
        logging.info(f"Publisher for image at {key_image_color}")


        # Thermal image 
        key_image_thermal = keelson.construct_pubsub_key(
            base_path=args.realm,
            entity_id=args.entity_id,
            subject="image_raw",
            source_id=args.source_id + "/thermal",
        )
        publisher_image_thermal = session.declare_publisher(
            key_image_thermal,
            congestion_control=zenoh.CongestionControl.BLOCK
        )
        logging.info(f"Publisher for image at {key_image_thermal}")


        # CMOS RAW image
        key_image_raw_cmos = keelson.construct_pubsub_key(
            base_path=args.realm,
            entity_id=args.entity_id,
            subject="image_raw",
            source_id=args.source_id + "/raw/cmos",
        )
        publisher_image_raw_cmos = session.declare_publisher(
            key_image_raw_cmos,
            # congestion_control=zenoh.CongestionControl.BLOCK
        )
        logging.info(f"Publisher for image at {key_image_raw_cmos}")


        # IMU data
        key_imu = keelson.construct_pubsub_key(
            base_path=args.realm,
            entity_id=args.entity_id,
            subject="imu_reading",
            source_id=args.source_id,
        )
        publisher_imu = session.declare_publisher(
            key_imu
        )
        logging.info(f"Publisher for image at {key_imu}")


        logging.info("Pipeline setup...")
        pipeline_OAK_T = pipelines.setup_pipeline_OAK_T(args.fps) 
        logging.info("Device setup... this may take a while...")
        device_info = dai.DeviceInfo(args.ip_address)


        with dai.Device(deviceInfo=device_info, pipeline=pipeline_OAK_T) as device:

            logging.info(f"MxID: {device.getMxId()}, name: {device.getDeviceName()}, platform: {device.getDeviceInfo()}")

            videoQueue = device.getOutputQueue(name="video", maxSize=1, blocking=False)
            thermoQueue = device.getOutputQueue(name="thermo", maxSize=1, blocking=False)
            imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)

            while True:
                videoIn = videoQueue.get() 
                thermoIn = thermoQueue.get()
                imuIn = imuQueue.get() #TODO: could put imu queue in a separate thread

                ingress_timestamp = time.time_ns()

                for imuPacket in imuIn.packets:
                    acceleroValues = imuPacket.acceleroMeter
                    gyroValues = imuPacket.gyroscope

                    acceleroTs = acceleroValues.getTimestampDevice()
                    gyroTs = gyroValues.getTimestampDevice()

                    imuF = "{:.06f}"
                    tsF  = "{:.03f}"

                    logging.debug(f"Accelerometer timestamp: {acceleroTs} ms")
                    logging.debug(f"type of acc timestamp: {type(acceleroTs)}")

                    logging.debug(f"Accelerometer [m/s^2]: x: {imuF.format(acceleroValues.x)} y: {imuF.format(acceleroValues.y)} z: {imuF.format(acceleroValues.z)}")
                    logging.debug(f"Gyroscope timestamp: {gyroTs} ms")
                    logging.debug(f"Gyroscope [rad/s]: x: {imuF.format(gyroValues.x)} y: {imuF.format(gyroValues.y)} z: {imuF.format(gyroValues.z)} ")

                    # Publish IMU data
                    payload = ImuReading()
                    payload.timestamp.FromNanoseconds(ingress_timestamp)
                    payload.linear_acceleration.x = acceleroValues.x
                    payload.linear_acceleration.y = acceleroValues.y
                    payload.linear_acceleration.z = acceleroValues.z
                    payload.angular_velocity.x = gyroValues.x
                    payload.angular_velocity.y = gyroValues.y
                    payload.angular_velocity.z = gyroValues.z
                    serialized_payload = payload.SerializeToString()
                    envelope = keelson.enclose(serialized_payload)
                    publisher_imu.put(envelope)


                time_rbg = videoQueue.get().getTimestamp()
                time_therm = thermoQueue.get().getTimestamp() # ).total_seconds()*1000
                
                #Get BGR frame from NV12 encoded video frame to show with opencv
                cmos_fram_bytes = videoIn.getCvFrame() # numpy array 
                logging.debug(f"CMOS frame type: {type(cmos_fram_bytes)}")
                logging.debug(f"CMOS frame NUM: {cmos_fram_bytes.dtype}")
                
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
                logging.debug(f"CMOS frame shape: {cmos_fram_bytes.shape}")
                logging.debug(f"CMOS frame size: {cmos_fram_bytes.size}")
                logging.debug(f"CMOS frame dtype: {cmos_fram_bytes.dtype}")
                logging.debug(f"CMOS frame total byte length: {len(cmos_fram_bytes.tobytes())}, widthstep: {cmos_fram_bytes.strides[0]}")
                logging.debug(f"CMOS frame width: {cmos_fram_bytes.shape[1]}, height: {cmos_fram_bytes.shape[0]}")


                # # Encode the BGR frame as JPEG
                # ret, jpeg_bytes = cv2.imencode('.jpg', cmos_fram_bytes)
                # if not ret:
                #     logging.error("Failed to encode frame as JPEG")
                #     return
                # jpeg_data = jpeg_bytes.tobytes()
        
                # cv2.imwrite("rbg.jpg",  cmos_fram_bytes)


                #Get frame from thermal camera, then normalize and colorize it
                thermo_frame = thermoIn.getCvFrame().astype(np.float32)

                payload_bytes = TimestampedBytes()
                payload_bytes.timestamp.FromNanoseconds(ingress_timestamp)
                payload_bytes.value = thermo_frame.tobytes()
                serialized_payload = payload_bytes.SerializeToString()
                envelope = keelson.enclose(serialized_payload)
                publisher_image_thermal.put(envelope)
                logging.debug(f"...published on {key_image_thermal}")
                logging.debug(f"Thermal frame shape: {thermo_frame.shape}")

                logging.debug(f"Thermal frame type: {thermo_frame}")
                logging.debug(f"Thermal frame NUM: {thermo_frame.dtype}")
                logging.debug(f"Thermal frame shape: {thermo_frame.shape}")
                logging.debug(f"Thermal frame size: {thermo_frame.size}")
                logging.debug(f"Thermal frame total byte length: {len(thermo_frame.tobytes())}, widthstep: {thermo_frame.strides[0]}")
                logging.debug(f"Thermal frame width: {thermo_frame.shape[1]}, height: {thermo_frame.shape[0]}")
                logging.debug(f"Thermal frame timestamp: {time_therm} ms")
                thermo_frame = cv2.normalize(thermo_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                thermo_frame = cv2.applyColorMap(thermo_frame, cv2.COLORMAP_MAGMA)

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

                cv2.imwrite("thermo.jpg", thermo_frame)


            # while True:
            #     # Wait for a coherent pair of frames: depth and color
            #     frames = pipeline.wait_for_frames()
            #     ingress_timestamp = time.time_ns()
            #     logging.info("Got new frame, at time: %d", ingress_timestamp)

            #     depth_frame = frames.get_depth_frame()
            #     color_frame = frames.get_color_frame()
                
            #     if not depth_frame or not color_frame:
            #         continue

            #     # Convert images to numpy arrays
            #     depth_image = np.asanyarray(depth_frame.get_data())
            #     # logging.debug("Depth image shape: %s", depth_image.shape)


            #     color_image = np.asanyarray(color_frame.get_data())

            #     # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            #     depth_colormap = cv2.applyColorMap(
            #         cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            #     )

            #     buffer.append((depth_colormap, color_image, depth_frame, ingress_timestamp))
        
     

        # try:
          
            
          
            # while True:
            #     try:
            #         depth_colormap, color_image, depth_frame, ingress_timestamp = buffer.pop()
            #     except IndexError:
            #         time.sleep(0.01)
            #         continue
            #     except Exception as e:
            #         logging.error("Error while popping from buffer: %s", e)
            #         continue


            #     logging.debug("Processing raw frame")

            #     height_dep, width_dep, _ = depth_colormap.shape
            #     data_dep = depth_colormap.tobytes()
            #     width_step_dep = len(data_dep) // height_dep 

            #     height_col, width_col, _ = color_image.shape
            #     data_color = color_image.tobytes()
            #     width_step_color = len(data_color) // height_col

            #     logging.debug(
            #         "Frame total byte length: %d, widthstep: %d", len(data_color), width_step_color
            #     )

            #     if "raw_color" in args.publish:
            #         logging.debug("Send RAW COLOR frame...")
            #         payload = RawImage()
            #         payload.timestamp.FromNanoseconds(ingress_timestamp)
            #         if args.frame_id is not None:
            #             payload.frame_id = args.frame_id
            #         payload.width = width_col
            #         payload.height = height_col
            #         payload.encoding = "bgr8"  # Default in OpenCV
            #         payload.step = width_step_color
            #         payload.data = data_color

            #         serialized_payload = payload.SerializeToString()
            #         envelope = keelson.enclose(serialized_payload)
            #         publisher_image_color.put(envelope)
            #         logging.debug(f"...published on {key_image_color}")

            #     if "raw_depth" in args.publish:
            #         logging.debug("Send RAW DEPTH frame...")
            #         payload = RawImage()
            #         payload.timestamp.FromNanoseconds(ingress_timestamp)
            #         if args.frame_id is not None:
            #             payload.frame_id = args.frame_id
            #         payload.width = width_dep
            #         payload.height = height_dep
            #         payload.encoding = "bgr8"
            #         payload.step = width_step_dep
            #         payload.data = data_dep

            #         serialized_payload = payload.SerializeToString()
            #         envelope = keelson.enclose(serialized_payload)
            #         publisher_image_depth.put(envelope)
            #         logging.debug(f"...published on {key_image_depth}")

            #     if "point_cloud" in args.publish:
            #         logging.debug("Send POINT CLOUD frame...")

            #         payload = PointCloud()
            #         payload.timestamp.FromNanoseconds(ingress_timestamp)
            #         if args.frame_id is not None:
            #             payload.frame_id = args.frame_id


            #         # Zero relative position
            #         payload.pose.position.x = 0
            #         payload.pose.position.y = 0
            #         payload.pose.position.z = 0

            #         # Identity quaternion
            #         payload.pose.orientation.x = 0
            #         payload.pose.orientation.y = 0
            #         payload.pose.orientation.z = 0
            #         payload.pose.orientation.w = 1

            #         # Fields
            #         payload.fields.add(name="x", offset=0, type=PackedElementField.FLOAT64)
            #         payload.fields.add(name="y", offset=8, type=PackedElementField.FLOAT64)
            #         payload.fields.add(name="z", offset=16, type=PackedElementField.FLOAT64)

            #         # Generate point cloud
            #         pc = rs.pointcloud()
            #         points = pc.calculate(depth_frame)
            #         logging.debug("Point cloud calculated %s", points)
            #         vtx = np.asanyarray(points.get_vertices())
            #         logging.debug(f"Point cloud shape: {vtx.shape} " )
            #         logging.debug("Point cloud: %s", vtx)

            #         # Ensure the point cloud data is in float64 format
            #         vtx_float64 = np.zeros(vtx.shape, dtype=[('x', np.float64), ('y', np.float64), ('z', np.float64)])
            #         vtx_float64['x'] = vtx['f0'].astype(np.float64)
            #         vtx_float64['y'] = vtx['f1'].astype(np.float64)
            #         vtx_float64['z'] = vtx['f2'].astype(np.float64)

            #         data = vtx_float64.tobytes()
            #         payload.point_stride = len(data) // len(vtx_float64)  # 3 fields (x, y, z) each of 8 bytes (float64)
            #         payload.data = data


            #         serialized_payload = payload.SerializeToString()
            #         envelope = keelson.enclose(serialized_payload)
            #         publisher_point_cloud.put(envelope)
            #         logging.debug(f"...published on {key_point_cloud}")



        # except KeyboardInterrupt:
        #     logging.info("Closing down on user request!")
        #     logging.debug("Joininye :)")



if __name__ == "__main__":
    main()