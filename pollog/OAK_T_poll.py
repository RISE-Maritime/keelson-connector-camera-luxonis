#!/usr/bin/env python3

import cv2
import depthai as dai
import time
import numpy as np
from datetime import timedelta
import argparse

#MiS - Martin Sanfridson, RISE, March 2025

#TODO: add options as arginputs?
#TODO: check if rectification is needed of the thermal camera and possible colorcamera, do actual intrinsic calibration

#Alignment of the thermal camera and the RGB camera is not done in the pipeline, something for the post processing


def setup_pipeline_OAK_T(fps=10):
    # Create pipeline
    pipeline = dai.Pipeline()
    #RGB
    camRgb = pipeline.create(dai.node.ColorCamera)  #note: not using the class videoencode
    camRgb.setFps(fps)
    #xoutVideo = pipeline.create(dai.node.XLinkOut)
    #xoutVideo.setStreamName("video")
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setVideoSize(1920, 1080)
    camRgb.setInterleaved(False)
    #xoutVideo.input.setBlocking(False)
    #xoutVideo.input.setQueueSize(1)
    #camRgb.video.link(xoutVideo.input)

    #TODO: check this is uncompressed image format

    #thermo
    camThermo = pipeline.create(dai.node.Camera)
    camThermo.setFps(fps)
    #xlinkOut = pipeline.create(dai.node.XLinkOut)
    #camThermo.raw.link(xlinkOut.input) #changed from preview to raw
    camThermo.setBoardSocket(dai.CameraBoardSocket.CAM_E)
    #xlinkOut.setStreamName("thermo")
    #xlinkOut.input.setBlocking(False)
    #xlinkOut.input.setQueueSize(1)

    #sync and demux
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=50))
    #doesn't seem to work will that many combinations of {raw, isp, video, preview, still}, gets artifacts in the video stream
    camRgb.isp.link(sync.inputs["video"]) 
    camThermo.raw.link(sync.inputs["thermo"]) #note: with raw and isp intrinsic calibration is not done

    demux = pipeline.create(dai.node.MessageDemux)
    xout1 = pipeline.create(dai.node.XLinkOut)
    xout1.setStreamName("video")
    xout2 = pipeline.create(dai.node.XLinkOut)
    xout2.setStreamName("thermo")

    sync.out.link(demux.input)
    demux.outputs["video"].link(xout1.input)
    demux.outputs["thermo"].link(xout2.input)   

    imu = pipeline.create(dai.node.IMU)
    xout3 = pipeline.create(dai.node.XLinkOut)
    xout3.setStreamName("imu")

    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 100)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)
    imu.out.link(xout3.input)

    return pipeline



def timeDeltaToMilliS(delta) -> float:
        return delta.total_seconds()*1000


def run_pipeline_OAK_T(pipeline, deviceInfo, ntry=5):
    quit_watchdog_loop = False

    for i in range(ntry):
        if quit_watchdog_loop:
            break
        try:
            with dai.Device(pipeline,deviceInfo) as device:
                #print('Connected cameras:', device.getConnectedCameraFeatures())
                #print('Device name:', device.getDeviceName(), ' Product name:', device.getProductName())

                videoQueue = device.getOutputQueue(name="video", maxSize=1, blocking=False)
                thermoQueue = device.getOutputQueue(name="thermo", maxSize=1, blocking=False)
                imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)

                baseTs = None
                while True:
                    videoIn = videoQueue.get() 
                    thermoIn = thermoQueue.get()
                    imuIn = imuQueue.get() #TODO: could put imu queue in a separate thread

                    for imuPacket in imuIn.packets:
                        acceleroValues = imuPacket.acceleroMeter
                        gyroValues = imuPacket.gyroscope

                        acceleroTs = acceleroValues.getTimestampDevice()
                        gyroTs = gyroValues.getTimestampDevice()
                        if baseTs is None:
                            baseTs = acceleroTs if acceleroTs < gyroTs else gyroTs
                        acceleroTs = timeDeltaToMilliS(acceleroTs - baseTs)
                        gyroTs = timeDeltaToMilliS(gyroTs - baseTs)

                        imuF = "{:.06f}"
                        tsF  = "{:.03f}"

                        print(f"Accelerometer timestamp: {tsF.format(acceleroTs)} ms")
                        print(f"Accelerometer [m/s^2]: x: {imuF.format(acceleroValues.x)} y: {imuF.format(acceleroValues.y)} z: {imuF.format(acceleroValues.z)}")
                        print(f"Gyroscope timestamp: {tsF.format(gyroTs)} ms")
                        print(f"Gyroscope [rad/s]: x: {imuF.format(gyroValues.x)} y: {imuF.format(gyroValues.y)} z: {imuF.format(gyroValues.z)} ")


                    latencyVideo = (dai.Clock.now() - videoQueue.get().getTimestamp()).total_seconds()*1000
                    timediffSensors = (videoQueue.get().getTimestamp() - thermoQueue.get().getTimestamp()).total_seconds()*1000
                    

                    #Get BGR frame from NV12 encoded video frame to show with opencv
                    cmos_frame = videoIn.getCvFrame()
                    cv2.putText(cmos_frame, "Latency {:.2f} ms".format(latencyVideo), (10, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
                    cv2.imshow("video", cmos_frame)


                    #Get frame from thermal camera, then normalize and colorize it
                    thermo_frame = thermoIn.getCvFrame().astype(np.float32)
                    thermo_frame = cv2.normalize(thermo_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    thermo_frame = cv2.applyColorMap(thermo_frame, cv2.COLORMAP_MAGMA)
                    cv2.putText(thermo_frame, "Sensors timediff: {:.2f} ms".format(timediffSensors), (10, 10), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
                    cv2.imshow("thermo", thermo_frame)


                    if cv2.waitKey(1) == ord('q'):
                        print(f"Size of thermo_frame: {thermo_frame.shape}, and size of cmos_frame: {cmos_frame.shape}")
                        quit_watchdog_loop = True
                        break


        except Exception as e:
            print(f'Pipeline for OAT-T failed to run with exception {e}. Retrying to connect in attemp {i+1} of {ntry}')
            time.sleep(3)


#--- Main ---

OAK_T_MXID = '14442C1091E0AECF00' #could also use IP address, but more of a problem with USB cameras
fps = 10
ntry = 5

deviceInfo = None
for device in dai.Device.getAllAvailableDevices():
    print(f"MxID: {device.getMxId()}, state: {device.state}, name: {device.name}, platform: {device.platform}, protocol: {device.protocol}")
    if device.getMxId() == OAK_T_MXID:
        deviceInfo = dai.DeviceInfo(OAK_T_MXID)
        print(f"Found device with address {device.name} and MxID {OAK_T_MXID}")

if deviceInfo is not None:
    pipeline_OAK_T = setup_pipeline_OAK_T(fps) #add more options here later on
    run_pipeline_OAK_T(pipeline_OAK_T, deviceInfo,ntry)


    



