#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import argparse
import time

#MiS - Martin Sanfridson, RISE, March 2025


#TODO: add conversion of disparity map to pcl outside of the camera
#TODO: add uncertainty estimation of the depth map?
#TODO: add options as arginputs? need to list all the options
#TODO: add more filtering of the disparity map




def setup_pipeline_OAK_LR(fps=30):
    # The disparity is computed at this resolution, then upscaled to RGB resolution
    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

    # Create pipeline
    pipeline = dai.Pipeline()
    device = dai.Device()
    queueNames = []

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.Camera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    rgbOut = pipeline.create(dai.node.XLinkOut)
    disparityOut = pipeline.create(dai.node.XLinkOut)

    rgbOut.setStreamName("rgb")
    queueNames.append("rgb")
    disparityOut.setStreamName("disp")
    queueNames.append("disp")

    #Properties
    rgbCamSocket = dai.CameraBoardSocket.CAM_A

    camRgb.setBoardSocket(rgbCamSocket)
    camRgb.setSize(1280, 720)
    camRgb.setFps(fps)

    # For now, RGB needs fixed focus to properly align with depth.
    # This value was used during calibration
    try:
        calibData = device.readCalibration2()
        lensPosition = calibData.getLensPosition(rgbCamSocket)
        if lensPosition:
            camRgb.initialControl.setManualFocus(lensPosition)
    except:
        raise
    left.setResolution(monoResolution)
    left.setCamera("left")
    left.setFps(fps)
    right.setResolution(monoResolution)
    right.setCamera("right")
    right.setFps(fps)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(rgbCamSocket)

    # Linking
    camRgb.video.link(rgbOut.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.disparity.link(disparityOut.input)

    camRgb.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
    
    return pipeline


def run_pipeline_OAK_LR(pipeline, deviceInfo, ntry=5):
    quit_watchdog_loop = False

    for i in range(ntry):
        if quit_watchdog_loop:
            break
        try:
            with dai.Device(pipeline,deviceInfo) as device:
                rgbWindowName = "rgb"
                depthWindowName = "depth"
                cv2.namedWindow(rgbWindowName)
                cv2.namedWindow(depthWindowName)

                while True:
                    latestPacket = {}
                    latestPacket["rgb"] = None
                    latestPacket["disp"] = None

                    # Get latest packets from all queues
                    queueEvents = device.getQueueEvents(("rgb", "disp"))
                    for queueName in queueEvents:
                        packets = device.getOutputQueue(queueName).tryGetAll()
                        if len(packets) > 0:
                            latestPacket[queueName] = packets[-1]

                    # Show images
                    if latestPacket["rgb"] is not None:
                        rgb_frame = latestPacket["rgb"].getCvFrame()
                        cv2.imshow(rgbWindowName, rgb_frame)

                    if latestPacket["disp"] is not None:
                        disp_frame = latestPacket["disp"].getFrame()
                        disp_frame = np.ascontiguousarray(disp_frame) #use this at more places
                        cv2.imshow(depthWindowName, disp_frame)

                    # Check if the user pressed 'q' to quit
                    if cv2.waitKey(1) == ord('q'):
                        print(f"Size of disp_frame: {disp_frame.shape}, and size of rgb_frame: {rgb_frame.shape}")
                        quit_watchdog_loop = True
                        break


        except Exception as e:
            print(f'Pipeline for OAT-T failed to run with exception {e}. Retrying to connect in attemp {i+1} of {ntry}')
            time.sleep(3)



#--- Main ---
fps = 30
ntry = 5
OAK_LR_MXID = '18443010E148391300' #this is an OAK-D2 not OAK-D LR

deviceInfo = None
for device in dai.Device.getAllAvailableDevices():
    print(f"MxID: {device.getMxId()}, state: {device.state}, name: {device.name}, platform: {device.platform}, protocol: {device.protocol}")
    if device.getMxId() == OAK_LR_MXID:
        deviceInfo = dai.DeviceInfo(OAK_LR_MXID)
        print(f"Found device with address {device.name} and MxID {OAK_LR_MXID}")

if deviceInfo is not None:
    pipeline_OAK_LR = setup_pipeline_OAK_LR(fps)
    run_pipeline_OAK_LR(pipeline_OAK_LR, deviceInfo, ntry)

