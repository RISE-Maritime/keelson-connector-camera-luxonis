#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
import time
import rerun as rr  # pip install rerun-sdk
import open3d as o3d

#MiS - Martin Sanfridson, RISE, June 2025

'''
Some notes on decisions made in this code:
- Try have as hight FPS as possible
- Use 400P rather than downssampling
- Extract only disparity map from the camera, then convert to depth and point cloud at the receiver side. 
- Reason: Depth map does not take extra camera CPU, but point cloud does.
- Extended disparity is only for short range, so not used.

#the default is good enough, but in this file we set the parameters manually
#stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT) 
#stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
#stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)


For more information on the OAK-LR, stereo and pointcloud, see:
https://docs.luxonis.com/software/depthai-components/nodes/stereo_depth/#StereoDepth-Currently%20configurable%20blocks
https://docs.luxonis.com/software/depthai/examples/rgb_depth_aligned/

For real-time properities, see https://docs.luxonis.com/software/depthai-components/nodes/stereo_depth/#StereoDepth-Stereo%20depth%20FPS

Ses also
https://oak-web.readthedocs.io/en/latest/components/nodes/stereo_depth/


#way to calculate depth from disparity
def disparity_to_depth(disparity_map, focal_length_px, baseline_mm):
    # Avoid division by zero, disparity map of type uint16
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    valid_disparity = disparity_map > 0  # Mask for valid disparity values
    depth_map[valid_disparity] = (focal_length_px * baseline_mm) / disparity_map[valid_disparity]
    return depth_map

    

 '''

class FPSCounter:
    def __init__(self):
        self.frameCount = 0
        self.fps = 0
        self.startTime = time.time()

    def tick(self):
        self.frameCount += 1
        if self.frameCount % 10 == 0:
            elapsedTime = time.time() - self.startTime
            self.fps = self.frameCount / elapsedTime
            self.frameCount = 0
            self.startTime = time.time()
        return self.fps


def timeDeltaToMilliS(delta) -> float:
        return delta.total_seconds()*1000



def define_Q_matrix(calibData, stereo_resolution):
    '''
    See opencv documentation for reprojectImageTo3D. Alternativ is to use stereoRectify to get Q matrix.
    Q matrix can be logged to allow for post runtime calculation of point cloud.
    define_Q_matrix(focal_length, cx, cy, baseline):
    '''

    intrinsics = calibData.getCameraIntrinsics(calibData.getStereoLeftCameraId())
    reported_base_line = calibData.getBaselineDistance()
    reported_FoV = calibData.getFov(calibData.getStereoLeftCameraId() )
    extrinsics = calibData.getCameraExtrinsics(calibData.getStereoLeftCameraId() ,calibData.getStereoRightCameraId() ) #not used
    reported_focal_length = intrinsics[0][0]
    HFOV_LR = reported_FoV
    print(f"Reported baseline: {reported_base_line} cm, reported FoV: {reported_FoV} degrees, reported focal length {reported_focal_length}")
   

    # Get max disparity to normalize the disparity images
    disparityMultiplier = 95.0 / stereo.initialConfig.getMaxDisparity() #uncertain if perfectly correct
    print(f"Disparity multiplier: {disparityMultiplier}") 

    #https://oak-web.readthedocs.io/en/latest/components/nodes/stereo_depth/
    #Dm = (baseline/2) * tan((90 - HFOV / HPixels)*pi/180)
    #Dm_400P = (reported_base_line / 2) * np.tan((90 - reported_FoV / 640) * np.pi / 180)
    #Dm_800P = (reported_base_line / 2) * np.tan((90 - reported_FoV / 1280) * np.pi / 180)
    #print(f"Max stereo distance for 400P: {Dm_400P} cm, and for 800P: {Dm_800P} cm")

    
    if stereo_resolution == '400P':
        width, height = 640, 400
        cy, cx = width//2, height//2 #disp_frame_raw.shape[1] // 2, disp_frame_raw.shape[0] // 2  # Put principal point in the middle of the image
        focal_length = width * 0.5 / np.tan(HFOV_LR * 0.5 * np.pi / 180) #focal length in pixels
        baseline = reported_base_line
    elif stereo_resolution == '720P':
        raise ValueError(f"Untested resolution: {stereo_resolution}.")
    elif stereo_resolution == '800P':
        width, height = 1280, 800
        cy, cx = width//2, height//2
        focal_length = width * 0.5 / np.tan(HFOV_LR * 0.5 * np.pi / 180)
        baseline = reported_base_line
    else:
        raise ValueError(f"Unsupported resolution: {stereo_resolution}. Supported values are '800P'(, '720P'), and '400P'.")

    Q = np.array([[1, 0, 0, -cx],
                [0, 1, 0, -cy],
                [0, 0, 0, focal_length], #MiS: debug
                [0, 0, -1 / baseline, 0]], dtype=np.float32)
    return Q, disparityMultiplier


def disparity_to_point_cloud(disp_frame, Q, decimation_factor):
    '''
    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1bc1152bd57d63bc524204f21fde6e02
    Modifed to turn coordinates to x pointing forward, y to the left and z up, from original:

    Use cv to find Q matrix by "stereRectify" and popultate Q matrix from this example:
    https://docs.luxonis.com/software/depthai/examples/calibration_reader/
   '''
    points_3D = cv2.reprojectImageTo3D(disp_frame, Q, handleMissingValues=False)
    depth_map = -points_3D[:, :, 2].copy()  # extract depth values (Z-coordinates)
    valid = np.isfinite(points_3D).all(axis=2) & (disp_frame > 0)
    points_3D = points_3D[valid] #nb: this also flattens the array
    points_3D[:] = np.stack([-points_3D[:, 2], points_3D[:, 0], points_3D[:, 1]], axis=1) #rotate
    
    #NOTE: decimation factor > 1 does not currently work, it warps the point cloud

    return points_3D[::decimation_factor,:], depth_map[::decimation_factor,::decimation_factor]



def setup_pipeline_OAK_LR(fps_stereo=15, fps_rgb=10, stereo_resolution='400P', enable_imu=True, confidence_threshold=255):
    """
    Note that depth and point cloud are not defined as options since they consume too much CPU. Instead, these should be calculated
    at the receiver side from the disparity map.

    Use of the IMU should be avoided since it consumes a lot of CPU from the LR camera.
    """

    #Basic settings with no need to make configurable
    extended_disparity = False  # Closer-in minimum depth, disparity range is doubled (from 95 to 190): not of interest to use
    subpixel = True # Better accuracy for longer distance, fractional disparity 32-levels: of interest to use (possible to set nb of bits also somehow)
    lr_check = True # Better handling for occlusions: if subpixel is enabled, this should be enabled as well

    pipeline = dai.Pipeline()
    queueNames = []

    # Define pipeline nodes
    left = pipeline.create(dai.node.ColorCamera) #always same order?
    center = pipeline.create(dai.node.ColorCamera) #CAM_B
    right = pipeline.create(dai.node.ColorCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    #is this needed stereo.enableDistortionCorrection(True)?

    # Define outputs
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    queueNames.append("rgb")
    xout_disp = pipeline.create(dai.node.XLinkOut)
    xout_disp.setStreamName("disp")
    queueNames.append("disp")

    xout_confidence = pipeline.create(dai.node.XLinkOut)
    xout_confidence.setStreamName("confid")
    queueNames.append("confid")

    left.setBoardSocket(dai.CameraBoardSocket.CAM_B) #same as default assigment, but explicit here
    left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
    left.setCamera("left") 
    left.setFps(fps_stereo)

    center.setBoardSocket(dai.CameraBoardSocket.CAM_A)     
    center.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
    center.setCamera("center") 
    center.setFps(fps_rgb) #could assign a different FPS for this camera, but then the queues have to be handled differently and alignment(?)

    left.setBoardSocket(dai.CameraBoardSocket.CAM_C) #same as default assigment, but explicit here
    right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
    right.setCamera("right") 
    right.setFps(fps_stereo)

    
    # Decimate the frames to reduce CPU load
    if stereo_resolution == '800P': #this is actually maximum for stereo
        left.setIspScale(2, 3) #800P
        right.setIspScale(2, 3) 
    elif stereo_resolution == '720P':
        left.setIspScale(5, 3) #720P
        right.setIspScale(5, 3)
    elif stereo_resolution == '400P':
        left.setIspScale(1, 3) #400P
        right.setIspScale(1, 3) 
    else:
        raise ValueError(f"Unsupported resolution: {stereo_resolution}. Supported values are '800P', '720P', and '400P'.")

    # Basic settings for the stereo depth node
    #Is setDepthAlign() needed, it seems to be increaseing CPU load
    #stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A) #syncs with center camera, could skip center depending on resolution!?
    
   
    stereo.setLeftRightCheck(lr_check)
    stereo.setExtendedDisparity(extended_disparity) 
    stereo.setSubpixel(subpixel)
    stereo.initialConfig.setConfidenceThreshold(confidence_threshold) #255 accepts all, 0 execludes all
    #stereo.setInputResolution(1920, 1200) #seems to make no difference

    # Linking nodes together
    stereo.confidenceMap.link(xout_confidence.input)
    left.isp.link(stereo.left)
    right.isp.link(stereo.right)
    center.isp.link(xout_rgb.input)
    stereo.disparity.link(xout_disp.input) 

    #IMU should not be enabled since it takes a lot of CPU from the LR camera
    if enable_imu:
        imu = pipeline.create(dai.node.IMU)
        xout_imu = pipeline.create(dai.node.XLinkOut)
        xout_imu.setStreamName("imu")
        queueNames.append("imu")
        imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 400)
        imu.setBatchReportThreshold(20) #try also 1
        imu.setMaxBatchReports(20) #try also 10
        imu.out.link(xout_imu.input)

    return pipeline, stereo, queueNames




def setup_postprocessing(stereo, maxRange, decimation_factor=1): 
    #Much of the post processing could also be done offline, but we've got no library for this
    #     
    #For chosing seetings, see Depth presets https://docs.luxonis.com/software/depthai-components/nodes/stereo_depth
    #Post processing, see: https://docs.luxonis.com/software/depthai/examples/depth_post_processing/

    #Setting chosen not to be configurable
    post_filter = 'KERNEL_7x7' # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default), NB:h/w accelerated
    speckle_filter_settings = {
        "speckleRange": 8,  # Range in pixels for speckle filter
        "differenceThreshold": 100  # Difference threshold for speckle filter
    }
    speckle_filter_settings = None
    temporal_filter_settings = {
        "alpha": 0.0,  # Alpha value for temporal filter
        "delta": 0,  # Delta value for temporal filter
        "persistency": 'VALID_1_IN_LAST_5'  # Persistency mode for temporal filter, dynamics depends on sampling rate
    }
    #temporal_filter_settings = None
    spatial_filter_settings = {
        "delta": 3,  #step size boundary
        "holeFillingRadius": 3,  # Radius for hole filling in spatial filter
        "numIterations": 2  # Number of iterations for spatial filter
    }
    spatial_filter_settings = None #this disables


    # Persistency algorithm type.  Members:    
    # PERSISTENCY_OFF :     
    # VALID_8_OUT_OF_8 :     
    # VALID_2_IN_LAST_3 :     
    # VALID_2_IN_LAST_4 :     
    # VALID_2_OUT_OF_8 :     
    # VALID_1_IN_LAST_2 :     
    # VALID_1_IN_LAST_5 :     
    # VALID_1_IN_LAST_8 :     
    # PERSISTENCY_INDEFINITELY : 
    # https://docs.luxonis.com/software/depthai/examples/depth_post_processing/

    if post_filter == 'KERNEL_3x3':
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_3x3)
    elif post_filter == 'KERNEL_5x5':   
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    elif post_filter == 'KERNEL_7x7':
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    elif post_filter == 'MEDIAN_OFF':
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
    else:
        raise ValueError(f"Unsupported post filter: {post_filter}. Supported values are 'KERNEL_3x3', 'KERNEL_5x5', 'KERNEL_7x7', and 'MEDIAN_OFF'.")
    

    #Need to set setDepthAlign()?

    config = stereo.initialConfig.get()
    if speckle_filter_settings is not None:
        config.postProcessing.speckleFilter.enable = True
        config.postProcessing.speckleFilter.speckleRange = speckle_filter_settings["speckleRange"] #e.g. 30
        config.postProcessing.speckleFilter.differenceThreshold = speckle_filter_settings["differenceThreshold"] #integer
    else:
        config.postProcessing.speckleFilter.enable = False

    #config.postProcessing.BrightnessFilter.enable = False

    #https://docs.luxonis.com/software/api/python/#depthai.RawStereoDepthConfig.PostProcessing.TemporalFilter.__init__
    if temporal_filter_settings is not None:
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.temporalFilter.alpha = temporal_filter_settings["alpha"] 
        config.postProcessing.temporalFilter.delta = temporal_filter_settings["delta"] 
        if temporal_filter_settings["persistency"] == 'PERSISTENCY_OFF':
            config.postProcessing.temporalFilter.persistencyMode = config.postProcessing.temporalFilter.PersistencyMode.PERSISTENCY_OFF
        elif temporal_filter_settings["persistency"] == 'VALID_8_OUT_OF_8':
            config.postProcessing.temporalFilter.persistencyMode = config.postProcessing.temporalFilter.PersistencyMode.VALID_8_OUT_OF_8
        elif temporal_filter_settings["persistency"] == 'VALID_2_IN_LAST_3': 
            config.postProcessing.temporalFilter.persistencyMode = config.postProcessing.temporalFilter.PersistencyMode.VALID_2_IN_LAST_3
        elif temporal_filter_settings["persistency"] == 'VALID_2_IN_LAST_4':
            config.postProcessing.temporalFilter.persistencyMode = config.postProcessing.temporalFilter.PersistencyMode.VALID_2_IN_LAST_4
        elif temporal_filter_settings["persistency"] == 'VALID_2_OUT_OF_8':
            config.postProcessing.temporalFilter.persistencyMode = config.postProcessing.temporalFilter.PersistencyMode.VALID_2_OUT_OF_8
        elif temporal_filter_settings["persistency"] == 'VALID_1_IN_LAST_2':
            config.postProcessing.temporalFilter.persistencyMode = config.postProcessing.temporalFilter.PersistencyMode.VALID_1_IN_LAST_2
        elif temporal_filter_settings["persistency"] == 'VALID_1_IN_LAST_5':
            config.postProcessing.temporalFilter.persistencyMode = config.postProcessing.temporalFilter.PersistencyMode.VALID_1_IN_LAST_5
        elif temporal_filter_settings["persistency"] == 'VALID_1_IN_LAST_8':
            config.postProcessing.temporalFilter.persistencyMode = config.postProcessing.temporalFilter.PersistencyMode.VALID_1_IN_LAST_8
        else:
            raise ValueError(f"Unsupported persistency mode: {temporal_filter_settings['persistency']}. Supported values are 'PERSISTENCY_OFF', 'VALID_8_OUT_OF_8', 'VALID_2_IN_LAST_3', 'VALID_2_IN_LAST_4', 'VALID_2_OUT_OF_8', 'VALID_1_IN_LAST_2', 'VALID_1_IN_LAST_5', and 'VALID_1_IN_LAST_8'.")
    else:
        config.postProcessing.temporalFilter.enable = False

    if spatial_filter_settings is not None:
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.delta = spatial_filter_settings["delta"]
        config.postProcessing.spatialFilter.holeFillingRadius = spatial_filter_settings["holeFillingRadius"] #e.g. 3
        config.postProcessing.spatialFilter.numIterations = spatial_filter_settings["numIterations"] #e.g. 1
    else:
        config.postProcessing.spatialFilter.enable = False

    #consider making below more configurable
    config.postProcessing.thresholdFilter.minRange = 400
    config.postProcessing.thresholdFilter.maxRange = maxRange #mm
    config.postProcessing.decimationFilter.decimationFactor = decimation_factor #NB: keeps output image size, leaving "redundant squares"
    config.postProcessing.decimationFilter.decimationMode = config.postProcessing.decimationFilter.DecimationMode.NON_ZERO_MEDIAN 
    #NON_ZERO_MEAN = <DecimationMode.NON_ZERO_MEAN: 2>
    #NON_ZERO_MEDIAN = <DecimationMode.NON_ZERO_MEDIAN: 1>
    #PIXEL_SKIPPING = <DecimationMode.PIXEL_SKIPPING: 0>

    config.postProcessing.filteringOrder = [
        dai.RawStereoDepthConfig.PostProcessing.Filter.MEDIAN,
        dai.RawStereoDepthConfig.PostProcessing.Filter.SPECKLE,
        dai.RawStereoDepthConfig.PostProcessing.Filter.TEMPORAL,
        dai.RawStereoDepthConfig.PostProcessing.Filter.SPATIAL,
        dai.RawStereoDepthConfig.PostProcessing.Filter.DECIMATION,
    ]

    stereo.initialConfig.set(config)




def run_pipeline_OAK_LR(pipeline, deviceInfo, queueNames, stereo_resolution, enable_imu=False, decimation_factor=1):

    rr.init("depthai")
    rr.spawn(memory_limit='10%')

    with dai.Device(pipeline,deviceInfo) as device:
        #device.setLogLevel(dai.LogLevel.INFO)
        #device.setLogOutputLevel(dai.LogLevel.INFO)

        calibData = device.readCalibration() 

        Q_matrix, disparityMultiplier = define_Q_matrix(calibData, stereo_resolution)
        
        fpsCounter_rgb = FPSCounter()
        fpsCounter_disp = FPSCounter()
        fpsCounter_imu = FPSCounter()

        baseTs = None
        count = 0 
        while not device.isClosed():
            count += 1
            if count >= 10: 
                try:
                    temp_oak = round(device.getChipTemperature().average)
                    cpu_css_oak = round(device.getLeonCssCpuUsage().average*100)
                    cpu_mss_oak  = round(device.getLeonMssCpuUsage().average*100)
                    #print(f"Temp {temp_oak} C, CPU load CSS {cpu_css_oak}%, MSS {cpu_mss_oak}%")
                    rr.log("depthai/status/temp", rr.Scalars(temp_oak)) #Celsius
                    rr.log("depthai/status/cpu_css", rr.Scalars(cpu_css_oak)) #%
                    rr.log("depthai/status/cpu_mss", rr.Scalars(cpu_mss_oak)) #%
                except RuntimeError:
                    print("Could not get temperature or CPU usage from device.")
                count = 0

            latestPacket = {}
            latestPacket["rgb"] = None
            latestPacket["confid"] = None
            latestPacket["disp"] = None
            latestPacket["imu"] = None
            
            # Get latest packets from all queues, this way admits receiving data with multiple periods
            queueEvents = device.getQueueEvents(queueNames) #plc- exclude
            for queueName in queueEvents:
                packets = device.getOutputQueue(queueName).tryGetAll() #is reading all at the same time better than reading one at a time?
                if len(packets) > 0:
                    latestPacket[queueName] = packets[-1]


            #if rgbIn is not None:
            if latestPacket["rgb"] is not None:
                rgbIn = latestPacket["rgb"]
                rr.log("depthai/fps_rgb", rr.Scalars(fpsCounter_rgb.tick()))
                rr.log("depthai/image", rr.Image(cv2.cvtColor(rgbIn.getCvFrame(), cv2.COLOR_BGR2RGB))) #check size of image


            #if dispIn is not None: #and confIn is not None:
            if latestPacket["disp"] is not None: #and confIn is not None:
                dispIn = latestPacket["disp"]
                disp_timestamp_raw = dispIn.getTimestamp()
                disp_timestamp = disp_timestamp_raw.seconds + disp_timestamp_raw.microseconds*1e-6
                rr.set_time("disp_timeline",timestamp=disp_timestamp) #UTC time
                rr.log("depthai/fps_disp", rr.Scalars(fpsCounter_disp.tick()))
                disp_frame_raw = dispIn.getFrame() #outputs 16 bit
                disp_frame_scaled = np.ascontiguousarray(disp_frame_raw).astype(np.float32)* disparityMultiplier #to be called cv function needs float32
                rr.log("depthai/disp", rr.Image(disp_frame_raw)) #bw, float32

                #point cloud and depth map
                point_cloud, depth_map = disparity_to_point_cloud(disp_frame_scaled, Q_matrix, decimation_factor=1) 

                rr.log("depthai/disp_pcl", rr.Points3D(point_cloud/100)) #convert from [cm] to [m]
                rr.log("depthai/statistics/pcl_size", rr.Scalars(len(point_cloud))) #number of points in point cloud
                rr.log("depthai/disp_depth_image_full", rr.Image(depth_map)) #bw, float32

                #debug: save pointcloud to file
                if False:
                    pcd = o3d.geometry.PointCloud()
                    pcl = point_cloud[::2//decimation_factor,:]/100 #covert to [m]
                    pcd.points = o3d.utility.Vector3dVector(pcl) 
                    o3d.io.write_point_cloud("./testdata.ply", pcd)

            if latestPacket["confid"] is not None:
                confIn = latestPacket["confid"]
                conf_frame_raw = confIn.getCvFrame() #8 bits, already decimated
                conf_downsampled = conf_frame_raw[::decimation_factor,::decimation_factor] #does not make sense to have higher resolution than disparity
                rr.log("depthai/confidence", rr.Image(conf_downsampled)) #8 bits, 0-255


            #if imuIn is not None:
            if latestPacket["imu"] is not None:
                imuIn = latestPacket["imu"]
                for imuPacket in imuIn.packets:
                    acceleroValues = imuPacket.acceleroMeter
                    gyroValues = imuPacket.gyroscope

                    acceleroTs = acceleroValues.getTimestampDevice()
                    gyroTs = gyroValues.getTimestampDevice()
                    if baseTs is None:
                        baseTs = acceleroTs if acceleroTs < gyroTs else gyroTs
                    acceleroTs = timeDeltaToMilliS(acceleroTs - baseTs)
                    gyroTs = timeDeltaToMilliS(gyroTs - baseTs)

                    rr.log("depthai/fps_imu", rr.Scalars(fpsCounter_imu.tick()))

                    rr.log("depthai/imu/acc_x", rr.Scalars(acceleroValues.x)) 
                    rr.log("depthai/imu/acc_y", rr.Scalars(acceleroValues.y)) 
                    rr.log("depthai/imu/acc_z", rr.Scalars(acceleroValues.z)) 
                    rr.log("depthai/imu/gyro_x", rr.Scalars(acceleroValues.x)) 
                    rr.log("depthai/imu/gyro_y", rr.Scalars(acceleroValues.y))
                    rr.log("depthai/imu/gyro_z", rr.Scalars(acceleroValues.z))

                    imu_timestamp = acceleroTs
                    rr.set_time("imu_timeline",timestamp=imu_timestamp) #UTC time



#--- Settings 

#device name
#OAK_LR_MXID = '14442C10B1401AD000' #an OAK-D LR
OAK_LR_NAME = '192.168.3.13' #NB: need IP address to point to the device not MXID
maxRange = 50*1000  # Maximum range for depth filtering, in mm, NB: size of point cloud increases also
confidence_threshold = 155  # Confidence threshold for point cloud generation, adjust as needed, 255 accepts all, 0 execludes all, does introduce more noise?

#setting that gives 70-80% utilization, not to cause too much jitter
if False:
    fps_stereo = 10
    fps_rgb = 10 #RGB camera runs at 2*fps, so if fps is 5, RGB camera runs at 10 FPS
    stereo_resolution = '400P' #options: '800P', '720P', '400P'; camera resolution for the rgb image is no affected by this setting
    enable_imu = True # Enable IMU, this consumes a lot of CPU from the LR camera
else:
    fps_stereo = 12
    fps_rgb = 12 #RGB camera runs at 2*fps, so if fps is 5, RGB camera runs at 10 FPS
    stereo_resolution = '400P' #options: '800P', '720P', '400P'; camera resolution for the rgb image is no affected by this setting
    enable_imu = False # Enable IMU, this consumes a lot of CPU from the LR camera



#--- Connect, configure and run the pipeline
deviceInfo = dai.DeviceInfo(OAK_LR_NAME)
if deviceInfo is not None:
    pipeline, stereo, queueNames = setup_pipeline_OAK_LR(
        fps_stereo=fps_stereo,
        fps_rgb=fps_rgb,
        enable_imu=enable_imu,
        stereo_resolution=stereo_resolution,
        confidence_threshold=confidence_threshold,
    )
    setup_postprocessing(stereo,maxRange=maxRange)
    print(f"Running pipeline for OAK-LR at {OAK_LR_NAME}.")
    run_pipeline_OAK_LR(pipeline, deviceInfo, queueNames, stereo_resolution=stereo_resolution, enable_imu=enable_imu)
else:
    print(f"Could not find device at {OAK_LR_NAME}.")
