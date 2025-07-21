import depthai as dai
from datetime import timedelta

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
    if stereo_resolution == "800p": #this is actually maximum for stereo
        left.setIspScale(2, 3) #800p
        right.setIspScale(2, 3) 
    elif stereo_resolution == "720p":
        left.setIspScale(5, 3) #720p
        right.setIspScale(5, 3)
    elif stereo_resolution == "400p":
        left.setIspScale(1, 3) #400p
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



