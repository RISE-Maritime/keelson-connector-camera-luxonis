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