import pyrealsense2 as rs

# Create a pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Enable the default depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    # Start the pipeline with the default configuration
    pipeline.start(config)
    print("Pipeline started successfully with default configuration.")
    
    # Stop the pipeline
    pipeline.stop()
    
    # Reconfigure with custom settings
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    
    # Restart the pipeline with custom configuration
    pipeline.start(config)
    print("Pipeline started successfully with custom configuration.")
except RuntimeError as e:
    print(f"Failed to start pipeline: {e}")