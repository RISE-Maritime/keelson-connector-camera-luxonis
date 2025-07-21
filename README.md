# Keelson Connector Camera Realsense 

Connector based on https://www.intelrealsense.com/sdk-2/ 

The use of OAK T - Thermal camera can require up to 4min until accurate temperature readings is established due to Flat Field Correction (FFC) Initialization.

## Quick start

```bash 
# Thermal Camera
python3 bin/main.py --log-level 10 -r rise -e storakrabban -s camera/thermal --ip-address 192.168.3.12 run_thermal 

# Options 
--enable-imu



# Stereo Camera
python3 bin/main.py --log-level 10 -r rise -e storakrabban -s camera/stereo  --ip-address 192.168.3.13 run_stereo 

# Options 
--enable-imu


```

## Python

- [Python DOC](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-tutorial-1-depth.py)
- [Examples code](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-tutorial-1-depth.py)