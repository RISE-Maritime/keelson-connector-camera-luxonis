import depthai as dai

# Replace with your actual IP address
ip = "192.168.3.12"
device_info = dai.DeviceInfo(ip)

# Attempt to connect
with dai.Device(deviceInfo=device_info) as device:
    print("Successfully connected to device:", device)