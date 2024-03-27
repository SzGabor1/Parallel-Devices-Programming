import pyopencl as cl

def main():
    try:
        platforms = cl.get_platforms()
        print("Detected OpenCL platforms:", len(platforms))

        platform = platforms[0]
        platform_name = platform.get_info(cl.platform_info.NAME)
        print("Platform:", platform_name)

        devices = platform.get_devices(device_type=cl.device_type.GPU)
        print("Number of devices:", len(devices))

        device = devices[0]
        device_name = device.get_info(cl.device_info.NAME)
        print("Device name:", device_name)

    except Exception as e:
        print("[ERROR] An error occurred:", e)

if __name__ == "__main__":
    main()
