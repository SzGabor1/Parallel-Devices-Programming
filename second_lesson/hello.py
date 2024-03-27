import pyopencl as cl
import numpy as np

kernel_code = """
__kernel void hello_kernel(__global int* buffer, int n) {
    int gid = get_global_id(0);
    if (gid < n) {
        buffer[gid] = 11;
    }
}
"""

SAMPLE_SIZE = 1000

def main():
    platforms = cl.get_platforms()
    platform = platforms[0]  # Assuming there is at least one platform

    devices = platform.get_devices(device_type=cl.device_type.GPU)
    device = devices[0]  # Assuming there is at least one GPU device

    context = cl.Context(devices=[device])

    program = cl.Program(context, kernel_code).build()

    queue = cl.CommandQueue(context)

    host_buffer = np.arange(SAMPLE_SIZE, dtype=np.int32)

    device_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_buffer)

    program.hello_kernel(queue, host_buffer.shape, None, device_buffer, np.int32(SAMPLE_SIZE))

    cl.enqueue_copy(queue, host_buffer, device_buffer).wait()

    for i in range(SAMPLE_SIZE):
        print(f"[{i}] = {host_buffer[i]}")

if __name__ == "__main__":
    main()
