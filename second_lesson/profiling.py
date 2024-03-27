import pyopencl as cl
import numpy as np

def load_kernel_source(path):
    try:
        with open(path, 'r') as source_file:
            kernel_code = source_file.read()
        return kernel_code
    except FileNotFoundError:
        return None

SAMPLE_SIZE = 1000

def main():
    platforms = cl.get_platforms()
    platform = platforms[0]  # Assuming there is at least one platform

    devices = platform.get_devices(device_type=cl.device_type.GPU)
    device = devices[0]  # Assuming there is at least one GPU device

    context = cl.Context(devices=[device])

    kernel_code = load_kernel_source("second_lesson/kernels/sample.cl")
    if kernel_code is None:
        print("Failed to load kernel source.")
        return

    program = cl.Program(context, kernel_code).build()

    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

    host_buffer = np.arange(SAMPLE_SIZE, dtype=np.int32)

    device_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_buffer)

    kernel = program.sample_kernel

    kernel.set_args(device_buffer, np.int32(SAMPLE_SIZE))

    local_work_size = 256
    global_work_size = ((SAMPLE_SIZE + local_work_size - 1) // local_work_size) * local_work_size

    event = cl.enqueue_nd_range_kernel(queue, kernel, (global_work_size,), (local_work_size,))


    queue.finish()

    queued_time = event.get_profiling_info(cl.profiling_info.QUEUED)
    print("Queued:", queued_time, "ns")

    end_time = event.get_profiling_info(cl.profiling_info.END)
    print("End:", end_time, "ns")

    cl.enqueue_copy(queue, host_buffer, device_buffer).wait()

    for i in range(SAMPLE_SIZE):
         print(f"[{i}] = {host_buffer[i]}")

if __name__ == "__main__":
    main()
