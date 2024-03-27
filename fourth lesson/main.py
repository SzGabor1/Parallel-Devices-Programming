import pyopencl as cl
import numpy as np
import math

def load_kernel_source(path):
    try:
        with open(path, 'r') as source_file:
            kernel_code = source_file.read()
        return kernel_code
    except FileNotFoundError:
        return None

SAMPLE_SIZE = 10

def main():
    platforms = cl.get_platforms()
    platform = platforms[0]  # Assuming there is at least one platform

    devices = platform.get_devices(device_type=cl.device_type.GPU)
    device = devices[0]  # Assuming there is at least one GPU device

    context = cl.Context(devices=[device])

    kernel_code = load_kernel_source(r"fourth lesson\kernel\matrix_multiplication.cl")
    if kernel_code is None:
        print("Failed to load kernel source.")
        return

    program = cl.Program(context, kernel_code).build()

    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

    host_buffer = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE), dtype=np.float32)
    host_a = np.random.rand(SAMPLE_SIZE, SAMPLE_SIZE).astype(np.float32)
    host_b = np.random.rand(SAMPLE_SIZE, SAMPLE_SIZE).astype(np.float32)

    device_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=host_buffer.nbytes)
    device_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_a)
    device_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=host_b)

    kernel = program.matrix_multiplication

    kernel.set_args(device_a, device_b, device_buffer, np.int32(SAMPLE_SIZE))

    # Get the maximum work item sizes supported by the device
    max_work_item_sizes = device.max_work_item_sizes[0]

    # Choose a valid local work size based on the device's maximum work item size
    local_work_size = int(math.sqrt(max_work_item_sizes))
    global_work_size = (SAMPLE_SIZE, SAMPLE_SIZE)

    # Find largest square divisor of the global work size
    while global_work_size[0] % local_work_size != 0:
        local_work_size -= 1

    # Make sure local_work_size is greater than 0
    assert local_work_size > 0

    local_work_size = (local_work_size, local_work_size)

    event = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size)

    queue.finish()

    cl.enqueue_copy(queue, host_buffer, device_buffer).wait()

    for i in range(SAMPLE_SIZE):
        print("Row", i, ":", host_buffer[i])

    queued_time = event.get_profiling_info(cl.profiling_info.QUEUED)
    print("Queued:", queued_time, "ns")

    end_time = event.get_profiling_info(cl.profiling_info.END)
    print("End:", end_time, "ns")

    print("Execution time:", end_time - queued_time, "ns")

if __name__ == "__main__":
    main()
