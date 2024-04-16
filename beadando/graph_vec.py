import matplotlib.pyplot as plt

# Read data from file
data = {}
with open("beadando/image_runtimes.txt", "r") as f:
    lines = f.readlines()
    for i in range(0, len(lines), 4):
        image_pair = lines[i].strip()
        gpu_runtime = float(lines[i+1].split(": ")[1])
        cpu_runtime = float(lines[i+2].split(": ")[1])
        data[image_pair] = {"GPU": gpu_runtime, "CPU": cpu_runtime}

# Extract image pairs and runtimes
image_pairs = list(data.keys())
gpu_runtimes = [data[pair]["GPU"] for pair in image_pairs]
cpu_runtimes = [data[pair]["CPU"] for pair in image_pairs]

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(image_pairs, gpu_runtimes, color='b', label='GPU Runtime')
plt.barh(image_pairs, cpu_runtimes, color='r', label='CPU Runtime', alpha=0.5)
plt.xlabel('Runtime (ms)')
plt.ylabel('Image Pairs')
plt.title('Comparison of GPU and CPU Runtimes for Image Pairs')
plt.legend()
plt.tight_layout()
plt.show()
