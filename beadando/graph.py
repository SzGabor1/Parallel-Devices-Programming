import matplotlib.pyplot as plt

def read_runtimes(filepath):
    runtimes = {}
    with open(filepath, "r") as file:
        for line in file:
            image_number, runtime = line.strip().split(" - ")
            runtimes[int(image_number)] = float(runtime)
    return runtimes

def plot_runtimes(gpu_runtimes, cpu_runtimes):
    plt.figure(figsize=(10, 6))
    
    # Plot GPU runtimes
    plt.plot(list(gpu_runtimes.keys()), list(gpu_runtimes.values()), marker='o', label='GPU Runtime')
    
    # Plot CPU runtimes
    plt.plot(list(cpu_runtimes.keys()), list(cpu_runtimes.values()), marker='o', label='CPU Runtime')
    
    plt.title('Comparison of GPU and CPU Runtimes')
    plt.xlabel('Image Number')
    plt.ylabel('Runtime (miliseconds)')
    plt.legend()
    plt.grid(True)
    plt.xticks(list(gpu_runtimes.keys()))  # Setting x-ticks to match the image numbers
    plt.tight_layout()
    
    plt.show()

def main():
    gpu_runtimes = read_runtimes("beadando/gpu_runtime.txt")
    cpu_runtimes = read_runtimes("beadando/cpu_runtime.txt")
    
    plot_runtimes(gpu_runtimes, cpu_runtimes)

if __name__ == "__main__":
    main()
