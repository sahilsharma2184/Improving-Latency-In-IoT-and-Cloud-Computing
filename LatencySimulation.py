import random
import time
import matplotlib.pyplot as plt

class IoTDevice:
    def __init__(self, device_id):
        self.device_id = device_id

    def transmit_data(self, packet_size):
        network_latency = random.uniform(10, 100)  # Simulate latency between 10-100ms
        time.sleep(network_latency / 1000)  # Convert ms to seconds
        return network_latency

def simulate_latency(devices, packet_size):
    total_latency = 0
    latencies = []  # Store latencies for each device

    for device in devices:
        latency = device.transmit_data(packet_size)
        print(f"Device {device.device_id}: Latency = {latency:.2f} ms")
        latencies.append(latency)
        total_latency += latency

    avg_latency = total_latency / len(devices)
    print(f"Average Network Latency: {avg_latency:.2f} ms")

    # Plotting the latency for each device
    device_ids = [device.device_id for device in devices]
    
    plt.bar(device_ids, latencies, color='lightblue')
    plt.xlabel('IoT Device ID')
    plt.ylabel('Network Latency (ms)')
    plt.title('Network Latency for Each IoT Device')
    plt.axhline(y=avg_latency, color='r', linestyle='--', label=f'Average Latency ({avg_latency:.2f} ms)')
    plt.legend()
    plt.show()

# === User Input Section ===
try:
    num_devices = int(input("Enter the number of IoT devices: "))
    packet_size = int(input("Enter the packet size (in bytes): "))

    # Create IoT devices based on user input
    devices = [IoTDevice(i) for i in range(num_devices)]

    # Run the simulation with user-defined inputs
    simulate_latency(devices, packet_size)

except ValueError:
    print("Please enter valid numeric inputs.")
