import random
import time
import matplotlib.pyplot as plt

#class to represent an IoT Device
class IoTDevice:
    def __init__(self, device_id):
        """
        Initializing an IoT device with a unique ID.
        Parameters:
        - device_id: Unique identifier for the IoT device
        """
        self.device_id = device_id

    def transmit_data(self, packet_size):
        """
        Simulate data transmission with network latency.
        
        Parameters:
        - packet_size: Size of the packet being transmitted (in bytes)
        
        Returns:
        - network_latency: Randomly generated latency for transmission (in ms)
        """
        network_latency = random.uniform(10, 100)  # Simulate latency between 10-100ms
        time.sleep(network_latency / 1000)  # Convert ms to seconds for sleep
        return network_latency

# Function to simulate latency across multiple IoT devices
def simulate_latency(devices, packet_size):
    """
    Simulate and calculate network latency for each IoT device.
    
    Parameters:
    - devices: List of IoTDevice instances
    - packet_size: Size of the packet being transmitted (in bytes)
    
    Returns:
    - None (plots the latency for each device)
    """
    total_latency = 0
    latencies = []  # Store latency values for each device

    # Calculate latency for each device
    for device in devices:
        latency = device.transmit_data(packet_size)
        print(f"Device {device.device_id}: Latency = {latency:.2f} ms")
        latencies.append(latency)
        total_latency += latency

    # Calculate the average latency across all devices
    avg_latency = total_latency / len(devices)
    print(f"Average Network Latency: {avg_latency:.2f} ms")

    # Plot the latency for each device
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
    # Get the number of IoT devices and packet size from the user
    num_devices = int(input("Enter the number of IoT devices: "))
    packet_size = int(input("Enter the packet size (in bytes): "))

    # Create IoT devices based on the user input
    devices = [IoTDevice(i) for i in range(num_devices)]

    # Run the latency simulation with user-defined inputs
    simulate_latency(devices, packet_size)

except ValueError:
    print("Please enter valid numeric inputs.")