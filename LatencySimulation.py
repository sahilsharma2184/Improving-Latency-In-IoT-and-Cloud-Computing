import random
import time

class IoTDevice:
    def __init__(self, device_id):
        self.device_id = device_id

    def transmit_data(self, packet_size):
        network_latency = random.uniform(10, 100)  # Simulate latency between 10-100ms
        time.sleep(network_latency / 1000)  # Convert ms to seconds
        return network_latency

def simulate_latency(devices, packet_size):
    total_latency = 0
    for device in devices:
        latency = device.transmit_data(packet_size)
        print(f"Device {device.device_id}: Latency = {latency:.2f} ms")
        total_latency += latency

    avg_latency = total_latency / len(devices)
    print(f"Average Network Latency: {avg_latency:.2f} ms")

# Example usage
devices = [IoTDevice(i) for i in range(5)]
simulate_latency(devices, packet_size=256)
