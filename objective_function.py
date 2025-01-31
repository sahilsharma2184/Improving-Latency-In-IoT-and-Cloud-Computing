import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
NUM_DEVICES = 100  # Number of IoT devices
LIMITED_BANDWIDTH = 10_000  # Total available bandwidth (kbps)
PACKET_SIZE_RANGE = (500, 1500)  # Packet size range in kilobits
BASE_LATENCY = 20  # Base network latency (ms)
CONGESTION_FACTOR = 0.1  # Congestion impact factor

# Generate random packet sizes for each device
packet_sizes = np.random.randint(PACKET_SIZE_RANGE[0], PACKET_SIZE_RANGE[1], NUM_DEVICES)

# Calculate transmission time for each device
transmission_times = packet_sizes / LIMITED_BANDWIDTH  # Transmission delay in seconds

# Convert to milliseconds
transmission_times_ms = transmission_times * 1000

# Simulate congestion-based network latency
congestion_latency = BASE_LATENCY + (CONGESTION_FACTOR * np.cumsum(transmission_times_ms))

# Total latency = Transmission delay + Congestion latency
total_latency = transmission_times_ms + congestion_latency

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(range(NUM_DEVICES), total_latency, marker='o', linestyle='-', color='b', label="Total Network Latency")
plt.axhline(y=np.mean(total_latency), color='r', linestyle='--', label=f"Avg Latency ({np.mean(total_latency):.2f} ms)")
plt.xlabel("IoT Device ID")
plt.ylabel("Network Latency (ms)")
plt.title("IoT Network Latency Minimization with Limited Data Rate")
plt.legend()
plt.grid(True)
plt.show()
