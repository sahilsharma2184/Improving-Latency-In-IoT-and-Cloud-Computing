import matplotlib.pyplot as plt

class FogNode:
    def __init__(self, node_id, processing_speed):
        self.node_id = node_id
        self.processing_speed = processing_speed  # Speed in tasks per second

    def process_task(self, task_size):
        processing_time = task_size / self.processing_speed
        print(f"Fog Node {self.node_id}: Processing Time = {processing_time:.2f} s")
        return processing_time

def offload_to_fog(nodes, task_size):
    processing_times = [node.process_task(task_size) for node in nodes]
    
    # Find the optimal node with the shortest processing time
    optimal_node = min(nodes, key=lambda node: task_size / node.processing_speed)
    print(f"Selected Fog Node: {optimal_node.node_id}")
    
    return processing_times

# === User Input Section ===
try:
    num_nodes = int(input("Enter the number of fog nodes: "))
    fog_nodes = []

    # Get processing speed for each fog node
    for i in range(num_nodes):
        speed = float(input(f"Enter processing speed (tasks per second) for Fog Node {i}: "))
        fog_nodes.append(FogNode(i, speed))

    # Get task size input
    task_size = float(input("Enter the task size (number of operations): "))

    # Run the offloading simulation
    processing_times = offload_to_fog(fog_nodes, task_size)

    # Plotting the processing times
    node_ids = [node.node_id for node in fog_nodes]
    
    plt.bar(node_ids, processing_times, color='skyblue')
    plt.xlabel('Fog Node ID')
    plt.ylabel('Processing Time (s)')
    plt.title(f'Processing Time for Task Size {task_size}')
    plt.show()

except ValueError:
    print("Please enter valid numeric inputs.")
