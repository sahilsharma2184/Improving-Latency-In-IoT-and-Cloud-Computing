class FogNode:
    def __init__(self, node_id, processing_speed):
        self.node_id = node_id
        self.processing_speed = processing_speed  # Speed in tasks per second

    def process_task(self, task_size):
        processing_time = task_size / self.processing_speed
        print(f"Fog Node {self.node_id}: Processing Time = {processing_time:.2f} s")
        return processing_time

def offload_to_fog(nodes, task_size):
    # Find the optimal node with the shortest processing time
    optimal_node = min(nodes, key=lambda node: task_size / node.processing_speed)
    print(f"Selected Fog Node: {optimal_node.node_id}")
    return optimal_node.process_task(task_size)

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
    offload_to_fog(fog_nodes, task_size)

except ValueError:
    print("Please enter valid numeric inputs.")
