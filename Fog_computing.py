class FogNode:
    def __init__(self, node_id, processing_speed):
        self.node_id = node_id
        self.processing_speed = processing_speed  # Speed in tasks per second

    def process_task(self, task_size):
        processing_time = task_size / self.processing_speed
        print(f"Fog Node {self.node_id}: Processing Time = {processing_time:.2f} s")
        return processing_time

def offload_to_fog(nodes, task_size):
    optimal_node = min(nodes, key=lambda node: task_size / node.processing_speed)
    return optimal_node.process_task(task_size)

# Example usage
fog_nodes = [FogNode(i, speed) for i, speed in enumerate([5, 10, 8])]  # Nodes with varying speeds
offload_to_fog(fog_nodes, task_size=50)
