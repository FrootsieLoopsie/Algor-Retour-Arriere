
import math

class Edge:
    
    node_from = None
    node_to = None
    max_capacity = -1
    current_traffic = -1
    
    def __init__(self, from_node, to_node, capacity):
        self.node_from = from_node
        self.node_to = to_node
        self.max_capacity = capacity
        
    def is_full_capacity(self):
        return self.max_capacity > 0 and self.max_capacity <= self.current_traffic
        
    def add_traffic(self, new_traffic):
        if(new_traffic <= 0):
            return new_traffic
        traffic_that_can_be_added = self.max_capacity - self.current_traffic
        traffic_added = math.max(0, math.min(new_traffic, traffic_that_can_be_added))
        self.current_traffic += traffic_added
        return new_traffic - traffic_added
       
    def reduce_traffic_to_zero(self):
        traffic_reduced = self.current_traffic
        self.current_traffic = 0
        return traffic_reduced
    
    def reduce_capacity_to_zero(self):
        self.max_capacity = 0
        return self.reduce_capacity_to_zero()
        
    

class Node:
    
    name = ""
    outgoing_edges = []
    ingoing_edges = []
    
    def __init__(self, node_name):
        self.name = node_name
    
    def add_outgoing(self, node, capacity):
        self.outgoing_edges.append(Edge(self, node, capacity))
        
    def add_ingoing(self, node, capacity):
        self.ingoing_edges.append((node, capacity)) 
    
    def is_source(self):
        return len(self.outgoing_edges) > 0 and len(self.ingoing_edges) == 0

    def is_sink(self):
        return len(self.outgoing_edges) == 0 and len(self.ingoing_edges) > 0
    

