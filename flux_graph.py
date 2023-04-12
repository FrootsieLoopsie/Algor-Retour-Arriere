
import math
import csv

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
    
    def get_capacity_left(self):
        return self.max_capacity - self.current_traffic
        
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
    outgoing_edges = {}
    ingoing_edges = {}
    
    def __init__(self, node_name):
        self.name = node_name
    
    def add_outgoing(self, edge):
        self.outgoing_edges[edge.node_to] = edge
        
    def add_ingoing(self, edge):
        self.ingoing_edges[edge.node_from] = edge
    
    def is_source(self):
        return len(self.outgoing_edges) > 0 and len(self.ingoing_edges) == 0

    def is_sink(self):
        return len(self.outgoing_edges) == 0 and len(self.ingoing_edges) > 0
  
  
class FluxGraph:
    
    source_node = None
    sink_node = None
    nodes = {}
    most_clogged_edge = None
    num_edges = 0
    num_edges_to_remove = 0
    flux = -1
    
    def __init__(self, csvFileName):
        with open(csvFileName, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';')
            csv_rows = list(csvreader)
            
            # Read and add the nodes:
            self.num_edges_to_remove = csv_rows[0][1]
            for i in range(1, len(csv_rows)):
                node_name = csv_rows[i][0]
                self.nodes[node_name] = Node(node_name)
            
            # Read and add the edges:
            for i in range(1, len(csv_rows)):
                node = self.nodes[csv_rows[i][0]]
                for j in range(1, len(csv_rows[i])):
                    cvs_edge = csv_rows[i][j].replace(")", "").split("(")
                    node_to = self.nodes[cvs_edge[0]]
                    edge = Edge(node, node_to, int(cvs_edge[1]))
                    node.add_outgoing(edge)
                    node_to.add_ingoing(edge)
        
        
    
    def recalculate_flux():
        most_clogged_edge = None
        flux = -1
        pass



fg = FluxGraph("ex1.csv")