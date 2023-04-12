
import math
import heapq
import csv

class Edge:
    
    node_from = None
    node_to = None
    __max_capacity = -1
    is_blocked = False
    current_capacity = -1
    
    def __init__(self, from_node, to_node, capacity):
        self.node_from = from_node
        self.node_to = to_node
        self.__max_capacity = capacity
        self.current_capacity = capacity
        
    def __lt__(self, other):
        return self.capacity > other.capacity
        
    def is_full(self):
        return self.current_capacity <= 0
    
    def has_traffic(self):
        return (not self.is_blocked) and (self.current_capacity < self.__max_capacity)
    
    def get_capacity(self):
        return self.current_capacity
        
    def reduce_capacity(self, new_traffic):
        if(new_traffic <= 0):
            return new_traffic
        traffic_added = math.min(new_traffic, self.current_capacity)
        self.current_capacity = self.current_capacity - traffic_added
        return new_traffic - traffic_added
    
    def reset_capacity(self):
        self.is_blocked = False
        self.current_capacity = self.max_capacity
    
    def reduce_capacity_to_zero(self):
        self.is_blocked = True
        self.current_capacity = 0

class Node:
    name = ""
    __outgoing_edges_heap = []
    __ingoing_edges_heap = []
    
    def __init__(self, node_name):
        self.name = node_name
    
    def add_outgoing(self, edge):
        heapq.heappush(self.__outgoing_edges_heap, edge)
        
    def add_ingoing(self, edge):
        heapq.heappush(self.__ingoing_edges_heap, edge)
        
    def pop_highest_capacity_outgoing_edge(self):
        return heapq.heappop(self.__outgoing_edges_heap)
    
    def pop_highest_capacity_ingoing_edge(self):
        return heapq.heappop(self.__ingoing_edges_heap)
    
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
                num_outgoing_edges = len(csv_rows[i])
                if(num_outgoing_edges == 0):
                    self.sink_node = node
                else:
                    for j in range(1, num_outgoing_edges):
                        cvs_edge = csv_rows[i][j].replace(")", "").split("(")
                        node_to = self.nodes[cvs_edge[0]]
                        edge = Edge(node, node_to, int(cvs_edge[1]))
                        node.add_outgoing(edge)
                        node_to.add_ingoing(edge)
                        
    
    def recalculate_flux():
        most_clogged_edge = None
        flux = 0
        pass



fg = FluxGraph("ex1.csv")