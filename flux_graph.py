from collections import deque
import heapq
import csv

class Edge:

    def __init__(self, from_node, to_node, capacity):
        self.node_from = from_node
        self.node_to = to_node
        self._max_capacity = capacity
        self.remaining_capacity = capacity
        self.is_blocked = False
        
    def __lt__(self, other):
        return self.remaining_capacity > other.remaining_capacity
        
    def is_full(self):
        return self.remaining_capacity <= 0
    
    def has_traffic(self):
        return (not self.is_blocked) and (self.remaining_capacity < self._max_capacity)
    
    def get_capacity(self):
        return self.remaining_capacity
        
    def reduce_capacity(self, new_traffic):
        if(new_traffic <= 0):
            return new_traffic
        traffic_added = min(new_traffic, self.remaining_capacity)
        self.remaining_capacity = self.remaining_capacity - traffic_added
        return new_traffic - traffic_added
    
    def reset_capacity(self):
        self.is_blocked = False
        self.remaining_capacity = self._max_capacity
    
    def reduce_capacity_to_zero(self):
        self.is_blocked = True
        self.remaining_capacity = 0



class Node:

    def __init__(self, node_name):
        self.name = node_name
        self.outgoing_edges_heap = []
        self.ingoing_edges_heap = []
        self.max_flux_capacity = 0
    
    def add_outgoing(self, edge):
        self.max_flux_capacity = max(edge.remaining_capacity, self.max_flux_capacity)
        heapq.heappush(self.outgoing_edges_heap, edge)
        
    def add_ingoing(self, edge):
        heapq.heappush(self.ingoing_edges_heap, edge)
        
    def get_highest_capacity_outgoing_edge(self):
        return self.outgoing_edges_heap[0]
    
    def pop_highest_capacity_ingoing_edge(self):
        return heapq.heappop(self.ingoing_edges_heap)
    
    def is_source(self):
        return len(self.outgoing_edges_heap) > 0 and len(self.ingoing_edges_heap) == 0

    def is_sink(self):
        return len(self.outgoing_edges_heap) == 0 and len(self.ingoing_edges_heap) > 0
   
   
   
class FluxGraph:
    
    def __init__(self, csvFileName):
        self.source_node = None
        self.sink_node = None
        self.nodes = {}
        self.edges = []
        self.most_clogged_edge = None
        self.num_edges = 0
        self.num_edges_to_remove = 0
        self.flow = 0
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
                        self.edges.append(Edge(node, node_to, int(cvs_edge[1])))
                        node.add_outgoing(self.edges[len(self.edges) - 1])
                        node_to.add_ingoing(self.edges[len(self.edges) - 1])
                        
                print(node.name + " -> ", end="")
                for edge in node.outgoing_edges_heap:
                    print(edge.node_to.name + "(" + str(edge.get_capacity()) + ") ", end="")    
                print()
                        
            # Find the source:
            while(not node.is_source()):
                node = node.ingoing_edges_heap[0].node_from
            self.source_node = node
        self.recalculate_flow()
                        
    
    def recalculate_flow(self):
        self.flow = 0
        for edge in self.edges:
            edge.reset_capacity()
        while True:
            augmenting_path = self._find_augmenting_path_bfs()
            if augmenting_path is None:
                break
            
            # Find the bottleneck capacity of the augmenting path
            bottleneck_capacity = min(edge.capacity for edge in augmenting_path)
            
            # Update the flow and residual graph
            for edge in augmenting_path:
                edge.reduce_capacity(bottleneck_capacity)
                
            self.flow += bottleneck_capacity

    

    def _find_augmenting_path_bfs(self):
        visited = set()
        queue = deque([(self.source_node, [])])
        while queue:
            node, path = queue.popleft()
            if node == self.sink_node:
                return path
            visited.add(node)

            return None
            for edge in node.outgoing_edges_heap:
                if not edge.is_full() > 0 and edge.to_node not in visited:
                    queue.append((edge.to_node, path + [edge]))

        return None



fg = FluxGraph("ex1.csv")
print(fg.flow)