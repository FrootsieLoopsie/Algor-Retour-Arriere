

class Node:
    
    name = ""
    outgoing_edges = []
    ingoing_edges = []
    
    def __init__(node_name):
        name = node_name
    
    def add_outgoing(self, node, capacity):
        self.outgoing_edges.append((node, capacity))
        
    def add_ingoing(self, node, capacity):
        self.outgoing_edges.append((node, capacity)) 