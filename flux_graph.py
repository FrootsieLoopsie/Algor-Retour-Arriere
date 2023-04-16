from collections import deque
import heapq
import csv
import os
import networkx as nx
import matplotlib.pyplot as plt

class Edge:

    def __init__(self, from_node, to_node, capacity):
        self.node_from = from_node
        self.node_to = to_node
        self._max_capacity = capacity
        self.remaining_capacity = capacity
        self.is_blocked = False
        
    def __lt__(self, other):
        return self.remaining_capacity < other.remaining_capacity
        
    def is_full(self):
        return self.remaining_capacity <= 0
    
    def has_flow(self):
        return (not self.is_blocked) and (self.remaining_capacity < self._max_capacity)
    
    def get_flow(self):
        if( self.is_blocked):
            return 0
        else:
            return self._max_capacity - self.remaining_capacity
    
    def get_capacity(self):
        return self.remaining_capacity
    
    def get_max_capacity(self):
        return self._max_capacity
        
    def reduce_capacity(self, new_traffic):
        if(new_traffic <= 0):
            return new_traffic
        traffic_added = min(new_traffic, self.remaining_capacity)
        self.remaining_capacity = self.remaining_capacity - traffic_added
        return new_traffic - traffic_added
    
    def reset_capacity(self, ignore_block = False):
        if(ignore_block or not self.is_blocked):
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

   
   
class FlowNetwork:
    
    def __init__(self, csvFileName):
        self.source_node = None
        self.sink_node = None
        self.nodes = {}
        self.edges = []
        self._most_clogged_edge = None
        self.num_edges = 0
        self.num_edges_to_remove = 0
        self.flow = 0
        
        with open(csvFileName, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=';')
            csv_rows = list(csvreader)
            
            # Read and add the nodes:
            self.num_edges_to_remove = int(csv_rows[0][1])
            for i in range(1, len(csv_rows)):
                node_name = csv_rows[i][0]
                self.nodes[node_name] = Node(node_name)
            
            # Read and add the edges:
            for i in range(1, len(csv_rows)):
                node = self.nodes[csv_rows[i][0]] 
                num_outgoing_edges = len(csv_rows[i]) - 1
                if(num_outgoing_edges == 0):
                    self.sink_node = node
                else:
                    for j in range(1, num_outgoing_edges + 1):
                        cvs_edge = csv_rows[i][j].replace(")", "").split("(")
                        node_to = self.nodes[cvs_edge[0]]
                        self.edges.append(Edge(node, node_to, int(cvs_edge[1])))
                        node.add_outgoing(self.edges[len(self.edges) - 1])
                        node_to.add_ingoing(self.edges[len(self.edges) - 1])

            # Find the source (pretty proud of this heuristic):
            while(not node.is_source()):
                node = node.ingoing_edges_heap[0].node_from
                
            self.source_node = node
        self.recalculate_flow()
                        
    
    def reset_edge_capacities(self, unblock_edges=False):
        for edge in self.edges:
            edge.reset_capacity(unblock_edges)
    
    
    def recalculate_flow(self):
        self.flow = 0
        self.reset_edge_capacities(False)
        while True:
            augmenting_path = self._find_augmenting_path_bfs()
            if augmenting_path is None:
                break
            
            # Find the bottleneck capacity of the augmenting path
            # Since we're using heaps to store the edges, we can just look at the first edge; it's the one with the least capacity
            bottleneck_capacity = min(edge.get_capacity() for edge in augmenting_path)
            
            # Update the flow and residual graph
            for edge in augmenting_path:
                edge.reduce_capacity(bottleneck_capacity)
                
                # Bonus heuristic: 
                if(self._most_clogged_edge == None):
                    self._most_clogged_edge = edge
                    
                elif(self._candidate_edge.get_flow() < edge.get_flow()):
                    self._most_clogged_edge = edge
                
            self.flow += bottleneck_capacity
    

    def _find_augmenting_path_bfs(self):
        visited = set()
        queue = deque([(self.source_node, [])])
        while queue: 
            node, path = queue.popleft()
            
            # Condition d'arrêt:
            if node == self.sink_node:
                return path
            
            # Visiter les prochains noeuds:
            visited.add(node)
            for edge in node.outgoing_edges_heap:
                if (not edge.is_full()) and (edge.node_to not in visited):
                    queue.append((edge.node_to, path + [edge]))
        return None


    def _get_best_edge_to_block_using_brute_force(self):
        min_flow = 2147483648
        min_edges = []
        self.reset_edge_capacities()

        for edge in self.edges:
            edge.reduce_capacity_to_zero()
            self.recalculate_flow()
            
            if(self.flow == 0):
                return (0, edge)
                
            elif(self.flow < min_flow):
                min_edges = [edge]
                min_flow = fg.flow
                
            elif(self.flow == min_flow):
                min_edges.append(edge)
            
            edge.is_blocked = False
            self.reset_edge_capacities()
        
        return (min_flow, min_edges)
    
    
    
    def _get_best_edges_to_block_using_backtracking(self, use_heuristics=False):
        
        # Noter qu'à l'initialisation, le flow initial est déjà calculé par Ford-Fulkerson et BFS.
        
        # On initialise une pile pour stocker les noeuds à explorer dans l'arbre de recherche.
        # Chaque élément représentera une configuration du FlowNetwork:
        #   - Les arcs retirés, et le flow maximal associé.
        #   - Élément initial: aucun arc n'est blocké
        heap = [([], self.flow)]
        
        min_edges = []
        min_flow = []
        
        # Tant que la pile n'est pas vide, on prend le prochain tuple de la pile et on calcule le flot maximal 
        # correspondant à ce noeud en utilisant Ford-Fulkerson.
        while len(heap) > 0:
            edges_removed, flow = heapq.heappop(heap)
            for edge in edges_removed:
                edge.is_blocked = True
                
            
        
        pass
    
    
    
    
    # Bonus!
    def draw(self, fileName = "flux_network_result"):
        G = nx.Graph()
        
        for node_name in self.nodes:
            G.add_node(node_name)
        
        for edge in self.edges:
            edge_name = "(" + str(edge.get_flow()) + "/" + str(edge.get_max_capacity()) + ")"
            edge_color = "navy" if edge.has_flow() else "pink" if edge.is_blocked else "gray"
            G.add_edge(edge.node_from.name, edge.node_to.name, label=edge_name, color=edge_color)
        
        # Draw the graph with node labels, edge labels, edge colors, and arrows
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos=pos)
        nx.draw_networkx_labels(G, pos=pos)
        edge_labels = nx.get_edge_attributes(G, 'label')
        for edge, label in edge_labels.items():
            u, v = edge
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dx = x2 - x1
            dy = y2 - y1
            arrow_length = 0.05
            dx *= 1 - arrow_length / (dx ** 2 + dy ** 2) ** 0.5
            dy *= 1 - arrow_length / (dx ** 2 + dy ** 2) ** 0.5
            plt.arrow(x1, y1, dx, dy, head_width=0.025, head_length=0.05, length_includes_head=True, color=G[u][v]['color'])
            plt.text((x1+x2)/2, (y1+y2)/2, label, horizontalalignment='center', verticalalignment='center', fontsize=10, color=G[u][v]['color'])

        # Save the graph drawing to a file
        plt.savefig("solution_" + fileName + ".png")



# Change working directory to the directory of the current python file:
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


file_name = "ex1.csv"
fg = FlowNetwork(file_name)
print("Max initial flow is: " + str(fg.flow))

for i in range(fg.num_edges_to_remove):
    
    min_flow, min_edges = fg._get_best_edge_to_block_using_brute_force()
    fg.recalculate_flow()
    
    if(len(min_edges) == 1):
        print("Réponse (par force brute): flow de " + str(min_flow) + " en enlevant le Edge allant de " + min_edges[0].node_from.name + " à " + min_edges[0].node_to.name)
    else:
        print("Réponse (par force brute) est un flow de " + str(min_flow) + " en enlevant un des Edges suivants: ")
        for edge in min_edges:
            print(" -> " + edge.node_from.name + " à " + edge.node_to.name + " (" + str(edge.get_flow()) + "/" + str(edge.get_max_capacity()) + ")")
    
    min_edges[0].reduce_capacity_to_zero()

fg.recalculate_flow()
fg.draw(file_name.replace(".csv", ""))