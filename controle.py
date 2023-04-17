# Algorithm and logic:
from collections import deque
from itertools import combinations
import heapq

# Reading the input and csv file:
import csv
import os
import sys
import time

# Drawing the png:
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
        return self.remaining_capacity > other.remaining_capacity
        
    def __str__(self):
        return "(" + self.node_from.name + " à " + self.node_to.name + ")"  

    def is_full(self):
        return self.remaining_capacity <= 0
    
    def has_flow(self):
        return (not self.is_blocked) and (self.remaining_capacity < self._max_capacity)
    
    def get_flow(self):
        if( self.is_blocked):
            return 0
        else:
            return self._max_capacity - self.remaining_capacity
    
    def get_remaining_capacity(self):
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
    
    def block(self):
        self.is_blocked = True
        self.remaining_capacity = 0

    def unblock(self):
        self.is_blocked = False
        
    
        

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
        return self.outgoing_edges_heap[-1]
    
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
            bottleneck_capacity = min(edge.get_remaining_capacity() for edge in augmenting_path)
            
            # Update the flow and residual graph
            for edge in augmenting_path:
                edge.reduce_capacity(bottleneck_capacity)
                
            self.flow += bottleneck_capacity
        return self.flow
    

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

    
    
    def get_best_edges_to_block_using_backtracking(self):
        
        # Noter qu'à l'initialisation de l'objet FlowNetwork, le flow initial est déjà calculé par Ford-Fulkerson et BFS.
        
        # Ci-dessous; en stockant les configurations optimales, on peut utiliser la même instance de FlowNetwork:
        #   - Ça nous permet d'éviter de copier les objets dans le heap et de devoir ajuster leurs pointers,
        #     comme serait le cas avec un arbre de recherche exhaustif et tuples.
        min_flow = self.flow
        min_edges = []
        
        # Amélioration: Vérification d'un cas trivial
        #
        #   - On vérifie si le nombre d'arcs entrant dans le puis ou le nombre d'arcs sortants de la source
        #     est égal ou inférieur au nombre d'arcs que l'on peut bloquer. Si c'est le cas, en une opération,
        #     nous aurions trouvé la solution. Sinon, le temps d'exécution ne sera pas du tout impacté, donc aucune 
        #     raison de pas l'inclure.
        #
        if(len(self.source_node.outgoing_edges_heap) <= self.num_edges_to_remove):
            return (0, self.source_node.outgoing_edges_heap)
        if(len(self.sink_node.ingoing_edges_heap) <= self.num_edges_to_remove):
            return (0, self.sink_node.ingoing_edges_heap)
        
        # Initialisation d'une matrice des combinaisons possibles d'arcs à retirer (max profondeur de k = num_edges_to_remove)
        edges_with_flow_heap = []
        for edge in self.edges:
            if(self.num_edges_to_remove > 1 or edge.has_flow()): heapq.heappush(edges_with_flow_heap, edge)

        # Améliorations: Ne regarder que les arêtes intéressantes, et en ordre d'intérêt
        #
        #   - Si on ne veut enlever qu'un arc, plutôt qu'évaluer tous les arcs, on ne regarde que ceux qui ont initialement du flow.
        #     Lorsque c'est le cas, le nombre de configurations du graphe à vérifier sont diminuées d'un ordre de grandeur, et sinon,
        #     ça ne coûte que m opérations booléeanes supplémentaires, coût négligable.
        #
        #   - On regarde les arcs en ordre descendant de leur flow initial, ce qui a comme but de de trouver la solution optimale
        #     plus vite. Nous permettrait des heuristiques et solutions approximatives rapides si la taille de m est trop grande.
        #     Concrètement par contre, sans heuristique, le temps d'éxecution reste idem car on va vérifier chaque configuration
        #     importante avec le retour-arrière.
        
        # Pour chaque 'noeud' ou configuration suivant dans l'arbre de recherche, c.a.d. pour chaque combinaison 
        # d'arêtes qui pourraientt être bloquées ensemble, on compare sa valeur (i.e. son flow max).

        # Parcourir toutes les k-combinaisions d'arcs, sans réevaluer les doublons:
        leaves = combinations(edges_with_flow_heap, self.num_edges_to_remove)
            
        # Amélioration: Parcours de l'arbre de recherche en commençant par les feuilles
        #
        #   - Habituellement, le parcours le l'arbre de recherche dans l'algorithme de retour-arrière se
        #     fait en PreOrder, c.a.d. on commence par la racine, puis on ajoute un élément en développant
        #     ses childrens récursivement.
        #           -> D'ailleurs, c'est de là que vient le nom de l'algorithme; si une solution non-optimale
        #              ou un maximum local est trouvée, l'algorithme remonte l'arbre vérifier d'autres noeuds.
        #
        #   - Or, dans ce cas-ci, nous ne parcourons que les solutions de l'arbre ayant k arcs retirés.
        #           -> Le but est que, pour notre mise en situation, nous avons *toujours* avantage à blocker plus
        #              d'arcs plutôt que moins, donc la solution sera trouvée plus tôt. Ça l'aurait été une autre
        #              histoire s'il y avait un coût associé à bloquer un arc.
        #           -> Vu que nous ne cherchons pas le minimum d'arcs à retirer, mais plutôt le flow minimal après
        #              retirer k arcs, alors nous n'avons aucune raison d'évaluer les profondeurs < k.
        #           -> De plus, cette amélioration permet d'éviter de garder une arborescence en mémoire et éviter les 
        #              appels récursifs, ce qui améliore la performance temporelle et de mémoire;
        #                   - Aucune copie ni reconstruction de graphe n'est faite, et le stockage en mémoire de minimums 
        #                     locaux de l'arbre de recherche n'est plus nécessaire avec cette approche. 
        #
        #   - On peut se le permettre car une instance de FlowNetwork peut aisément blocker plusieurs de ses arcs
        #     avant même de recalculer le flow maximal par Ford-Fulkerson. 
        #           -> Plus k est grand, moins il y aura de combinaisons à évaluer.
        #              Ça va venir diminuer la complexité de notre algorithme.
            
        # Parcourir toutes les combinaisons de k arcs à retirer, et faire le parcours:
        for edge_combination in leaves:
                    
            # On supprime toutes les arêtes, puis réevalue le flow en utilisant F-F:
            for edge in edge_combination:
                edge.block()
            self.recalculate_flow()
                    
            # Si le nouveau flow est nul, nous avons trouvé une solution optimale:
            if(self.flow == 0):
                return (0, edge_combination)
                
            # Si le nouveau flow est le minimum par rapport à cette profondeur de l'arbre:
            if(self.flow < min_flow):
                min_flow = self.flow
                min_edges = edge_combination
            
            # On débloque les arcs, ça va reset pour la prochaine itération/feuille:
            for edge in edge_combination:
                edge.unblock()
            
        # Retourner le meilleur minimum local, si aucune solution optimale (flow = 0) n'a été trouvée:
        return (min_flow, min_edges)
            

    
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



if __name__ == '__main__':

    # Vérification des arguments en ligne de commande:
    if(len(sys.argv) != 2):
        print("Mauvais input!\nCommande attendue: python3 controle.py ex1.csv\nVeuillez réessayer.\n")
        sys.exit(1)

    # Lecture du fichier et construction du graphe:
    file_name = sys.argv[1]
    G = FlowNetwork(file_name)
    print("\nMax initial flow is: " + str(G.flow))

    # Timer, start!
    time_started = time.time()
    
    # On roule l'algo:
    min_flow, min_edges = G.get_best_edges_to_block_using_backtracking()
    
    # Done!
    time_elapsed = time.time() - time_started
    
    if(len(min_edges) == 0): print("Réponse trouvée était de suprimer aucun arc. Erreur?")
    elif(len(min_edges) == 1): print("Réponse est un flow de flow de " + str(min_flow) + " en enlevant l'arc " + str(min_edges[0]))
    else: print("Réponse est un flow de " + str(min_flow) + " en enlevant les arcs suivants: " + (", ").join([str(edge) for edge in min_edges]))
    print(f'Temps d\'ex: {(time_elapsed * 1000):.3f} millisecondes\n')
    
    # Recalculer pour l'image (for the png):
    for edge in min_edges: edge.block()
    G.recalculate_flow()

    # Change working directory to the directory of the current python file:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Save the png:
    G.draw(file_name.replace(".csv", ""))

    # Save the txt:
    with open("resultats_" + file_name.replace(".csv", "") + ".txt","w+", encoding='utf-8') as f:
        f.write(("\n").join([str(edge) for edge in min_edges]))