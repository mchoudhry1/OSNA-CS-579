"""
Cluster data.
"""
import networkx as nx
import matplotlib.pyplot as plt

def read_graph():
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    Done for you.
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist('names.txt', delimiter=':')

def partition_girvan_newman(G, depth=0):
    if G.order() == 1:
        return [G.nodes()]
    
    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)

        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]

    components = [c for c in nx.connected_component_subgraphs(G)]
    indent = '   ' * depth  
    while len(components) == 1:
        edge_to_remove = find_best_edge(G)
        G.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(G)]
    result = [c for c in components]
    return result

def write_data():
    file = open("cluster_answers.txt","w")
    graph = read_graph()
    graph1 = graph.copy()
    file.write('Original graph has %d nodes and %d edges' % 
          (graph.order(), graph.number_of_edges()))
    clusters = partition_girvan_newman(graph1, 3)
    file.write('\nCommunities discovered: %d' % len(clusters))
    file.write('\ncluster 1 has %d nodes and cluster 2 has %d nodes' %
         (clusters[0].order(), clusters[1].order()))
    file.write('\n')
    file.write('names in cluster 1 nodes')
    file.write('\n')
    file.write(str(clusters[0].nodes()))
    file.write('\n')
    file.write('names in cluster 2 nodes')
    file.write('\n')
    file.write(str(clusters[1].nodes()))
    file.write('\n')
    a = graph.order()
    b = len(clusters)
    average = float(a/b)
    file.write('\n')
    file.write('Average number of users per community: %.2f' %(average))
    file.write('\n')
    file.close()

def draw_network(graph,filename):
#     list1=[]
    plt.figure(figsize=(22,14))
    plt.axis('off')
#     for u in users:
#         lis=u['screen_name']
#         list1.append(lis)
#     lab={la: la for la in list1}
    nx.networkx.draw_networkx(graph, node_color='red', alpha =.5,width = .5,node_size = 100, edge_color ='black', with_labels=False)
    plt.savefig(filename)
    # plt.show()

def main():
    graph = read_graph()
    graph1 = graph.copy()
    print('Original graph has %d nodes and %d edges' % 
          (graph.order(), graph.number_of_edges()))
    clusters = partition_girvan_newman(graph1, 3)
    print('Communities discovered: %d' % len(clusters)) 
    print('cluster 1 has %d nodes and cluster 2 has %d nodes' %
         (clusters[0].order(), clusters[1].order()))
    a = graph.order()
    b = len(clusters)
    average = float(a/b)
    print('Average number of users per community: %.2f' %(average))
#     print('\n')
#     print('names in cluster 1 nodes')
#     print(clusters[0].nodes())
#     print('\n')
#     print('names in cluster 2 nodes')
#     print(clusters[1].nodes())
    write_data()
    print("Answers written in cluster_answers.txt")
    print("Orinal Graph saved in cluster1.png")
    draw_network(graph,"cluster1.png")
    print("Graph after applying community Detection algorithm saved in cluster2.png ")
    draw_network(graph1,"cluster2.png")

if __name__ == "__main__":
    main()