
import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()
G.add_edges_from([(1,2),(2,3),(3,1),(1,4)]) #define G
fixed_positions = {1:(0,0), 2:(0,1), 3:(1,0), 4:(1,1)}
fixed_nodes = fixed_positions.keys()
pos = nx.spring_layout(G,pos=fixed_positions, fixed = fixed_nodes)
nx.draw_networkx(G,pos)

nx.draw(G,pos,node_size=0,alpha=0.4,edge_color='r',font_size=16)
plt.savefig("network.png")
plt.show()

