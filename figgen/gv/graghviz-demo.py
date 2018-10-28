import pygraphviz as pgv

#https://www.graphviz.org/doc/info/attrs.html

A=pgv.AGraph()

A.add_edge(1,2)
A.add_edge(2,3)
A.add_edge(1,3)

print(A.string()) # print to screen
print("Wrote simple.dot")
A.write('simple.dot') # write to simple.dot

B=pgv.AGraph('simple.dot') # create a new graph from file
B.layout() # layout with default (neato)
B.draw('simple.png') # draw png
print("Wrote simple.png")


A=pgv.AGraph(directed=True)

#A.node_attr['style']='filled'
#A.node_attr['shape']='circle'
    
A.add_node(1,label='x_t',fillcolor='red',pos="0,0!",style='filled') 
A.add_node(2,color='blue',pos="0,1!",shape='square')
A.add_node(3,color='blue',pos="1,0!",width=0.2, style='bold')
A.add_node(4,color='yellow', pos="1,1!", shape='diamond')

A.add_edge(1,2,color='green')
A.add_edge(2,3,style='dotted')
A.add_edge(2,4,"foo")
A.add_edge(1,3, arrowhead='none')

A.graph_attr['epsilon']='0.001'
A.layout()
#print(A.string()) # print dot file to standard output
A.draw('foo.pdf') # write to file