import streamlit as st
import networkx as nx
from networkx.readwrite import json_graph

import numpy as np
import matplotlib.pyplot as plt

import json
import os
import csv
import sys

#define css for different classes 
st.markdown("""
    <style>
    .maintitle {
        letter-spacing: 1px;
        color: #000080;
        font-size: 45px;
        font-family: "Lucida Grande", Verdana, Helvetica, Arial, sans-serif;
        font-weight: 100;
        
    }
    .info {
        
        letter-spacing: 1px;
        color: #000080;
        font-size: 15px;
        font-family: "Lucida Grande", Verdana, Helvetica, Arial, sans-serif;
        font-weight: 100;
        
    }    
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="maintitle">Signatories Network Example</p>', unsafe_allow_html=True) 

twenty_distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0',\
                          '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',\
                          '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff',\
                          '#000000']

#Agreement-actor matrix
A = []
A.append([1,1,0])
A.append([0,0,0])
A.append([1,1,0])
A.append([1,0,1])

A = np.array(A)

V = np.matmul(A.T,A)
print(V)
print()

W = np.matmul(A,A.T)
#print(W)

# Read the CSVs
with open('./data/node_table.csv', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f)
    # Get the header row
    nodes_header = next(reader)
    # Put the remaining rows into a list of lists
    nodes_data = [row for row in reader]
    
with open('./data/links_table.csv', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f)
    # Get the header row
    links_header = next(reader)
    # Put the remaining rows into a list of lists
    links_data = [row for row in reader]

with open('./data/agreements_dict.json') as f:
    agreements_dict = json.load(f)
    
# Build a vertices dictionary with node_id as key
vertices_dict = {row[nodes_header.index('node_id')]:row for row in nodes_data}

# Collect all vertex types
vertex_types = []
for k,v in vertices_dict.items():
    type_ = v[nodes_header.index('type')]
    if len(type_) == 0:
        type_ = 'AGT'
    vertex_types.append(type_)
vertex_types = sorted(list(set(vertex_types)))
#print(vertex_types)

# Build a colour map for types (empty type is annoying)
color_map = {type_:twenty_distinct_colors[i] for i,type_ in enumerate(vertex_types)}

#get list of unique PP to add to selector
ppname_index = links_header.index('PPName')
unique_ppnames = set()
for row in links_data:
    unique_ppnames.add(row[ppname_index])
    
unique_ppnames_list = list(unique_ppnames)

#show selectbox for PP options to select
pp_selection=st.selectbox("Select Peace Process", unique_ppnames_list, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose a Peace Process", disabled=False, label_visibility="visible")


# Get section set counts for PP agreements
with open('./data/counts_dict.json') as f:
    section_counts_dict = json.load(f)
    
    
edges = [row for row in links_data if row[links_header.index('PPName')].strip()==pp_selection]

# Collect from_node_id values
from_vertices = list(set([row[links_header.index('from_node_id')] for row in edges]))

# Collect to_node_id values
to_vertices = list(set([row[links_header.index('to_node_id')] for row in edges]))

all_vertices = []
all_vertices.extend(from_vertices)
all_vertices.extend(to_vertices)
all_vertices = list(set(all_vertices))

# Count each type for peace process
counts = []
for vertex_type in vertex_types:
    counts.append((vertex_type,len([v for v in all_vertices if v.split('_')[0]==vertex_type])))
counts = sorted(counts,key=lambda t:t[1],reverse=True)

# Build an edge dictionary with agreement as key and list of actors as value
edge_dict = {}
for row in edges:
    if row[5] in edge_dict:
        edge_dict[row[5]].append(row[12])
    else:
        edge_dict[row[5]] = [row[12]]

#Build and plot an undirected multigraph
graph = nx.MultiGraph()

vertices = []
vertices.extend(from_vertices)
vertices.extend(to_vertices)
graph.add_nodes_from(vertices)
for row in edges:
    from_vertex = row[links_header.index('from_node_id')]
    to_vertex = row[links_header.index('to_node_id')]
    graph.add_edge(from_vertex,to_vertex,weight=1)
    
nr_edges=len(graph.edges)    
st.write("There are", nr_edges, "edges in this graph." )

node_colors = [color_map[v.split('_')[0]] for v in vertices]

f = plt.figure(figsize=(16,16))
pos = nx.spring_layout(graph) 
nx.draw_networkx(graph,pos,node_color=node_colors,font_size='8',alpha=0.8)
plt.grid(False)
#plt.show()
st.pyplot(f)

#Query vertices using depth-first search

#radio button to select operator type

st.sidebar.write(" # Query the network")

operator=["AND", "OR"]
select_operator=st.sidebar.radio("Select operator", operator, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, horizontal=False, captions=None, label_visibility="visible")

#depth=1

depth=st.sidebar.slider("Select depth", min_value=1, max_value=10, value=1, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

#get list of unique nodes to add to selector
node_index = links_header.index('from_node_name')
unique_nodes = set()
for row in links_data:
    unique_nodes.add(row[node_index])
    
unique_from_node_list = list(unique_nodes)
#show selectbox for PP options to select
node_selection=st.sidebar.multiselect("Select Node", unique_from_node_list, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder="Choose a Node", disabled=False, label_visibility="visible")

tree_list = []
for v in node_selection:
    tree_list.append(nx.dfs_tree(graph, source=v, depth_limit=depth))

found_vertices = set(tree_list[0].nodes) 
for tree in tree_list:
    if operator == 'AND':
        found_vertices = found_vertices.intersection(set(tree.nodes))
    elif operator =="OR":
        found_vertices = found_vertices.union(set(tree.nodes))
    
found_vertices = list(found_vertices)    
found_vertices.extend(node_selection)
#print(found_vertices)


found_edges = []
for tree in tree_list:
    for e in tree.edges:
        if e[0] in found_vertices and e[1] in found_vertices:
            found_edges.append(e)
            
#print('Found edges:',len(found_edges))

found_graph = nx.MultiGraph()
found_graph.add_nodes_from(found_vertices)
for e in found_edges:
    found_graph.add_edge(e[0],e[1],weight=1)
    
node_colors = [color_map[v.split('_')[0]] for v in found_graph.nodes()]

f1 = plt.figure(figsize=(16,16))
pos = nx.spring_layout(graph) 
nx.draw_networkx(found_graph,pos,node_color=node_colors,font_size='8',alpha=0.8)
plt.grid(False)
st.pyplot(f1)
