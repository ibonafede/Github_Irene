#python2 classifier.py

from __future__ import division
import pandas as pd
import numpy as np
import os, sys, argparse, csv
#import wmd 
import math
import pickle as pck

#from graph_tool.all import *
#from numpy.random import *
from numpy.linalg import norm
import numpy.linalg as la 
import scipy.cluster.vq as vq
import graph_tool.all as gt
import networkx as nx
import collections
import matplotlib.pyplot as plt
import matplotlib


def split_by_pattern(sequence, pattern):
    """ Split sequence by different char in pattern
    """
    #print(sequence)
    #print(pattern)
    if len(sequence) != len(pattern):
        print(sequence)
        print(pattern)
    sequence_splitted = sequence[0]
    for i in range(1, len(pattern)):
        if pattern[i] != pattern[i-1]:
            sequence_splitted += " "
        sequence_splitted += sequence[i]
    return sequence_splitted


def preprocessing(text_file,colsequence,colpattern):
    print(text_file.head())
    examples = text_file.apply(lambda x: split_by_pattern(x[colsequence],x[colpattern]),axis=1).values
    sentences=list()
    for ex in examples:
        sentences.append(ex.lower().split())
    # Remove stopwords.
    stop_words = ['a','c','g','t','u','n']
    filtered_words=list()
    for n,s in enumerate(sentences):
        filtered_words.append([word for word in s if word not in stop_words])
    return filtered_words,text_file,sentences


def count(sentences):
    tot=[]
    for s in sentences:
        l=[]
        for c in s:
            #print(c)
            #single motif
            #char=list(set(c))[0]
            #single word
            char=c
            #print(char)
            l.append((char,len(c)))
        tot.append(l)
    return tot

'''
def list_of_tuple2matrix(tot,col):
    l=list()
    for n in range(len(tot)):
        el=pd.DataFrame(tot[n])[col]
        l.append(el)
    df=pd.DataFrame(l).T
    df.columns=range(0,len(tot))
    return df'''





def buildGraph_seq(words,qsentences):
    dcounts_list=list()
    DFtot_l=list()
    DFmap_l=list()
    g = gt.Graph()
    #weight=list()
    #stop_words = ['z','a','q','t']
    #filtered_words=list()
    #for n,s in enumerate(words):
    #    filtered_words.append([word for word in s if word not in stop_words])
    for n,w in enumerate(words):
        #print(n,w)
        dftot=pd.DataFrame()
        df_map=pd.DataFrame()
        string_df=pd.DataFrame()
        edge_list=list()
        #d = collections.OrderedDict()
        #dcounts=pd.DataFrame()
        #dcounts=pd.DataFrame(w, columns=['motif', 'n'])
        #dcounts['weight']=len(k) for k in w[0]]
        #dcounts_list.append(dcounts)
        for i in range(0,len(w)):
            #print(i,n,w[i])
            if i==0:
                node1='start'
                node2=w[i]
            #print(node1,node2)
                edge_list.append((node1,node2,n))
                node3=w[i+1]
                edge_list.append((node2,node3,n))
            if i>0:
                node1=w[i]
                try:
                    node2=w[i+1]
                    print(node1,node2)
                    edge_list.append((node1,node2,n))
                except:
                    #print('end:',i)
                    edge_list.append((node1,'end',n))
        string_df=pd.DataFrame(edge_list)
        string_df.columns=['source','target','nseq']
        #string_df=string_df[string_df['nseq']==0].reset_index(drop=True)
        vertices=string_df['source'].astype("str").append(string_df['target'].astype("str")).reset_index(drop=True)
        vertices_encoded=vertices.astype('category')
        vertices_encoded=vertices_encoded.cat.codes
        vertices_encoded=vertices_encoded.reset_index(drop=True)
        df_map['source_0']=string_df['source']
        df_map['target_0']=string_df['target']
        df_map['nseq']=string_df['nseq']
        df_map['source']=vertices_encoded[0:int(len(vertices_encoded)/2)]
        df_map['target']=vertices_encoded[int(len(vertices_encoded)/2):int(len(vertices_encoded))].reset_index(drop=True) #groups
        df_map["vertex_order0"]=range(0,df_map.shape[0])
        df_map["vertex_order1"]=range(1,df_map.shape[0]+1)
        df_map["target_index"]=df_map.apply(lambda x:get_position(qsentences,x['target_0'],x['nseq']),axis=1)
        print(df_map.head())
        #add the property to vertex object
        tmp=df_map[['target_0','target',"vertex_order0","vertex_order1","target_index",'nseq']].reset_index(drop=True)
        tmp=tmp.rename(columns={'target_0':'source_0','target':'source'})
        dftot=df_map[['source_0','source',"vertex_order0","vertex_order1","target_index",'nseq']].append(tmp).reset_index(drop=True)
        print(dftot)
        #vsum=sum_degree(df_map)
        #df_map["in_degree"]=vsum
        DFtot_l.append(dftot)
        DFmap_l.append(df_map)
    DFtot=pd.concat(DFtot_l).reset_index(drop=True)
    DFmap=pd.concat(DFmap_l).reset_index(drop=True)
    g = gt.Graph(directed = True)
    #c=zip(list(DFmap['source']),list(DFmap['target']))
    c=zip(list(DFmap['source']),list(DFmap['target']))
    #e_weight=g.new_edge_property('float')
    #v_names=g.add_edge_list(c) 
    g.add_edge_list(c)
    return g,DFmap,DFtot,dcounts_list


def get_position(qsentences,el,nseq):
    word=qsentences[nseq]
    #get position of the word in the sequence
    index=[i for i,n in enumerate(word) if n==el]
    return index


def set_vertex_name(g,dftot,opt,qsentences):
    vertices=list()
    #g.vertex_properties["name"]=v_prop
    v_prop = g.new_vertex_property("string")
    for v in g.vertices():
        if opt=='motif':
            #set motif as vertex name
            #print(list(set(dftot.ix[dftot['source']==v,'source_0']))[0])
            vertices.append(list(set(dftot.ix[dftot['source']==v,'source_0']))[0])
            v_prop[v] = list(set(dftot.ix[dftot['source']==v,'source_0']))[0]
        if opt=='nseq':
            list1=list(set(dftot.ix[dftot['source']==v,'nseq']))
            print(list1)
            if len(list1)<3:
                el=''.join(str(e) for e in list1)
                print(el)
                v_prop[v] = el
                vertices.append(el)
            if len(list1)>2:
                el='morethan3'
                print(el)
                v_prop[v] = el
                vertices.append(el)
        #vprop=g.vertex_index
        #print(list(vprop))
        #for e in g.edges():
        """vertices.append(g.edge_index[e])
        print(g.edge_index[e])
        v_prop[v]=g.edge_index[e]"""
        #list1=list(set(dftot.ix[dftot['source']==v,'vertex_order0']))
            #vertices.append(list(set(dftot.ix[dftot['source']==v,'nseq'])))
        #el=''.join(str(e) for e in list1)
        #print(list1,el)
        #v_prop[v] = el           
    return v_prop,vertices


def set_edges_weight(g):
    dfE=pd.DataFrame(g.get_edges())
    dfE.columns=['source','target','index']
    edge_weights = g.new_edge_property('double')
    g.properties[("e","weight")] = edge_weights
    for e in g.edges():
        edge_weights[e]=g.edge_index[e]
    return edge_weights


def set_vertices_weight(g):
    vertex_weights = g.new_vertex_property('double')
    g.properties[("v","weight")] = vertex_weights
    for v in g.vertices():
        vertex_weights[v]=g.get_in_degrees([v])[0]
    return vertex_weights


def set_vertex_color(g,labels,dftot):
    #vertices=list()
    c = g.new_vertex_property("string")
    for v in g.vertices():
        print(v)
        #set motif as vertex name
        print(list(set(dftot.ix[dftot['source']==v,'nseq'])))
        list1=list(set(dftot.ix[dftot['source']==v,'nseq']))
        if len(list1)==1:
        #vertices.append(list(set(dftot.ix[dftot['source']==v,'nseq'])))
            #el=''.join(str(e) for e in list1)
            c[v] = 0
        if len(list1)==2:
        #vertices.append(list(set(dftot.ix[dftot['source']==v,'nseq'])))
            el=''.join(str(e) for e in list1)
            c[v] = 1
        if len(list1)>2:
            el='0.5'
            c[v] = el
    #c.a=np.array(labels)
    #g.vertex_properties['plot_color'] = c
    return c


def set_pos(g):
    #coordinates are the same for vertices with same in-degree(number of ripetition)
    pos = g.new_vertex_property("vector<double>")
    g.properties[("v","pos")] = pos
    #coord=circle()
    for v in g.vertices():
        r=np.sqrt(int(v))
        theta=g.get_in_degrees([v])[0]
        x1=r*np.cos(theta)
        x2=r*np.sin(theta)
        coord=(x1,x2)
        pos[v]=coord
    return pos

def set_pos2():
    e = gt.circular_graph(71, 4)
    pos = gt.sfdp_layout(g, cooling_step=0.95)
    #e=gt.lattice([10,10])
    #e = gt.lattice([10,20], periodic=True)
    #pos = gt.sfdp_layout(g, cooling_step=0.95, epsilon=1e-2)
    return pos

def circle():
    theta=np.linspace(0,2*np.pi,71)
    r=np.sqrt(100)
    x1=r*np.cos(theta)
    x2=r*np.sin(theta)
    coord=zip(x1,x2)
    return coord

def draw_cond(DFmap,n,fs):
    df_map=DFmap[DFmap['nseq']==int(n)].reset_index(drop=True)
    g = gt.Graph(directed = True)
    c=zip(list(df_map['source']),list(df_map['target']))
    #v_names=g.add_edge_list(c) 
    g.add_edge_list(c)
    #v_prop.a=g.get_out_degrees(g.get_vertices())
    #vsum = gt.incident_edges_op(g, "in", "sum", g.edge_index)
    #print(vsum.a)
    edge_weights=set_edges_weight(g)
    vertex_weight=set_vertices_weight(g)
    #pos = gt.radial_tree_layout(g,g.vertex(0), node_weight=vertex_weight,rel_order=g.vertex_index,rel_order_leaf=True, weighted=True, node_weight=edge_weights, r=1.0))  
    #ebet=gt.betweenness(g)[1]
    #print(ebet)
    pos=set_pos(g)
    gt.graph_draw(g,pos=pos,vweight=vertex_weight, vertex_fill_color=vertex_weight,vertex_size=vertex_weight,vertex_text=g.vertex_index,vertex_font_size=int(fs),output=str(n)+'.single_seq.pdf')
    #pos = gt.radial_tree_layout(g,g.vertex(0), node_weight=v_prop)
    '''
    #state = gt.BlockState(g, B=5, deg_corr=True)
    #state.print_summary()
    #gt.mcmc_equilibrate(state, wait=1000)
    #b = state.get_blocks()
    #pos=gt.graph_draw(g, pos=gt.radial_tree_layout(g,g.vertex(0), node_weight=vertex_weight),vertex_fill_color=b, vertex_shape=b,vertex_size=vertex_weight,vertex_text=g.vertex_index,vertex_font_size=int(fs),output="prova.pdf")
    pos=gt.graph_draw(g, pos=pos, node_weight=vertex_weight,vertex_fill_color=b, vertex_shape=b,vertex_size=vertex_weight,vertex_text=g.vertex_index,vertex_font_size=int(fs),output="prova.pdf")
    position = g.new_vertex_property("vector<double>")
    g.properties[("v","pos")] = position
    for v in g.vertices():
        position[v]=pos[v]
    bg, bb, vcount, ecount, avp,aep = gt.condensation_graph(g, b, avprops=[g.vp["pos"]])
    pos=avp[0]
    vertex_pos = bg.new_vertex_property("vector<double>")
    bg.properties[("v","pos")] = vertex_pos
    vertex_pos=list(pos)
    vertex_weight=set_vertices_weight(bg)
    edge_weights=set_edges_weight(bg)
    gt.graph_draw(bg, pos=pos, vertex_fill_color=bb,vertex_size=vertex_weight,vertex_text=bg.vertex_index,output="cond.pdf")
    '''
    return g,state,b




def draw_graph_seq(g,min_filter,max_filter,dftot,outname,fs,words):
    #isolate vertices with degree>n
    """x=min_filter
    y=max_filter
    u = gt.GraphView(g, vfilt=lambda v: v.out_degree() > x )
    vprop,vertices=set_vertex_name(u,dftot)
    print(vertices)
    pos_u = gt.radial_tree_layout(u, u.vertex(0))
    get_positions(u,pos_u)
    #pos_u =gt.arf_layout(g, max_iter=0)
    print("filtering by out_degree>",str(x))
    tree = gt.min_spanning_tree(u)
    gt.graph_draw(u, efilt=tree,vcmap=matplotlib.cm.gist_heat,vertex_text=vprop,vertex_font_size=int(fs),output=outname+"_filtered_tree.pdf")
    print("get position..")"""
    pos = gt.sfdp_layout(g)
    pos = gt.radial_tree_layout(g, g.vertex(0))
    #get_positions(g,pos)
    #pos=set_pos(g)
    print("get betweenness..")
    vp, ep = gt.betweenness(g)
    #vertex_betweenness : A vertex property map with the vertex betweenness values.
    #edge_betweenness : An edge property map with the edge betweenness values.
    #g.vertex_properties["bw"] = vp
    #get hits hub
    ee, x, y = gt.hits(g)
    #for v in g.vertices():
    #    print(g.vp.bw[v],v)
    edge_pen_width=ep
    #color by seq
    print(dftot.head())
    labels=DFtot.nseq
    c=set_vertex_color(g,labels,DFtot)
    print("set vertices name..")
    vprop,vertices=set_vertex_name(g,DFtot,'nseq',qsentences)
    edge_weights=set_edges_weight(g)
    state = gt.minimize_nested_blockmodel_dl(g, deg_corr=True)
    
    vertex_weight=set_vertices_weight(g)
    #gt.graph_draw(g, pos=pos, vertex_fill_color=x,vorder=x,vcmap=matplotlib.cm.gist_heat,vertex_size=gt.prop_to_size(x, mi=5, ma=15), vertex_text=g.vertex_index,vertex_font_size=int(fs),output=outname+".order.pdf")
    pos = gt.sfdp_layout(g)
    pos = gt.radial_tree_layout(g, g.vertex(0))
    vprop,vertices=set_vertex_name(g,DFtot,'motif',qsentences)  
    #vertex_text=vprop
    gt.graph_draw(g,pos=pos, vertex_fill_color=vertex_weight,vertex_size=vertex_weight,vertex_font_size=int(fs),output='confernoname.pdf')
    gt.graph_draw(g, pos=pos, vertex_fill_color=x,vorder=x,vcmap=matplotlib.cm.gist_heat,vertex_size=gt.prop_to_size(x, mi=5, ma=15),vertex_font_size=int(fs),output=outname+".nonameorder.pdf")
    gt.draw_hierarchy(state, vertex_fill_color=x,vorder=x,vcmap=matplotlib.cm.gist_heat,vertex_size=gt.prop_to_size(x, mi=5, ma=15),vertex_font_size=int(fs),output=outname+".hierwithoutname.pdf")  
    #gt.draw_hierarchy(state, vertex_fill_color=x,vorder=x,vcmap=matplotlib.cm.gist_heat,vertex_size=gt.prop_to_size(x, mi=5, ma=15), vertex_text=vprop,vertex_font_size=int(fs),output=outname+".hier.pdf")
    gt.graph_draw(g, pos=pos, vertex_fill_color=x,vorder=x,vcmap=matplotlib.cm.gist_heat,vertex_font_size=(int(fs)-3),output=outname+".nomotifs.pdf")
    #color vertices by sequences
    gt.graph_draw(g, pos=pos, vertex_fill_color=c,vcmap=matplotlib.cm.gist_heat,vertex_font_size=(int(fs)-3),output=outname+".motifsperseqnoname.pdf")
    #color vertices by Jaccard sim
    #color by jaccard similarity
    #s = gt.vertex_similarity(g, "jaccard")
    #colorJ= g.new_vp("double")
    #colorJ.a = s[0].a
    #gt.graph_draw(g, pos=pos, vertex_fill_color=colorJ,vcmap=matplotlib.cm.gist_heat, vertex_text=vprop,vertex_font_size=(int(fs)-3),output=outname+".motifsperJsim.pdf")
    gt.graph_draw(g,pos=pos,vertex_text=vprop, vertex_fill_color=vertex_weight,vertex_size=vertex_weight,vertex_font_size=int(fs),output='confername.pdf')
    gt.graph_draw(g, pos=pos,vertex_text=vprop, vertex_fill_color=x,vorder=x,vcmap=matplotlib.cm.gist_heat,vertex_size=gt.prop_to_size(x, mi=5, ma=15),vertex_font_size=int(fs),output=outname+".nameorder.pdf")
    gt.draw_hierarchy(state,vertex_text=vprop, vertex_fill_color=x,vorder=x,vcmap=matplotlib.cm.gist_heat,vertex_size=gt.prop_to_size(x, mi=5, ma=15),vertex_font_size=int(fs),output=outname+".hiername.pdf")  
    #gt.draw_hierarchy(state, vertex_fill_color=x,vorder=x,vcmap=matplotlib.cm.gist_heat,vertex_size=gt.prop_to_size(x, mi=5, ma=15), vertex_text=vprop,vertex_font_size=int(fs),output=outname+".hier.pdf")
    gt.graph_draw(g, pos=pos, vertex_text=vprop,vertex_fill_color=x,vorder=x,vcmap=matplotlib.cm.gist_heat,vertex_font_size=(int(fs)-3),output=outname+".namemotifs.pdf")
    #color vertices by sequences
    gt.graph_draw(g, pos=pos,vertex_text=vprop, vertex_fill_color=c,vcmap=matplotlib.cm.gist_heat,vertex_font_size=(int(fs)-3),output=outname+".motifsperseqnoname.pdf")
    #color vertices by Jaccard sim

    #vprop,vertices=set_vertex_name(g,dftot,'motif')
    #gt.graph_draw(g, pos=pos,eweight=edge_weights ,vertex_fill_color=vprop,vorder=vprop,vcmap=matplotlib.cm.gist_heat,vertex_size=gt.prop_to_size(x, mi=5, ma=15),vertex_text=vprop,vertex_font_size=(int(fs)-3),output=outname+".motifs.pdf")
#gt.graph_draw(g, pos=pos, vertex_fill_color=vp,vorder=vp,vcmap=matplotlib.cm.gist_heat,output=outname+".pdf")




def vertex_percolation(g):
    #Compute the size of the largest component as vertices are (virtually) removed from the graph.
    vertices = sorted([v for v in g.vertices()], key=lambda v: v.out_degree())
    sizes, comp = gt.vertex_percolation(g, vertices)
    np.random.shuffle(vertices)
    sizes2, comp = gt.vertex_percolation(g, vertices)
    plt.figure()
    plt.plot(sizes, label="Targeted")
    plt.plot(sizes2, label="Random")
    plt.xlabel("Vertices remaining")
    plt.ylabel("Size of largest component")
    plt.legend(loc="lower right")
    plt.savefig("vertex-percolation.pdf")


def kcore_dec(g):
    g = gt.GraphView(g, vfilt=gt.label_largest_component(g))
    kcore = gt.kcore_decomposition(g)
    gt.graph_draw(g, pos=g.vp["pos"], vertex_fill_color=kcore, vertex_text=kcore, output="netsci-kcore.pdf")


def laplacian(g,DFtot):
    #return Compressed Sparse Row matrix
    L = gt.laplacian(g,normalized=True,weight=None, index=None) #weight Edge property map with the edge weights
    #L spectral decomposition
    l, U = la.eigh(L.todense())
    #Fiedler vector
    f = U[:,1]
    #use it for clustering, label based on sign of f
    labels = np.ravel(np.sign(f))
    fig = plt.figure()
    vprop,vertices=set_vertex_name(g,DFtot,'nseq',qsentences)
    c=set_vertex_color(g,labels,DFtot)
    #pos = gt.radial_tree_layout(g, g.vertex(0))
    pos=set_pos(g)
    vp, ep = gt.betweenness(g)
    edge_pen_width=ep
    gt.graph_draw(g, pos=pos, vertex_fill_color=c,vorder=vp,vcmap=matplotlib.cm.gist_heat, vertex_text=vprop,vertex_font_size=int(2),output=outname+".pdf")
'''#nx.draw_networkx_nodes(G, coord,node_size=45, node_color=labels)
#This should then produce a result similar to what we can see in Fig. 2(a). In order to cluster G into k > 2 sub-graphs, we next
    #apply k-means clustering to the rows of the matrix Uk of the k smallest eigenvectors (where we begin counting from the second smallest one). That is, we simply execute
    k = 3 
    means, labels = vq.kmeans2(U[:,1:k], k)
    #Solve an ordinary or generalized eigenvalue problem of a square matrix.
    #Find eigenvalues w and right or left eigenvectors of a general matrix:
    ew, ev = scipy.linalg.eig(L.todense())
    print('L,eigenvalues,eigenvectors:',L.todense(),ev,ew)
    plt.figure(figsize=(8, 2))
    plt.scatter(np.real(ew), np.imag(ew), c=np.sqrt(abs(ew)), linewidths=0, alpha=0.6)
    plt.xlabel(r"$\operatorname{Re}(\lambda)$")
    plt.ylabel(r"$\operatorname{Im}(\lambda)$")
    plt.tight_layout()
    plt.savefig("laplacian-spectrum.pdf")'''





"""
#offscreen = sys.argv[1] == "offscreen" if len(sys.argv) > 1 else False
offscreen='offscreen'
# We will use the network of network scientists, and filter out the largest
# component
g = GraphView(g, vfilt=label_largest_component(g), directed=True)
g = Graph(g, prune=True) #if prune is set to True, and g is specified, only the filtered graph will be copied, and the new graph object will not be filtered
pos = gt.radial_tree_layout(g, g.vertex(0))
#pos = g.vp["pos"]  # layout positions

ecolor = g.new_edge_property("vector<double>")
for e in g.edges():
    ecolor[e] = [0.6, 0.6, 0.6, 1]
vp, ep = gt.betweenness(g)
edge_pen_width=ep
print("set vertices name..")
vprop,vertices=set_vertex_name(g,dftot)
win = GraphWindow(g, pos, geometry=(500, 400), vertex_fill_color=vp,vorder=vp,vcmap=matplotlib.cm.gist_heat, vertex_text=vprop,vertex_font_size=int(fs))

orange = [0.807843137254902, 0.3607843137254902, 0.0, 1.0]
old_src = None
count = 0
def update_bfs(widget, event):
    global old_src, g, count, win
    src = widget.picked
    if src is None:
        return True
    if isinstance(src, PropertyMap):
        src = [v for v in g.vertices() if src[v]]
        if len(src) == 0:
            return True
        src = src[0]
    if src == old_src:
        return True
    old_src = src
    pred = shortest_distance(g, src, max_dist=3, pred_map=True)[1]
    for e in g.edges():
        ecolor[e] = [0.6, 0.6, 0.6, 1]
    #for v in g.vertices():
    #    vcolor[v] = [0.6, 0.6, 0.6, 1]
    for v in g.vertices():
        w = g.vertex(pred[v])
        if w < g.num_vertices():
            e = g.edge(w, v)
            if e is not None:
                ecolor[e] = orange
                #vcolor[v] = vcolor[w] = orange
    widget.regenerate_surface()
    widget.queue_draw()
    if offscreen:
        window = widget.get_window()
        pixbuf = Gdk.pixbuf_get_from_window(window, 0, 0, 500, 400)
        pixbuf.savev(r'./frames/bfs%06d.pdf' % count, 'png', [], [])
        count += 1

# Bind the function above as a montion notify handler
win.graph.connect("motion_notify_event", update_bfs)

# We will give the user the ability to stop the program by closing the window.
win.connect("delete_event", Gtk.main_quit)

# Actually show the window, and start the main loop.
win.show_all()
Gtk.main()"""

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-input') # inputfile path
    parser.add_argument('-output') # outputfile path
    parser.add_argument('-firstColumn') # opt - e.g. label or id
    parser.add_argument('-firstColumnPrefix') # opt - add prefix
    parser.add_argument('-mode') # run_similarities or graph_sequences
    parser.add_argument('-annotation') # opt , annotationfile path
    parser.add_argument('-min_degree_filter',type=int,default=4) # opt filter by min degree 
    parser.add_argument('-max_degree_filter',type=int,default=100) # opt filter by min degree 
    parser.add_argument('-species') # HS or MU
    args = parser.parse_args()
    #filename=args.input
    #outname=args.output
    #mode=args.mode
    filename1="/home/irene/irene/work/graph-tool/scripts/dataset_luglio.cell_MU_HS.m.tsv"
    outname='presentazione'
    print("Script runs with these arguments: ", args)
    #filename1="dataset_UNSUPERVISED_2806_complete.m.tsv" 
    #args.input='dataset_UNSUPERVISED_2806_complete.tsv'
    text_file=pd.read_table(filename1)
    #text_file=pd.read_table(args.input)
    text_file=text_file.dropna().reset_index(drop=True)
    filtered_words,text_file,sentences=preprocessing(text_file,'seqlist','qbearlist')
    qfiltered_words,text_file,qsentences=preprocessing(text_file,'qbearlist','qbearlist')
    print(qsentences[0])
    words=count(qsentences[0:1])
    g,DFmap,DFtot,dcounts_list=buildGraph_seq(words,qsentences)
    draw_graph_seq(g,1,5,DFtot,'provaclassifier',4,words)
