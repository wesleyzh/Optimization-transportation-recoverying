"""
Procedure PS:

Search all the independent path between all pair of nodes

implemented by weili zhang, 2014-09-16

1) Set the lengths of all edges are 1
2) For i =1,2,...,n and all j =1,2,...,n, j!=i, compose a node pair of (i,j)
3) Calculate Upbound(i,j) = min(di,dj), the upper bound of the possible number of assageways between node pair (i,j). Set the index of path number, k=1
4) Find the shortest path from i to j by the popular Dijkstra algorithm. Label the path k.
5) Delet ll edges in the shortest path k
6) If k = Up(i,j) or there is no parth connecting the node pair (i,j) to be found, recover deleted edges and go to Setp 2) for the next node pair. Otherwise, go to Step3).
7) If all node pairs are done, output the result and stop.

Input:

V = nodes list
E = edge list
weight = the weight of each link, default is 1 for all links

Output:
L =  all the independent path between all pair of nodes
e.g., L = {(1,2):{1:1,2}{2:1,3,2}}
"""

from gurobipy import *
import copy
import networkx as nx
import random

def Dijkstra(nodes, arcs, distance, start,end):
	attributes = {}                           #Initializations
	Q = []                                    
	for i in nodes:                           #Unknown distance function from source to v and Previous node in optimal path from source
		attributes[i] = [float("infinity"),None]
		Q.append(i)                       #unvisited list
		
	attributes[start] = [0,0]
	
	Q.remove(start)
	u = start	

	while(len(Q)>1):
		
		alt = {}
		for i,j in arcs.select(u,"*"):
			if j in Q:   #only caculate the neighbors in Q
				alt[j] = attributes[u][0]+distance[u,j]
				if alt[j] < attributes[j][0]:
					attributes[j] = [alt[j],u]   #caculate the distance from start, previeous node
		if alt != {}:
			best = min(alt,key = alt.get)        #find the minimum distance and remove from Q
			u = best
			Q.remove(best)
			previous = copy.deepcopy(alt)
			
		else:
			pass
			
		if best == end:
			break
		
	
	#get the shortest path of Dijkstra
	try:
		temp = end
		path = []
		path.append(temp)
		while(temp!=start):
			temp = attributes[temp][1]
			path.append(temp)
	except:
		path = []
		
	return path		

  
	


def main(V, E, EmergencyNodes,length, resilience, ADTT):
	random.seed(10)
	
	G_local = nx.Graph()
	
	nodelist = copy.deepcopy(V)
	edgeslist = copy.deepcopy(E)

	
	
	#STEP 1
	#length = {}    #length of each link which is 1 for all edges
    
	#for (i,j) in edgeslist:
		#length[i,j] = 1
        
	#STEP 2
	nodespair = []
	for i in nodelist:
		for j in nodelist:
			if i != j and (j,i) not in nodespair:
				nodespair.append((i,j))
    
	#STEP 3
	#compute degree for each node
	degree = {}

	for (i,j) in edgeslist:
		if i in degree.keys():
			degree[i] += 1
		else:
			degree[i] = 1
		if j in degree.keys():
			degree[j] += 1
		else:
			degree[j] = 1
    
	for i in nodelist:
		if i in degree.keys():
			degree[i] = degree[i]/2    
		else: degree[i] = 0
    
	#calculate upper bound
	ub= {}
	ipath = {}    #k-th independent path between nodes pair
	PathLength = {} #the length of kth independent path between node pari
	Path_Onlyresilience = {}
	PathADTT = {}
	
    
	for (w,q) in nodespair:
		#print w,q
		temp_edgelist = copy.deepcopy(edgeslist)            #creat a temp list to search path
		ub[w,q] = min(degree[w],degree[q])     #up bound of possible number of independent paths
		ipath[w,q] = {}
		PathLength[w,q] = {}
		Path_Onlyresilience[w,q] = {}
		PathADTT[w,q] = {}
		k = 1
		flag = 1
		while(flag == 1):
			G_local.clear()
			G_local.add_nodes_from(nodelist)
			#G_local.add_edges_from(temp_edgelist)
			for headnode, tailnode in temp_edgelist:
				G_local.add_edge(headnode, tailnode, length = length[headnode, tailnode])
			try:
				temp = copy.deepcopy(nx.shortest_path(G_local, source=w, target=q, weight='length'))
				if  temp != []:
					PathADTT[w,q][k] = float("inf")
					ipath[w,q][k] = copy.deepcopy(temp)   #find the shortest path
					PathLength[w,q][k] = 0
					Path_Onlyresilience[w,q][k] = 1
					ipathtuple = []
					#print len(ipath[w,q][k]) 
					if len(ipath[w,q][k]) == 2:
						ipathtuple.append((ipath[w,q][k][0],ipath[w,q][k][1]))
						PathLength[w,q][k] = length[ipath[w,q][k][0], ipath[w,q][k][1]]
						Path_Onlyresilience[w,q][k] = resilience[ipath[w,q][k][0], ipath[w,q][k][1]]
						PathADTT[w,q][k] = ADTT[ipath[w,q][k][0], ipath[w,q][k][1]]
					else:
						for p in range(0, len(ipath[w,q][k]) - 1):
							ipathtuple.append((ipath[w,q][k][p],ipath[w,q][k][p+1]))
							PathLength[w,q][k] += length[ipath[w,q][k][p],ipath[w,q][k][p+1]]
							Path_Onlyresilience[w,q][k] *= resilience[ipath[w,q][k][p],ipath[w,q][k][p+1]]
							PathADTT[w,q][k] = min(PathADTT[w,q][k], ADTT[ipath[w,q][k][p],ipath[w,q][k][p+1]])
						#print "k=",k, ipath[w,q][k]
						#print ipathtuple
					
					#print k, PathLength[w,q][k], Path_Onlyresilience[w,q][k]
					
					for (s,t) in ipathtuple:
						temp_edgelist.remove((s,t))    #delete edges that used in previous shortest paths
						temp_edgelist.remove((t,s))
						
					#print temp_edgelist
					
					k +=1
					#if k > ub[w,q]:    #k should be smaller or equalt to Upbound 
						#flag = 0     
					#else: pass
				else:
					flag = 0
			except:
				flag = 0                   #no feasible path found
				
		

	return ipath, PathLength, Path_Onlyresilience, PathADTT  #return all 