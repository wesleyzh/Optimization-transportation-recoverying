# Random generate a network
# Dijkstra's algorithm for shortest paths

import random
from gurobipy import *

"Generate a connected random network with distance"
def RandomNetwork(Nodes, Sparse):
	numNodes = Nodes        #number of nodes
	nodes = range(0,numNodes)  #creat a seq of nodes
	arcs = []
	
	if Sparse == -1:
		S = random.randint(numNodes-1, (numNodes*(numNodes-1))/2)      #divided by two since essentially each undirected edge will be converted to two directed edges 
	else:
		if Sparse > numNodes-1 and Sparse <= numNodes*(numNodes-1)/2:
			S = Sparse
		else:
			print "User Error: Sparse value is outside of bounds: ({},{}), Sparse={}".format(numNodes-1,numNodes*(numNodes-1)/2,Sparse)
			exit(1)	
	
			e = 0
	#create connected graph ----------------------------------------
	for i in range(numNodes-1):
		prevNode=random.randint(0,i)
		arcs.append((prevNode,i+1))
		arcs.append((i+1,prevNode))
		   
	
	complete=[]
	for i in range(numNodes-1):
		for j in range(i+1,numNodes):
			complete.append((i,j))
	
	missing = set(complete) - set(arcs)
	
	if len(missing) >= S-int(len(arcs)/2):
		toAdd = random.sample(missing, S-int(len(arcs)/2))
	
		for i,j in toAdd:
			arcs.append((i,j))
			arcs.append((j,i)) 
	arcs = tuplelist(arcs)
	#end create connected graph ----------------------------------	
	
	#assign distance/cost to each arcs
	distance = {}
	mind = 10  #low bound of distance
	maxd = 20  #up bound of distance
	for i,j in arcs:
		distance[i,j] = random.uniform(mind,maxd)
	
	for i,j in arcs:
		distance[i,j] = distance[j,i]
	
	return nodes, arcs, distance
		

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
		best = min(alt,key = alt.get)        #find the minimum distance and remove from Q
		u = best
		Q.remove(best)
		
		if best == end:
			break
		
	return attributes		

def path(dresult):   #get the shortest path of Dijkstra
	global start,end
	temp = end
	path = []
	path.append(temp)
	while(temp!=start):
		temp = dresult[temp][1]
		path.append(temp)
	return path
	
		
seed = 100
random.seed(seed)

Nodes = 10
Sparse = -1

nodes, arcs,distance = RandomNetwork(Nodes, Sparse)
start = 0
end = 9 #end is used to stop the Dijkstra algorithm beacuase we do not want to caculate the whole network and only interested the shortest path (start-end)
shortestdistance = Dijkstra(nodes, arcs, distance, start,end)
path = path(shortestdistance)
path.reverse()  #reverse the elements in path

print "Network Distance: \n", distance, "\n"
print "Dijkstra Result: \n", shortestdistance, "\n"
print "Shortest Path from {} to {}: {} and the distance is {}".format(start,end, path, shortestdistance[end][0])
