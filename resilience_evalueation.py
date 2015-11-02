import passagewaysearch as PS

def weight(pop, self_exjaust):
    
    sum = 0
    for i,j in pop.items():
        sum += j
	
   #getcontext().prec = 4   #define the digits after decimal
    
    if self_exjaust == 0:    
        weights = {}    
        for i in pop.keys():
            weights[i] = (float(pop[i])/float(sum))
            
        return weights
    else:
        weights = {}    
        for i in pop.keys():
            weights[i] = float(pop[i])/(float(sum)-float(pop[i]))
                   
        return weights        
    
#Procedure RE
def resilience_evalueation(Gnodes, L, Normal_Path_Length, Fresilience_Path, NodeWeight):

    #compute the resilience of node 
    Node_Resilience = {}
    for node in Gnodes:
	Node_Resilience[node] = 0
	for pairnode in Gnodes:
	    if pairnode != node:
		if (node, pairnode) in L.keys():
		    for k in L[node, pairnode].keys():
			Path_Resilience = Normal_Path_Length[node,pairnode][k]*Fresilience_Path[node,pairnode][k]
			Node_Resilience[node] += Path_Resilience
		else:    
		    for k in L[pairnode, node].keys():
			Path_Resilience = Normal_Path_Length[pairnode,node][k]*Fresilience_Path[pairnode,node][k]
			Node_Resilience[node] += Path_Resilience		
    
    #caculate the resilience of the network
    G_Reslience = 0
    for node in Gnodes:
	G_Reslience += NodeWeight[node]*Node_Resilience[node]
	
    return G_Reslience/2.0