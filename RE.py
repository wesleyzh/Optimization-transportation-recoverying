#Procedure RE
def resilience_evaluation(Gnodes, L, Normal_Path_Length, Fresilience_Path, NodeWeight, Normal_Path_ADTT):

    #compute the resilience of node 
    Node_Resilience = {}
    for node in Gnodes:
	Node_Resilience[node] = 0
	for pairnode in Gnodes:
	    if pairnode != node:
		if (node, pairnode) in L.keys():
		    for k in L[node, pairnode].keys():
			Path_Resilience = 0.5*(Normal_Path_Length[node,pairnode][k] + Normal_Path_ADTT[node,pairnode][k])*Fresilience_Path[node,pairnode][k]
			Node_Resilience[node] += Path_Resilience
		else:    
		    for k in L[pairnode, node].keys():
			Path_Resilience = Normal_Path_Length[pairnode,node][k]*Normal_Path_ADTT[pairnode,node][k]*Fresilience_Path[pairnode,node][k]
			Node_Resilience[node] += Path_Resilience		
    
    #caculate the resilience of the network
    G_Reslience = 0
    for node in Gnodes:
	G_Reslience += NodeWeight[node]*Node_Resilience[node]/(len(Gnodes)-1)
	
    return G_Reslience