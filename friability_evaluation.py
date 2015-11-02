
import resilience_evalueation as RE
import copy

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
    
def friability_evaluation(V,E,u,q, R_G):
    
    
    nodelist = copy.deepcopy(V)   #store the orginal node list
    edgelist = copy.deepcopy(E)   #store the orginal node list
    
    R_G_k = {}
    
    
    f = {}                #fraiability of edge k of E   
    for (ii,jj) in E: 
	E.remove((ii,jj))
	E.remove((jj,ii))
	q = {}                #normal operation reliability of edge k of E
	for i,j in E:
	    q[i,j] = 0.99       
	R_G_k[ii,jj] = RE.resilience_evalueation(V,E,u,q)
	f[ii,jj] = R_G - R_G_k[ii,jj]
	E.append((ii,jj))
	E.append((jj,ii))
       
    E = copy.deepcopy(edgelist)
    
    F = {}                #Friability of node i of V
    R_G_H = {}
    for ii in nodelist:
	V.remove(ii)
	for (z,y) in E:
	    if z == ii:
		E.remove((z,y))
		E.remove((y,z))
	u = {}                #population of city node i of V
	for i in V:
	    u[i] = 100    
	q = {}
	for i,j in E:
	    q[i,j] = 1 	
	R_G_H[ii] = RE.resilience_evalueation(V,E,u,q) -0.3
	F[ii] = R_G- R_G_H[ii]
	V.append(ii)
	E = copy.deepcopy(edgelist)
	
    
    E = copy.deepcopy(edgelist)
    V = copy.deepcopy(nodelist)
    
    u = {}
    for i in V:
	u[i] = 100
    w = weight(u, self_exjaust=0)
    
    F_G = 0
    for e in V:
	F_G += w[e]*F[e]
	
    F_max = max(F.values())  
    
    output = [F_G + 0.1, F_max]
    
    return output