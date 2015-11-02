"""
Network work resilience program v2.0
Based on network resilience formulated by our paper
Implemented by Weili Zhang, 2014-09-16

Input:
numNodes              #Number of nodes
V = range(0,numNodes) #set of nodes or vertices representing the cities in the network
E = []                #set of edges or arcs representingthe roalds connected to nodes in the network
l = {}                #length of each arc
q = {}                #survival probability of edge k of E
weight = {}           #weight of each link(could be cost, length, etc.)



Output:
L = {}                #k-th passageway between node pair i and j, for all i and j
P = {}                #reliability of the passageway connecting node pair i and j, L{k{(i,j): } : }
R                     #resilience of the network G
"""

#import random
#from networkx import *
#import matplotlib.pyplot as plt
from gurobipy import *
#from pygraphviz import *
#import math
from decimal import *
import copy
import random
import numpy as np
import sys

import passagewaysearch as PS
import GeneticAlgorithm as GA
#import resilience_evalueation as RE
from RE import resilience_evaluation
import friability_evaluation as FE
import hazard_matrix_compute

#define some useful functions
def swap(edgelist):  #swap the i,j in edge list and return
    
    edgelist_swap = []
    for (a,b) in edgelist:
        edgelist_swap.append((b,a))    
    return edgelist_swap

def combine(list1,list2): #combine elements in two list
    
    for i in list2:
        list1.append(i)
    
    return list1

def maxval(d):
    """ a) create a list of the dict's keys and values; 
    b) return the key with the max value"""  
    value_list=list(d.values())
    return max(value_list)

def minval(d):
    value_list=list(d.values())
    return min(value_list)

def minnormalize(dictionary, maxvalue, minvalue, positive):
    if positive == 1:
	for key, value in dictionary.items():
	    dictionary[key] = (value-minvalue)/(maxvalue-minvalue)
    elif positive == 0:
	for key, value in dictionary.items():
	    dictionary[key] = (maxvalue-value)/(maxvalue-minvalue)
    
    return dictionary

def sumnormalize(dictionary,postive):
    sumvalue = sum(dictionary.values())
    if postive == 1:
	for key, value in dictionary.items():
	    dictionary[key] = value/sumvalue
    if postive == 0:
	for key, value in dictionary.items():
	    dictionary[key] = (sumvalue-value)/sumvalue    
	    
    return dictionary

def MNDNgenerator(mean, cov):
    '''
    generate N-dimension random numbers following multivariate normal distribution range between (0,1)
    '''

    random_vector = np.random.multivariate_normal(mean,cov)
    for index in xrange(0,len(random_vector)):
	if random_vector[index] <= 0:
	    random_vector[index] = random.uniform(0.65,0.84)
	elif random_vector[index] > 1:
	    random_vector[index] = random.uniform(0.65,0.84)
	   
    return random_vector
	    
	    
def cov_to_covar(mean, cov):
    """
    Transfer the ceeficient of variation to covariance matrix
    """
    covar = copy.deepcopy(cov)
    for row in xrange(0, len(cov)):
	for col in xrange(0, len(cov[0])):
	    if col == row:
		covar[row][col] = (cov[row][col]*mean[row])**2
	    else:
		covar[row][col] = cov[row][col]*(cov[row][row]*mean[row])*(cov[col][col]*mean[col])
    return covar
	    
	
	
#class of network
class Network(object):
    def _init_(self):
	self.nodes = []
	self.emergnode = []
	self.arcs = []
	
	
    def addnode(self, numNodes):
	self.nodes = range(1,numNodes+1)  #generate nodes from 1, 2, ...
	
    def addemergenode(self, emergnodes):
	self.emergnode = emergnodes
	    
    def addarc(self,arcs):
	swap_arcs = swap(arcs)
	self.arcs = combine(arcs,swap_arcs)   #finish undirected network edge list
	self.arcs = tuplelist(self.arcs)	
	
    def addlength(self):
	self.length = {}
	for headnode,tailnode in self.arcs :
	    if headnode < tailnode:
		self.length[headnode, tailnode] = random.uniform(0.3,0.8)*100
		self.length[tailnode,headnode] = self.length[headnode, tailnode]
    
    def addresilience(self, resilience_list):
	#self.resilience[node1, node2] = value
	#self.resilience[node2, node1] = value
	self.resilience = {}
	for headnode,tailnode in self.arcs :
	    if headnode < tailnode:
		self.resilience[headnode, tailnode] = resilience_list.pop(0)
		self.resilience[tailnode,headnode] = self.resilience[headnode, tailnode]
		
    def addADTT(self, ADT_list):
	self.ADTT = {}
	for headnode,tailnode in self.arcs :
	    if headnode < tailnode:
		self.ADTT[headnode, tailnode] = ADT_list.pop(0)
		self.ADTT[tailnode,headnode] = self.ADTT[headnode, tailnode]	
		
    def addcost (self, cost_list):
	self.cost = {}
	for headnode,tailnode in self.arcs :
	    if headnode < tailnode:
		self.cost[headnode, tailnode] = cost_list.pop(0)
		self.cost[tailnode,headnode] = self.cost[headnode, tailnode]	
    
    
#****************************data input***************************************

def main():
    
    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    
    SampleNum = 1000   #Number of Monte Carlo Sample
    numNodes = 30        #Number of nodes
    emerg_nodes = [9,17]
    
    #genetic algorithm parameters
    GA_Reinfoce_open = 1  #run GA solve the repairment problem
    GA_Construct_open = 0  #run GA solve the construction problem
    pareto = 1            #run GA to find the pareto frontier when the objectives are conflict
    #GA parameters    
    num_population = 10    #number of populations
    max_iteration = 500   #maximum number of iterations
    max_time = 60        #maximum runtime
    crossover_rate = 0.7  #probabilty of crossover
    mutation_rate = 0.3   #probability of crossover
    top = 4               #number of chromos selected fromthe top of ranked population
    investment = 0        #budget  
    
    
    #V = range(1,numNodes+1) #set of nodes or vertices representing the cities in the network
    
    
    #set of edges or arcs representingthe roalds connected to nodes in the network
    E = [(1,2)	,
        (1,4)	,
        (2,5)	,
        (3,5)	,
        (4,9)	,
        (5,6)	,
        (5,9)	,
        (6,10)	,
        (6,11)	,
        (7,11)	,
        (8,9)	,
        (9,10)	,
        (9,12)	,
        (10,13)	,
        (10,14)	,
        (11,14)	,
        (12,13)	,
        (13,16)	,
        (13,17)	,
        (15,16)	,
        (16,19)	,
        (17,18)	,
        (17,19)	,
        (17,22)	,
        (18,20)	,
        (19,21)	,
        (19,22)	,
        (20,22)	,
        (20,23)	,
        (21,26)	,
        (22,24)	,
        (22,29)	,
        (22,30)	,
        (23,25)	,
        (23,24)	,
        (26,27)	,
        (27,28)	]
    
    print "number of bridge",len(E)
 

#****************************input end***************************************

#create the network according to above input

    G = Network()
    G.addnode(numNodes)
    G.addemergenode(emerg_nodes)
    G.addarc(E)
    G.addlength()
    
    #devide the bridges to two classes with 19 and 18 respectively
    
    bridgeclass1, bridgeclass2 = [], []
    for headnode, tailnode in G.arcs:
	if headnode < tailnode:
	    if random.random() < 0.5:
		if len(bridgeclass1) < 38:
		    bridgeclass1.append((headnode,tailnode))
		    bridgeclass1.append((tailnode,headnode))
		else:
		    bridgeclass2.append((headnode,tailnode))
		    bridgeclass2.append((tailnode,headnode))		    
		
	    else:
		if len(bridgeclass2) < 38:
		    bridgeclass2.append((headnode,tailnode))
		    bridgeclass2.append((tailnode,headnode))
		else:
		    bridgeclass1.append((headnode,tailnode))
		    bridgeclass1.append((tailnode,headnode))		    
		
	    
    
    #assign same mean for each class
    reliability_mean1, reliability_mean2 = [], []
    for headnode, tailnode in bridgeclass1:
	if headnode < tailnode:
	    reliability_mean1.append(0.7)
	    
    for headnode, tailnode in bridgeclass2:
	if headnode < tailnode:
	    reliability_mean2.append(0.6)
	
    #use these mean values to generate 19 random number from MVND
   
    reliability_COV1 = []
    with open('reliability_COV1.txt','r') as f:
	for line in f:
	    reliability_COV1.append(map(float, line.split(',')))
    
    f.close()
    
    reliability_COV2 = []
    with open('reliability_COV2.txt','r') as f:
	for line in f:
	    reliability_COV2.append(map(float, line.split(',')))
    
    f.close()        
	
    #check the length of mean and cov
    if len(reliability_mean1) == len(reliability_COV1):
	#change the coefficients fo variation to covariance: var = (mean*cov)**2
	reliability_Covar1 = cov_to_covar(reliability_mean1, reliability_COV1)
	true_reliability_mean_list1 = MNDNgenerator(reliability_mean1, reliability_COV1)
    elif len(reliability_mean1) == len(reliability_COV2):
	#change the coefficients fo variation to covariance: var = (mean*cov)**2
	reliability_Covar1 = cov_to_covar(reliability_mean1, reliability_COV2)
	true_reliability_mean_list1 = MNDNgenerator(reliability_mean1, reliability_COV2)	
	pass
    else:
	sys.exit("Numpy Error message:  mean and cov must have same length, mean is {} and cov is {}".format(len(reliability_mean1),len(reliability_COV1)))
    
    #assign these random values to each bridge in class1
    bridge_true_reliability_mean = {}
    i = 0
    for headnode, tailnode in bridgeclass1:
	if headnode < tailnode:
	    
	    bridge_true_reliability_mean[headnode, tailnode] = true_reliability_mean_list1[i]
	    bridge_true_reliability_mean[tailnode, headnode] = bridge_true_reliability_mean[headnode, tailnode]
	    i += 1
	
    #use these mean values to generate 16 random numbers  from MVND
   

    #check the length of mean and cov
    if len(reliability_mean2) == len(reliability_COV2):
	#change the coefficients fo variation to covariance: var = (mean*cov)**2
	reliability_Covar2 = cov_to_covar(reliability_mean2, reliability_COV2)	
	true_reliability_mean_list2 = MNDNgenerator(reliability_mean2, reliability_COV2)
    elif len(reliability_mean2) == len(reliability_COV1):
	#change the coefficients fo variation to covariance: var = (mean*cov)**2
	reliability_Covar2 = cov_to_covar(reliability_mean2, reliability_COV1)	
	true_reliability_mean_list2 = MNDNgenerator(reliability_mean2, reliability_COV1)
    else:
	sys.exit("Numpy Error message:  mean and cov must have same length, mean is {} and cov is {}".format(len(reliability_mean2),len(reliability_COV2)))
    
    #assign these random values to each bridge in class2
    i = 0
    for headnode, tailnode in bridgeclass2:
	if headnode < tailnode:
	    if (headnode, tailnode) in bridge_true_reliability_mean.keys() or (tailnode, headnode) in bridge_true_reliability_mean.keys():
		sys.exit("Duplicate edges in both bridge classes")
		print headnode, tailnode
	    
	    bridge_true_reliability_mean[headnode, tailnode] = true_reliability_mean_list2[i]
	    bridge_true_reliability_mean[tailnode, headnode] = bridge_true_reliability_mean[headnode, tailnode]
	    i += 1
    
    
    #use bridge_true_reliability_mean to generate distribution for each bridge named hazard correlation
    
    hazard_mean = []
    for headnode, tailnode in G.arcs:
	if headnode < tailnode:
	    hazard_mean.append(bridge_true_reliability_mean[headnode, tailnode])
    
    hazard_matrix = hazard_matrix_compute.generate_hazard()
    
    
        
    #generate ADT mean from a Uniform distribution
    ADT_mean = {}
    for headnode, tailnode in G.arcs:
	if headnode < tailnode:    
	    ADT_mean[headnode, tailnode] = random.randint(200,3000)
	    
    #generate cost mean from a normal distribution
    cost_mean = {}
    for headnode, tailnode in G.arcs:
	if headnode < tailnode:    
	    cost_mean[headnode, tailnode] =5 * ( G.length[headnode, tailnode]/100 + (1 - bridge_true_reliability_mean[headnode, tailnode]))
    
    
    #export the parameters of bridges
    f = open('bridge_parameters.txt','w')
    f.write('bridge :  reliability, ADT, length, cost\n')
    for key, value in  bridge_true_reliability_mean.items():
	if key[0] < key[1]:
	    f.write('{}:{},{},{}, {}\n'.format(key,value,ADT_mean[key], G.length[key], cost_mean[key]))
    f.close()    

    f1 = open('network original resilience.txt','w')
    f1.close()
    f2 = open('network GA Repair resilience.txt','w')
    f2.close()     
    
    #Monete Carlo Sampling Starts---------------------------------------------------------------------------------
    for sample_ID in xrange(1, 2):  #SampleNum
	
	seed = 7
	random.seed(seed)
	np.random.seed(seed)	
	
	#Monte Carlo Sampling on ADT
	ADT_sample = []
	
	for key,value in ADT_mean.items():
	    #ADT_sample.append(value)
	    #ADT_sample.append(random.normalvariate(value, 0.08*value))	
	    ADT_sample.append(random.normalvariate(value, 0.08*value))	
	#print ADT_sample
	    
	G.addADTT(ADT_sample)
	
	
	#print G.ADTT
	
	#Monte Carlo Sampling on ADT
	cost_sample = []
	for key,value in cost_mean.items():
	    cost_sample.append(value)
	    #cost_sample.append(random.normalvariate(value, 0.08*value))	
	    
	G.addcost(cost_sample)	
	
	#print G.cost
	    
	#Generate SampleNum of Monete Carlo Samples
	#check the length of mean and cov
	true_reliability_sample_list = []
	if len(hazard_mean) == len(hazard_matrix):
	    #transfer hazard cov to hazard covar
	    hazard_covar_matrix = cov_to_covar(hazard_mean, hazard_matrix)		    
	    true_reliability_sample_list = MNDNgenerator(hazard_mean,hazard_covar_matrix)
	else:
	    sys.exit("Numpy Error message:  mean and cov must have same length, mean is {} and cov is {}".format(len(hazard_mean),len(hazard_matrix)))	
	    
	#true_reliability_sample_list = hazard_mean
	
	G.addresilience(true_reliability_sample_list.tolist())
	#G.addresilience(true_reliability_sample_list)
	
	#G.resilience[1,2], G.resilience[2,1] = 1,1
	
	#for key, value in G.resilience.items():
		#G.resilience[key] = 0.99
	#G.resilience[23,24], G.resilience[24,23] = 0.99, 0.99 
	#G.resilience[23,25], G.resilience[25,23] = 0.99, 0.99
	
	nodelist = copy.deepcopy(G.nodes)   #store the orginal node list
	edgelist = copy.deepcopy(G.arcs)   #store the orginal node list
	
	maxADTT = max(G.ADTT.values())
	minADTT = min(G.ADTT.values())
	#Normal_ADTT = minnormalize(G.ADTT, maxADTT, minADTT, 1)
	#Normal_ADTT = sumnormalize(G.ADTT, 1)
	
	L, Length_Path, Fresilience_Path, Path_ADTT = PS.main(G.nodes,G.arcs, G.emergnode, G.length, G.resilience,G.ADTT)               #all passageway between node pair i and j, for all i and j
	    
	f = open('Independet_Paths.txt','w')
	f.write('Sample ID {}\n'.format(sample_ID))
	for i, j in L.keys():
	    f.write('nodes pair: ({},{}), Total independent paths {} \n '.format(i,j, len(L[i,j].keys())))
	    for k in L[i,j].keys():
		f.write('{},{},{},{},{}\n'.format(k,L[i,j][k], Length_Path[i,j][k], Fresilience_Path[i,j][k], Path_ADTT[i,j][k] ))

		
	f.close()     
	
	#get the max and min value from L_pk(i,j)
	#Max_Length_Path, Min_Length_Path, Sum_Length_Path = -float("inf"), float("inf"), 0
	#for key1,key2 in Length_Path.keys():
	    #for key3,value in Length_Path[key1,key2].items():
		#Sum_Length_Path += value
		#if value < Min_Length_Path:
		    #Min_Length_Path = value
		#if value > Max_Length_Path:
		    #Max_Length_Path = value
			
	#Normal_Path_Length = {}    #normalized length of path
	#for key1,key2 in Length_Path.keys():
	    #Normal_Path_Length[key1,key2] = {}
	    #for key3,value in Length_Path[key1,key2].items():
		#Normal_Path_Length[key1,key2][key3] = (Max_Length_Path - value)/(Max_Length_Path - Min_Length_Path)
	
	#normmalize the path length 	
	Normal_Path_Length = {}	
	for headnode, tailnode in Length_Path.keys():
	    Normal_Path_Length[headnode, tailnode] = {}
	    if len(Length_Path[headnode, tailnode].keys()) == 1:
		Normal_Path_Length[headnode, tailnode][1] = 1
	    elif len(Length_Path[headnode, tailnode].keys()) > 1:
		Temp_Sum = 0
		for k,value in Length_Path[headnode, tailnode].items():
		    Temp_Sum += value
		Normal_sum = 0
		for k,value in Length_Path[headnode, tailnode].items():
		    Normal_sum += Temp_Sum - value
		for k,value in Length_Path[headnode, tailnode].items():
		    Normal_Path_Length[headnode, tailnode][k] = ((Temp_Sum - value)/Normal_sum)*len(Length_Path[headnode, tailnode].keys())
	
	#normalize ADTT
	Normal_Path_ADTT = {}
	for headnode, tailnode in Path_ADTT.keys():
	    Normal_Path_ADTT[headnode, tailnode] = {}
	    if len(Path_ADTT[headnode, tailnode].keys()) == 1:
		Normal_Path_ADTT[headnode, tailnode][1] = 1
	    elif len(Path_ADTT[headnode, tailnode].keys()) > 1:
		Temp_Sum = 0
		for k,value in Path_ADTT[headnode, tailnode].items():
		    Temp_Sum += value
		for k,value in Path_ADTT[headnode, tailnode].items():
		    Normal_Path_ADTT[headnode, tailnode][k] = (value/Temp_Sum)*len(Path_ADTT[headnode, tailnode].keys())
	
	#shortest distance of node i with all emergency nodes
	omega = {}  
	for node in G.nodes:
	    if node in G.emergnode:
		omega[node] = 1
	    else:
		omega[node] = 0
		
		
	for head, tail in Length_Path.keys():
	    if head in G.emergnode or tail in G.emergnode:
		omega[head] = max(omega[head], 1/Length_Path[head,tail][1])
		omega[tail] = max(omega[tail], 1/Length_Path[head,tail][1])
		
	#compute the weight of each node
	NodeWeight = {}
	sumomega = sum(omega.values())
	for node in G.nodes:
	    NodeWeight[node] = omega[node]/sumomega
	
	G_Resilience = resilience_evaluation(G.nodes, L, Normal_Path_Length, Fresilience_Path, NodeWeight, Normal_Path_ADTT)
	
	total_cost = sum(G.cost.values())/2.0
	
	#result = FE.friability_evaluation(V,E,u,q,R_G)
	
	#F_G = result[0]     #friability of the whole network
	#F_max = result[1]   #maximum friability of nodes
	
	    
	print "The resilience of network G is: ", G_Resilience
	#print "The friability of network G is: ", F_G
	#print "The maximum friability of network G is: ", F_max
	
	f1 = open('network original resilience.txt','a')
	f1.write('{}\n'.format(G_Resilience))
	f1.close()    	
	
	if GA_Reinfoce_open == 1:   #case 1 reinforcement
	    #next use GA to solve the optimization problem
	    BinVar = []
	    #for i in V:    #these are loop to find out all the complementary edges
		#for j in V:
		    #if j != i:
			#if (i,j) not in E:
			    #BinVar.append((i,j))
			    #BinVar.append((j,i))
			    
	    #BinVar = [(1,5),(1,3),(1,8),(1,10), (2,10),(3,10),(3,7),(6,8),(7,10)]
	    
	    for headnode, tailnode in G.arcs:
		if headnode < tailnode: 
		    BinVar.append((headnode, tailnode))
	    
	    BinVar_swap = swap(BinVar)
	    BinVar = combine(BinVar, BinVar_swap)
	    
	    #for (i,j) in BinVar:
		#q[i,j] = 0.99
	    
	    if pareto == 0:
		GA_iteration, GA_run_time, GA_best_fitness, GA_best_solution = GA.main(nodelist,edgelist, BinVar, num_population,max_iteration, max_time, crossover_rate, mutation_rate, top,  seed, investment, GA_Reinfoce_open, GA_Construct_open, G, sample_ID)
		
		GA_best_solution2 = {}
		for headnode, tailnode in GA_best_solution.keys():
		    if headnode < tailnode:
			if GA_best_solution[headnode, tailnode] == 1:
			    GA_best_solution2[headnode, tailnode] = 1
		
		
		Total_cost = 0
		for headnode, tailnode in GA_best_solution2.keys():
		    if headnode < tailnode:
			Total_cost += GA_best_solution[headnode, tailnode] * G.cost[headnode, tailnode]
		
		
		print 'GA_iteration', GA_iteration, 'GA_run_time', GA_run_time, 'GA_best_fitness', GA_best_fitness, 'Total number of bridges', sum(GA_best_solution2.values()), 'Total cost', Total_cost, 'GA_best_solution', GA_best_solution2
			
		
		f2 = open('network GA Repair resilience.txt','a')
		f2.write('{} \t {} \t {} \t {}\n'.format(GA_best_fitness, sum(GA_best_solution2.values()),Total_cost,  GA_best_solution2))
		f2.close() 	
		
	    if pareto == 1:
		for investment in range (0, 190, 3):
		    print investment
		    
		    GA_iteration, GA_run_time, GA_best_fitness, GA_best_solution = GA.main(nodelist,edgelist, BinVar, num_population,max_iteration, max_time, crossover_rate, mutation_rate, top,  seed, investment, GA_Reinfoce_open, GA_Construct_open, G, sample_ID)
		    
		    GA_best_solution2 = {}
		    for headnode, tailnode in GA_best_solution.keys():
			if headnode < tailnode:
			    if GA_best_solution[headnode, tailnode] == 1:
				GA_best_solution2[headnode, tailnode] = 1
		    
		    
		    Total_cost = 0
		    for headnode, tailnode in GA_best_solution2.keys():
			if headnode < tailnode:
			    Total_cost += GA_best_solution[headnode, tailnode] * G.cost[headnode, tailnode]
		    
		    
		    print 'budget', investment, 'GA_iteration', GA_iteration, 'GA_run_time', GA_run_time, 'GA_best_fitness', GA_best_fitness, 'Total number of bridges', sum(GA_best_solution2.values()), 'Total cost', Total_cost, 'GA_best_solution', GA_best_solution2
			    
		    
		    f2 = open('network GA Repair resilience.txt','a')
		    f2.write('{} \t {} \t {} \t {} \t {}\n'.format(investment, GA_best_fitness, sum(GA_best_solution2.values()),Total_cost,  GA_best_solution2))
		    f2.close() 		    
	    
	    
	        
	
        

if __name__ == "__main__":
    
    main()
