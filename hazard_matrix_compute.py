import math

def compute_distance(A,B):
    distance_value = math.sqrt((A['x']-B['x'])**2 + (A['y']-B['y'])**2)
    
    return distance_value

def generate_hazard():
    coordinate = {}
    
    coordinate = {
        (1,2)	:{ 'x' :	3.2	, 'y' :	17	},
        (1,4)	:{ 'x' :	4.5	, 'y' :	18	},
        (2,5)	:{ 'x' :	4	, 'y' :	14	},
        (3,5)	:{ 'x' :	4	, 'y' :	10	},
        (4,9)	:{ 'x' :	7	, 'y' :	16	},
        (5,6)	:{ 'x' :	6	, 'y' :	11	},
        (5,9)	:{ 'x' :	7	, 'y' :	14	},
        (6,10)	:{ 'x' :	8	, 'y' :	10	},
        (6,11)	:{ 'x' :	6	, 'y' :	6	},
        (7,11)	:{ 'x' :	4.5	, 'y' :	3	},
        (8,9)	:{ 'x' :	11	, 'y' :	17	},
        (9,10)	:{ 'x' :	9	, 'y' :	12	},
        (9,12)	:{ 'x' :	10.5	, 'y' :	14	},
        (10,13)	:{ 'x' :	10.5	, 'y' :	10	},
        (10,14)	:{ 'x' :	10	, 'y' :	8	},
        (11,14)	:{ 'x' :	9	, 'y' :	5	},
        (12,13)	:{ 'x' :	12.5	, 'y' :	12	},
        (13,16)	:{ 'x' :	14	, 'y' :	12	},
        (13,17)	:{ 'x' :	14	, 'y' :	10	},
        (15,16)	:{ 'x' :	14.5	, 'y' :	15	},
        (16,19)	:{ 'x' :	16	, 'y' :	13.2	},
        (17,18)	:{ 'x' :	16	, 'y' :	19	},
        (17,19)	:{ 'x' :	17	, 'y' :	12	},
        (17,22)	:{ 'x' :	18	, 'y' :	11	},
        (18,20)	:{ 'x' :	16.5	, 'y' :	6.5	},
        (19,21)	:{ 'x' :	18	, 'y' :	15	},
        (19,22)	:{ 'x' :	19	, 'y' :	13.2	},
        (20,22)	:{ 'x' :	19	, 'y' :	9	},
        (20,23)	:{ 'x' :	19	, 'y' :	4.5	},
        (21,26)	:{ 'x' :	21	, 'y' :	18	},
        (22,24)	:{ 'x' :	20.5	, 'y' :	9.5	},
        (22,29)	:{ 'x' :	22.5	, 'y' :	12	},
        (22,30)	:{ 'x' :	22.5	, 'y' :	10	},
        (23,25)	:{ 'x' :	22	, 'y' :	3	},
        (23,24)	:{ 'x' :	21	, 'y' :	5	},
        (26,27)	:{ 'x' :	24	, 'y' :	18	},
        (27,28)	:{ 'x' :	25.5	, 'y' :	18	},
        }
    
    #print coordinate
    
    #compute the distance between each pair of bridges
    distance = {}
    for key1,value1 in coordinate.items():
        for key2,value2 in coordinate.items():
            if key1 == key2:
                distance[key1,key2] = 0
            else:
                distance[key1,key2] = compute_distance(value1, value2)
    
    #print "Distance Matrix \n"
            
    ##output as matrix
    #for key1 in coordinate.keys():
        #print key1, ',',
    #print '\n'
        
    #for key1 in coordinate.keys():
        #print key1, ':',
        #for key2 in coordinate.keys():
            #print distance[key1,key2], ',',
        #print '\n'
    
    
    #compute the hazard correlation matrix with r_0 = average distance
    r_0 = sum(distance.values())/len(distance.values())
    
    hazard_cor = {}
    for key, value in distance.items():
        if value != 0:
            hazard_cor[key] = math.exp(-float(value/r_0))
        else:
            hazard_cor[key] = 0.1
    
    
    #print "Hazard Correlation Matrix \n"
        
        
    ##output as matrix
    #for key1 in coordinate.keys():
        #print key1, ';',
    #print '\n'
        
    #for key1 in coordinate.keys():
        ##print key1, ':',
        #for key2 in coordinate.keys():
            #print hazard_cor[key1,key2], ',',
        #print '\n'
    
    
    #convert the matrix into list
    hazard_COV = []
    
    for key1 in coordinate.keys():
        temp = []
        for key2 in coordinate.keys():
            temp.append(hazard_cor[key1,key2])
        hazard_COV.append(temp)
    
    return hazard_COV

if __name__ == "__main__":
    
    hazardmartix = generate_hazard()