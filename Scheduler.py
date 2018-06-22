STATION_TABLE = []

import numpy as np
# n - station
# p - time slice 

# 1 x n usage of stations sum = rewards

# retrun 1 x n 
# class Scheduler:

class Scheduler:
    def __init__(self, total, n, p):
        # self.opt = opt
        self.total = total
        self.station_table = init_table(n, p)
        self.last_assignment = None


    def greedy_scheduler(self, sample_matrix, usage_vector):
        thredhold = 10
        w = [1 , 1]
        
        zero_feature = [sum(np.array(sample_station) <= thredhold) for sample_station in sample_matrix]
        final_feature = [w[0] * zero_feature[i] + w[1] * usage_vector[i] for i in range(len(usage_vector)) ]

        assignment = [int(x / sum(final_feature) * self.total) for x in final_feature]
        assignment[-1] = self.total - sum(assignment[:-1])
        self.last_assignment = assignment
        # print(zero_feature)
        # assignment = None
        return assignment

    def naive_scheduler(self, sample_matrix, usage_vector):
        # not related to sample_matrix
        assignment = [int(x / sum(usage_vector) * self.total) for x in usage_vector]
        assignment[-1] = self.total - sum(assignment[:-1])
        self.last_assignment = assignment
        return assignment



def init_table(n, p):
    return [[0] * p] * n

def main():
    sche = Scheduler(100, 3, 3)
    sample_matrix = init_table(3,3)
    usage_vector = [0] * 3

    usage_vector[0] = 20
    usage_vector[1] = 1000
    usage_vector[2] = 50
    print (sche.greedy_scheduler(sample_matrix, usage_vector)) 
    # for i in range(n):
    #     for j in range(p):
    #         STATION_TABLE[i][j]=
             
            
        
    
    
if __name__ == '__main__':
    main()