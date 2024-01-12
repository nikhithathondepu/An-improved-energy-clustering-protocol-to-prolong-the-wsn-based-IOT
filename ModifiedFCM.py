import random
import numpy as np

#function to calculate distance between points
def euclideanDistance(p1, p2):
    return(sum((p1 - p2) ** 2)) ** 0.5

'''
this functiom will update centroid
'''
def centroidUpdate(update_cent):
    for k in range(len(update_cent)):
        for j in range(len(update_cent[k])):
            update_cent[k, j] = update_cent[k, j] / 100
    return update_cent

'''
function to sort nodes based on distance
'''
def sortNodes(cent, points):
    selected_cent = []
    for i in range(points.shape[0]):
        distance = []
        for j in range(cent.shape[0]):
            distance.append(0)
        selected_cent.append(distance)
    selected_cent = np.asarray(selected_cent)    
    for i in range(points.shape[0]):
        for j in range(cent.shape[0]):
            selected_cent[i, j] = euclideanDistance(points[i], cent[j])
    return selected_cent

#modified FCM to arrange nodes in clusteer based on nearest distance
def modifiedFCM(nodes, num_itr, num_cluster, random_cent_index):
    random_cent_index = []
    for i in range(0, num_cluster):
        random_cent_index.append(random.randint(0, (len(nodes)-1)))
    random_cent = nodes[random_cent_index, :]
    dist = sortNodes(random_cent, nodes)
    #now choose min distance points as cent
    cluster_label = np.asarray([np.argmin(i) for i in dist]) 
    for _ in range(num_itr): 
        selected_cent = []
        for i in range(num_cluster):
            tempCent = nodes[cluster_label == i].mean(axis=0) 
            selected_cent.append(tempCent)
        selected_cent = np.vstack(selected_cent)
        #print(selected_cent)
        minDist = sortNodes(selected_cent, nodes)
        cluster_label = np.asarray([np.argmin(k) for k in minDist])
    selected_cent = centroidUpdate(selected_cent)    
    return cluster_label, selected_cent 
