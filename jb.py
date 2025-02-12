# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 00:59:04 2024

@author: jishnu
"""

import numpy as np
import matplotlib.pyplot as plt

# FUNCTION TO SHOW AN IMAGE IN SPYDER
def show(image):
    import cv2
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    # resized = cv2.resize(image, (image.shape[1]//5,image.shape[0]//5))
    resized = cv2.resize(image, (1920,1080))
    # cv2.moveWindow("Output", 200,200)
    cv2.imshow("Output", resized)
    cv2.resizeWindow("Output", 860, 540)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def Plot_Histogram(image, ax=1):
    import cv2
    '''
    THE SECOND PARAMETER IS EITHERR 0 OR 1. 1 MEANS THE SUM IS DONE
    TOP TO BOTTOM, 0 MEANS SUM IS DONE HORIZONTALLY. IT RETURNS THE THE ARRAY OF SUM
    '''
    x = np.arange(0, image.shape[ax], dtype=np.uint32)
    y = np.zeros(image.shape[ax], dtype=np.uint32)
    for k in range(image.shape[ax]):
        if ax == 1:
            arr = image[:, k]
            y[k] = np.sum(arr)
        else:
            arr = image[k, :]
            y[k] = np.sum(arr)
    plt.plot(x,y)
    plt.show()
    return y



def Cosine_Similarity(vec1, vec2):
    """
    Parameters
    ----------
    vec1 :  TYPE.
                1D numpy array
            DESCRIPTION.
                vec1 is the first vector that will be used to calculate
        similarity.
        
    vec2 :  TYPE.
                1D numpy array
        DESCRIPTION.
                vec2 is the first vector that will be used to calculate
        similarity.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if type(vec1) is list:
        vec1 = np.array(vec1)
    if type(vec2) is list:
        vec2 = np.array(vec1)
    if(vec1.shape[0] != vec2.shape[0]):
        raise ValueError("Dimension size of the vectors must be same")
    else:
        dot_prod = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            print(norm1, norm2)
            return 0
        return dot_prod/(norm1 * norm2)

def Minkowski_Distance_Between(vec1, vec2, p=2):
    
    if(vec1.shape[0] != vec2.shape[0]):
        raise ValueError("Dimension size of the vectors must be same")
    sum = 0
    for a,b in zip(vec1, vec2):
        sum += (a-b)**p
    return sum**(1/p)
    
def Vector_Average(ls, vec_len):
    aver = [] # average vec
    n = len(ls) # n vectors in a cluster
    for i in range(vec_len):
        sum = 0
        for j in range(n):
            sum += ls[j][i]
        sum = sum/vec_len
        aver.append(sum)
    return aver

# Testing Purpose
# a1 = np.array([1,1])
# a2 = np.array([-1, -1])
# print(cosine_similarity(a1, a2))
