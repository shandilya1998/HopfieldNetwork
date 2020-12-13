from visualize import *
from images import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class Hopfield_Net():
    def __init__(self,niter):
        self.V = np.zeros((9000,1))
        self.U = np.zeros((9000,1))
        self.weights = np.zeros((9000,9000))
        self.U_d = np.zeros((9000,1))
        self.rmse = np.zeros((niter,1))
        self.flag = 0 # to load all images or only ball
    
    def load_weights(self, image_S = None):
        ''' 
        loads all images
        '''
        if self.flag==1:
            print('Loading all images')
            self.weights = np.matmul(mona_S,mona_S.T) + np.matmul(ball_S,ball_S.T) + np.matmul(cat_S,cat_S.T)
        if self.flag==0 and isinstance(image_S, np.ndarray):
            print('Loading the input image')
            self.weights = np.matmul(image_S,image_S.T)
    
    def image_loader(self,image):
        ''' 
        Loads patches of images
        '''
        new_image = np.zeros((90,100))
        new_image[0:45,25:50] = image[0:45,25:50]
        return new_image
    
    def damage_weights(self,p):
        ''' 
        Damages the weights of the network with probability p
        '''
        indices = np.random.randint(0,9000*9000-1,int(9000*9000*p))
        weights_damaged=np.copy(self.weights)
        weights_damaged=np.reshape(weights_damaged,(9000*9000,1))
        print('Damaging the weights')
        for i in tqdm(range(len(indices))):
            weights_damaged[indices[i]]=0
        weights_damaged = np.reshape(weights_damaged,(9000,9000))
        return weights_damaged


