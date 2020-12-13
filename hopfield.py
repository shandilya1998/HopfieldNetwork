import numpy as np 

class HopfieldNet():
    def __init__(self,niter, vec_size):
        self.vec_size = vec_size
        self.V = np.zeros((self.vec_size,1))
        self.U = np.zeros((self.vec_size,1))
        self.weights = np.zeros((self.vec_size,self.vec_size))
        self.U_d = np.zeros((self.vec_size,1))
        self.rmse = np.zeros((niter,1))
        self.flag = 0 # to load all images or only ball

    def set_flat(self, flag):
        self.flag = flag
    
    def load_weights(self, image_S):
        ''' 
        loads all images
        '''
        if self.flag==1:
            print('Loading all images')
            self.weights = np.matmul(image_S[0],image_S[0].T) + np.matmul(image_S[1],image_S[1].T) + np.matmul(image_S[2],image_S[2].T)
        if self.flag==0:
            print('Loading the image of the ball')
            self.weights = np.matmul(image_S, image_S.T)
    
    def image_loader(self, image):
        ''' 
        Loads patches of images
        '''
        shape = image.shape
        new_image = np.zeros((shape[0],shape[1]))
        new_image[0:int(shape[0]/2), int(shape[1]/4):int(shape[1]/2)] = image[0:int(shape[0]/2), int(shape[1]/4):int(shape[1]/2)]
        return new_image

    def damage_weights(self, p):
        '''
        Damages the weights of the network with probability p
        '''
        indices = np.random.randint(0,self.vec_size*self.vec_size-1,int(self.vec_size*self.vec_size*p))
        weights_damaged=np.copy(self.weights)
        weights_damaged=np.reshape(weights_damaged,(self.vec_size*self.vec_size,1))
        print('Damaging the weights')
        for i in tqdm(range(len(indices))):
            weights_damaged[indices[i]]=0
        weights_damaged = np.reshape(weights_damaged,(self.vec_size,self.vec_size))
        return weights_damaged
