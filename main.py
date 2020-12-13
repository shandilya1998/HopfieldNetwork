import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import os

def get_image(name):
    '''
    Import image from .txt files and return arrays
    '''
    df=pd.read_csv(name+'.txt',header=None)
    name_arr=np.array(df)
    name_arr=np.sign(name_arr)
    plt.imshow(name_arr,cmap='Greys_r')
    plt.title(f'Image of {name}')
    plt.axis('off')
    plt.savefig(os.path.join('images', name+'.png'))
    return name_arr
    
ball=get_image('ball')
mona=get_image('mona')
cat=get_image('cat')


vec_size=ball.shape[0]*ball.shape[1]
u=np.zeros(vec_size)
ball_S=np.reshape(ball,(vec_size,1))
cat_S=np.reshape(cat,(vec_size,1))
mona_S=np.reshape(mona,(vec_size,1))


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
            print('Loading the image of the ball')
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
            
                        
def retrieve(niter, lambdas, flag, p, image, image_S):
    dt=1/(100)
    Hop_net1=Hopfield_Net(niter)
    Hop_net1.flag=flag
    if flag == 0:
        Hop_net1.load_weights(image_S)
    else:
        Hop_net1.load_weights()
    Hop_net1.U = np.reshape(Hop_net1.image_loader(image),(9000,1))
    Hop_net1.weights=Hop_net1.damage_weights(p)
    Hop_net1.weights=Hop_net1.weights/9000
    images_arr=[]
    for i in tqdm(range(niter)):
        Hop_net1.U_d = -Hop_net1.U + np.matmul(Hop_net1.weights,Hop_net1.V)
        Hop_net1.U = Hop_net1.U + (Hop_net1.U_d)*dt
        Hop_net1.V = np.tanh(lambdas*Hop_net1.U)
        Hop_net1.rmse[i]=mean_squared_error(image_S,Hop_net1.V)
        
        img=np.reshape(Hop_net1.V,(90,100))
        images_arr.append(img)
    images_arr=np.array(images_arr)
    return images_arr,Hop_net1.rmse
    
def display(images_arr, rmse, niter, p, name):
    images_arr=np.array(images_arr)
    num = int(niter/10)
    fig, axes = plt.subplots(num, figsize = (5, 4.5*num))
    for i in range(int(niter/10)):
        axes[i].imshow(images_arr[10*i,:,:], 'Greys_r')
        axes[i].set_title(f'Image after {10*i} iterations for {p*100}% of weight damage')
        axes[i].axis('off')
    fig.savefig(os.path.join('images', name))

    plt.close('all')

    fig, axes = plt.subplots(1)
    axes.plot(rmse)
    axes.set_title(f'Plot of RMSE for {p*100}% of weight damage')
    axes.set_xlabel('Number of iterations')
    axes.set_ylabel('RMSE')
    axes.grid()
    fig.savefig(os.path.join('images', name[:-4]+'_rmse.png'))
    plt.close('all')
 
niter=50
images_arr,rmse=retrieve(niter, 10, 0, 0, ball, ball_S)   # for loading ball without damage
name = 'ball_{it}_0_damage_10_lambda_0_flag.png'.format(it = niter)
display(images_arr,rmse,niter,0, name)

niter=100
images_arr,rmse=retrieve(niter, 10, 1, 0.25, ball, ball_S)   # for loading all images with 25% damage
name = 'ball_{it}_25_damage_10_lambda_1_flag.png'.format(it = niter)
display(images_arr,rmse,niter,0.25, name)

images_arr,rmse=retrieve(niter, 10, 1, 0.5, ball, ball_S)   # for loading all images with 50% damage
name = 'ball_{it}_50_damage_10_lambda_1_flag.png'.format(it = niter)
display(images_arr,rmse,niter,0.5, name)

images_arr,rmse=retrieve(niter, 10, 1, 0.8, ball, ball_S)   # for loading all images with 80% damage
name = 'ball_{it}_80_damage_10_lambda_1_flag.png'.format(it = niter)
display(images_arr,rmse,niter,0.8, name)

niter=50
images_arr,rmse=retrieve(niter, 10, 0, 0, mona, mona_S)   # for loading mona without damage
name = 'mona_{it}_0_damage_10_lambda_0_flag.png'.format(it = niter)
display(images_arr,rmse,niter,0, name)

niter=100
images_arr,rmse=retrieve(niter, 10, 1, 0.25, mona, mona_S)   # for loading all images with 25% damage
name = 'mona_{it}_25_damage_10_lambda_1_flag.png'.format(it = niter)
display(images_arr,rmse,niter,0.25, name)

images_arr,rmse=retrieve(niter, 10, 1, 0.5, mona, mona_S)   # for loading all images with 50% damage
name = 'mona_{it}_50_damage_10_lambda_1_flag.png'.format(it = niter)
display(images_arr,rmse,niter,0.5, name)

images_arr,rmse=retrieve(niter, 10, 1, 0.8, mona, mona_S)   # for loading all images with 80% damage
name = 'mona_{it}_80_damage_10_lambda_1_flag.png'.format(it = niter)
display(images_arr,rmse,niter,0.8, name)

niter=50
images_arr,rmse=retrieve(niter, 10, 0, 0, cat, cat_S)   # for loading cat without damage
name = 'cat_{it}_0_damage_10_lambda_0_flag.png'.format(it = niter)
display(images_arr,rmse,niter,0, name)

niter=100
images_arr,rmse=retrieve(niter, 10, 1, 0.25, cat, cat_S)   # for loading all images with 25% damage
name = 'cat_{it}_25_damage_10_lambda_1_flag.png'.format(it = niter)
display(images_arr,rmse,niter,0.25, name)

images_arr,rmse=retrieve(niter, 10, 1, 0.5, cat, cat_S)   # for loading all images with 50% damage
name = 'cat_{it}_50_damage_10_lambda_1_flag.png'.format(it = niter)
display(images_arr,rmse,niter,0.5, name)

images_arr,rmse=retrieve(niter, 10, 1, 0.8, cat, cat_S)   # for loading all images with 80% damage
name = 'cat_{it}_80_damage_10_lambda_1_flag.png'.format(it = niter)
display(images_arr,rmse,niter,0.8, name)
