from visualize import *
from images import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from tqdm import tqdm
from hopfield import *
from sklearn.metrics import mean_squared_error

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

