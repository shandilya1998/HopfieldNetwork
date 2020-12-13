import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
