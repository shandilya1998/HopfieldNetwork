import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_image(filename):
    ''' 
    Import image from .txt files and return arrays
    '''
    df=pd.read_csv(filename+'.txt',header=None)
    arr=np.array(df)
    arr=np.sign(arr)
    plt.imshow(arr,cmap='Greys_r')
    plt.axis('off')
    plt.title(f'Image of {filename}')
    plt.savefig(filename+'.png')
    plt.show()
    return arr
