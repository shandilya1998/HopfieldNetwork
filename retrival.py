from hopfield import HopfieldNet
import numpy as np
import matplotlib.pyplot as plt

def retrieve(image, niter,lambdas,flag,p, dt):
    vec_size = image.shape[0]*image.shape[1]
    image_S = np.reshape(image, (vec_size,1))
    Hop_net1 = HopfieldNet(niter, vec_size)
    Hop_net1.flag = flag
    Hop_net1.load_weights(image_S)
    Hop_net1.U = np.reshape(Hop_net1.image_loader(image), (vec_size,1))
    Hop_net1.weights = Hop_net1.damage_weights(p)
    Hop_net1.weights = Hop_net1.weights/vec_size
    images_arr=[]
    for i in tqdm(range(niter)):
        Hop_net1.U_d = -Hop_net1.U + np.matmul(Hop_net1.weights, Hop_net1.V)
        Hop_net1.U = Hop_net1.U + (Hop_net1.U_d)*dt
        Hop_net1.V = np.tanh(lambdas*Hop_net1.U)
        Hop_net1.rmse[i]=mean_squared_error(image_S, Hop_net1.V)

        img=np.reshape(Hop_net1.V, (image.shape[0],image.shape[1]))
        images_arr.append(img)
    images_arr=np.array(images_arr)
    return images_arr,Hop_net1.rmse

def display(images_arr, rmse, niter, p, name):
    images_arr=np.array(images_arr)
    for i in range(int(niter/10)):
        plt.imshow(images_arr[10*i,:,:],'Greys_r')
        plt.title(f'Image after {10*i} iterations for {p*100}% of weight damage')
        plt.show()

    plt.plot(rmse)
    plt.title(f'Plot of RMSE for {p*100}% of weight damage')
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    plt.grid()
    plt.savefig(name)
