from visualize import get_image
from retrieval import retrieve, display

def main():
    dt = 0.001
    ball = get_image('ball')
    mona = get_image('mona')
    cat = get_image('cat')

    niter=50
    images_arr, rmse = retrieve(niter, 10, 0, 0)   
    # for loading ball without damage
    display(images_arr, rmse, niter, 0, '')

    niter=100
    images_arr, rmse = retrieve(niter, 10, 1, 0.25)   
    # for loading all images with 25% damage
    display(images_arr,rmse,niter,0.25)

    images_arr, rmse = retrieve(niter, 10, 1, 0.5)   
    # for loading all images with 50% damage
    display(images_arr,rmse,niter,0.5)

    images_arr, rmse = retrieve(niter, 10, 1, 0.8)   
    # for loading all images with 75% damage
    display(images_arr, rmse, niter, 0.8)

main()
