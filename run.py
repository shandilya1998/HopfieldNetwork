from retrieval import *
from visualize import *
from hopfield import *
from images import *

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
