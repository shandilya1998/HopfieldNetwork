from visualize import *

ball=get_image('ball')
mona=get_image('mona')
cat=get_image('cat')


vec_size=ball.shape[0]*ball.shape[1]
u=np.zeros(vec_size)
ball_S=np.reshape(ball,(vec_size,1))
cat_S=np.reshape(cat,(vec_size,1))
mona_S=np.reshape(mona,(vec_size,1))
