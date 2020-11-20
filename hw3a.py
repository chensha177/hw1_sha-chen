import tensorflow.compat.v1 as tf
import random

from collections import deque 
import time
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow_probability as tfp
import gym
import os
import time
import pickle
timeout=time.time()+60*60*24
#! /usr/bin/python -u
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) 
#os.environ['COLAB_SKIP_TPU_AUTH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()
env = gym.make("Breakout-v0")
obs=env.reset()
obs.shape
input_height=80
input_width=80
input_channels=1
conv_n_maps=[32,64,64]
conv_kernel_sizes=[(8,8),(4,4),(3,3)]
conv_strides=[4,2,1]
conv_paddings=["SAME"]*3
conv_activation=[tf.nn.relu]*3
n_hidden_in=80*80
n_hidden=512

learning_rate=0.001
momentum=0.95

replay_memory_size=500000
replay_memory=deque([],maxlen=replay_memory_size)

eps_min=0.1
eps_max=1.0
eps_decay_steps=20000000

def epsilon_greedy(q_values,step):
  epsilon=max(eps_min,eps_max-(eps_max-eps_min)*step/eps_decay_steps)
  if np.random.rand()<epsilon:
    return np.random.randint(n_outputs)
  else:
    return np.argmax(q_values)

n_steps=40000
training_start=1000
training_interval=4
save_steps=1000
copy_steps=10000
discount_rate=0.95
skip_start=90
batch_size=50
iteration=0
checkpoint_path="./my_dqn3t1.ckpt"
done=True

mapacman_color=210+164+74
def preprocess(image):
  image = image[35:195] # crop
  image = image[::2,::2,0] # downsample by factor of 2
  image[image == 144] = 0 # erase background (background type 1)
  image[image == 109] = 0 # erase background (background type 2)
  image[image != 0] = 1 # everything else just set to 1
  return np.reshape(image.astype(np.float).ravel(), [80,80,1])

# def preprocess_observation(obs):
#   img = obs[1:176:2, ::2] # crop and downsize
#   img = img.sum(axis=2) # to greyscale
#   img[img==mspacman_color] = 0 # Improve contrast
#   img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
#   return img.reshape(88, 80, 1)



class ReplayMemory:
  def __init__(self,maxlen):
    self.maxlen=maxlen
    self.buf=np.empty(shape=maxlen,dtype=np.object)
    self.index=0
    self.length=0
  def append(self,data):
    self.buf[self.index]=data
    self.length=min(self.length+1,self.maxlen)
    self.index=self.index+1

  def sample(self,batch_size,with_replacement=True):
    if with_replacement:
      indices=np.random.randint(self.length,size=batch_size)
    else:
      indices=np.random.permutation(self.length)[:batch_size]
    return self.buf[indices]
def sample_memories(batch_size):
  cols=[[],[],[],[],[]]##state, action, reward, next_state, continue
  for memory in replay_memory.sample(batch_size):
    for col,value in zip(cols,memory):
      col.append(value)
  cols=[np.array(col) for col in cols]
  return cols[0], cols[1],cols[2].reshape(-1,1),cols[3],cols[4].reshape(-1,1)

replay_memory_size=500000000
replay_memory=ReplayMemory(replay_memory_size)


hidden_activation=tf.nn.relu
n_outputs=env.action_space.n
initializer=tf.compat.v1.keras.initializers.VarianceScaling()
initializer=tf.initializers.glorot_uniform()
def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial, name=name)
def bias_variable(shape, name=None):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial, name=name)
def q_network(X_state,name):
  prev_layer=X_state/128.0
  with tf.variable_scope(name) as scope:
    for n_maps,kernel_size,strides,padding,activation in zip(conv_n_maps,conv_kernel_sizes,conv_strides,conv_paddings,conv_activation):
      prev_layer=tf.layers.conv2d(prev_layer,filters=n_maps,kernel_size=kernel_size,strides=strides,padding=padding,activation=activation,kernel_initializer=initializer)
    last_conv_layer_flat=tf.reshape(prev_layer,shape=[-1,n_hidden_in])
    # w_fc1 = weight_variable([n_hidden_in, n_hidden])
    # b_fc1 = bias_variable([n_hidden])
    # h_fc1 = tf.nn.relu(tf.matmul(last_conv_layer_flat, w_fc1) + b_fc1)
    # w_fc2 = weight_variable([n_hidden, n_outputs])
    # b_fc2 = bias_variable([n_outputs])
    # outputs = tf.matmul(h_fc1, w_fc2) + b_fc2

    hidden=tf.layers.dense(last_conv_layer_flat,n_hidden,activation=hidden_activation,kernel_initializer=initializer)
   

    outputs=tf.layers.dense(last_conv_layer_flat,n_outputs,kernel_initializer=initializer)
  trainable_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope.name)
  trainable_vars_by_name={var.name[len(scope.name):]:var for var in trainable_vars}
  return outputs, trainable_vars_by_name



tf.reset_default_graph()
X_state=tf.placeholder(tf.float32,shape=[None,input_height,input_width,input_channels])
online_q_values,online_vars=q_network(X_state,name="q_networks/onine")
target_q_values,target_vars=q_network(X_state,name="q_networks/target")
copy_ops=[target_var.assign(online_vars[var_name]) for var_name,target_var in target_vars.items()]
copy_online_to_target=tf.group(*copy_ops)
with tf.variable_scope("train"):
  X_action=tf.placeholder(tf.int32,shape=[None])
  y=tf.placeholder(tf.float32,shape=[None,1])
  q_value=tf.reduce_sum(online_q_values*tf.one_hot(X_action,n_outputs),axis=1,keepdims=True)
  error=tf.abs(y-q_value)
  clipped_error=tf.clip_by_value(error,0.0,1.0)
  linear_error=2*(error-clipped_error)
  loss=tf.reduce_mean(tf.square(clipped_error)+linear_error)
  global_step=tf.Variable(0,trainable=False,name='global_step')
  optimizer=tf.train.MomentumOptimizer(learning_rate,momentum,use_nesterov=True)
  training_op=optimizer.minimize(loss,global_step=global_step)
init=tf.global_variables_initializer()


loss_val=np.infty
game_length=0
total_max_q=0
mean_max_q=0.0
game_mean_max_q=[]

saver=tf.train.Saver()

#####train the neural network
with tf.Session() as sess:
  init.run()
  copy_online_to_target.run()
  while True:
    step=global_step.eval()
     # if step>= n_steps:
     #   break
    if time.time()>timeout:
      break
    
    if done:
      obs=env.reset()
      for skip in range(skip_start):
        obs,reward,done,info=env.step(0)
      state=preprocess(obs)
    ##online DQN  

    q_values=online_q_values.eval(feed_dict={X_state:[state]})
    action=epsilon_greedy(q_values,step)
    ##Online DQN plays
    obs,reward,done,info=env.step(action)
    next_state=preprocess(obs)

    replay_memory.append((state,action,reward,next_state,1.0-done))
    state=next_state

    total_max_q+=q_values.max()
    game_length+=1
    if done:
      mean_max_q=total_max_q/game_length
      total_max_q=0.0
      game_length=0
      iteration+=1
      game_mean_max_q.append(mean_max_q)
      f = open('somedata3t1', 'wb')
      pickle.dump([game_mean_max_q,iteration], f,protocol=2)
      f.close()
      if iteration % 20==0:
        print("===========")
        print("Iteration: ",iteration)
        print("-----",)
        print("Average maximal q value {}".format(mean_max_q))


    if iteration<training_start or iteration % training_interval !=0:
      continue
    X_state_val, X_action_val, rewards, X_next_state_val, continues=(sample_memories(batch_size))
    next_q_values=target_q_values.eval(feed_dict={X_state:X_next_state_val})
    max_next_q_values=np.max(next_q_values,axis=1,keepdims=True)
    y_val=rewards+continues*discount_rate*max_next_q_values

    _,loss_val=sess.run([training_op,loss],feed_dict={X_state:X_state_val,X_action:X_action_val,y:y_val})

    if step %copy_steps==0: 
      copy_online_to_target.run()
    if step % save_steps==0:
      saver.save(sess,checkpoint_path)

t=range(iteration)
plt.plot(t,game_mean_max_q)
plt.xlabel('Iteration')
plt.ylabel("maximum of the Q function")
plt.savefig('hw3a.pdf')





##use the trained model to play games

with tf.Session() as sess:
  #new_saver=tf.train.import_meta_graph('my_dqn311.ckpt.meta')
  if os.path.isfile(checkpoint_path+'.index'):
    saver.restore(sess, checkpoint_path)

  #copy_online_to_target.run()
  game_l=1000
  game_reward=[]
  sum_game_reward=[]
  step=global_step.eval()
  done=False
  obs=env.reset()
  state=preprocess(obs)
  for i in range(game_l):
    while True:
      q_values=target_q_values.eval(feed_dict={X_state:[state]})
      action=np.argmax(q_values)
      print(action)
      obs,reward,done,info=env.step(action)
      next_state=preprocess(obs)
      game_reward.append(reward)

      state=next_state
      if done:       
        sum_game_reward.append(sum(game_reward))
        f = open('somegame311', 'wb')
        pickle.dump([sum_game_reward,i], f,protocol=2)
        f.close()
        print("===========")
        print("Episode: ",i)
        print("-----")
        print(game_reward)
        print("Sum of rewards {}".format(sum(game_reward)))
        game_reward=[]
        obs=env.reset()
        state=preprocess(obs)
        break




t=range(game_l)
plt.plot(t,sum_game_reward)
plt.xlabel('episode')
plt.ylabel("total reward")
plt.savefig('hw3a2.pdf')


 
    





















