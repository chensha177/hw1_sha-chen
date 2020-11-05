import tensorflow.compat.v1 as tf
import random

from collections import deque 
import time
import matplotlib.pyplot as plt
import numpy as np
print(tf.__version__)
#import tensorflow_probability as tfp
import gym
import os
#os.environ['COLAB_SKIP_TPU_AUTH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

env = gym.make("CartPole-v0")
state_size=4
action_size = 2
print(env.action_space.n)
tf.compat.v1.disable_eager_execution()
possible_actions=np.identity(action_size,dtype=int).tolist()
learning_rate_train=0.001
learning_rate_fit=0.002
num_epochs=300
batch_size=800
training=True
epoch=1
##environment hyperparameters

class PGNetwork:
  def __init__(self,state_size,action_size,learning_rate_train,learning_rate_fit,name="PGNetwork"):
    self.state_size=state_size
    self.action_size=action_size
    self.learning_rate_train=learning_rate_train
    self.learning_rate_fit=learning_rate_fit
    with tf.variable_scope("name"):
      with tf.name_scope("input"):
        self.inputs_=tf.placeholder(tf.float32,[None,state_size],name="inputs_")
        self.actions=tf.placeholder(tf.float32,[None,action_size],name="actions")
        self.advantage_=tf.placeholder(tf.float32,[None], name="advantage_")
        self.values_=tf.placeholder(tf.float32,[None], name="values")
      with tf.name_scope("conv1"):
        self.fc1 = tf.layers.dense(inputs = self.inputs_, units = 32,kernel_initializer = tf.initializers.glorot_uniform(),activation = tf.nn.relu,name = 'fc1')
        self.fc2 = tf.layers.dense(inputs = self.fc1, units = 64, kernel_initializer = tf.initializers.glorot_uniform(),activation = tf.nn.relu, name = 'fc2')
       # self.fc3 = tf.layers.dense(inputs = self.fc2, units = 16, kernel_initializer = tf.initializers.glorot_uniform(),activation = tf.nn.relu,name = 'fc3')
        
        self.logits = tf.layers.dense(inputs = self.fc2, units = action_size, kernel_initializer = tf.initializers.glorot_uniform(),activation = None,name = 'logits')
        
      with tf.name_scope("softmax"):
        self.action_distribution = tf.nn.softmax(self.logits)
      with tf.name_scope("value"):
        self.value = tf.layers.dense(inputs = self.fc1, units = 1, activation = None,name = 'value')
        
      #   inputs=Input(shape=self.state_size)
    
      #   self.x=tf.keras.layers.conv2d(1,4,activation='relu',input_shape=input_shape[1:])(inputs)
      #   self.y=tf.keras.layers.Dense(24,activation='relu')(self.x)
      #   self.logits =tf.keras.layers.Dense(2,activation='softmax')(self.y)
        
      # with tf.name_scope("softmax"):
      #   self.action_distribution=tf.nn.softmax(self.logits)
      with tf.name_scope("loss"):
        self.cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.actions)
        self.weighted_negative_likelihoods=tf.multiply(self.cross_entropy,self.advantage_)
        self.loss=tf.reduce_mean(self.weighted_negative_likelihoods)
      with tf.name_scope("value_loss"):
       # self.error=tf.reduce_sum(tf.square(self.values_-self.value),name="mse")
        self.error_=tf.nn.l2_loss(self.values_-self.value,name="mse")
        self.error=tf.reduce_mean(self.error_)
      # with tf.name_scope("v_loss"):
      #   self.mse=tf.losses.mean_squared_error(labels=self.value,predictions=self.values_)
      #   self.v_loss=tf.reduce_mean(self.mse)
      with tf.name_scope("train"):
        self.optimizer=tf.train.RMSPropOptimizer(self.learning_rate_train)
        self.train_opt=self.optimizer.minimize(self.loss)
      with tf.name_scope("fit"):
        self.optimizer=tf.train.RMSPropOptimizer(self.learning_rate_fit)
        self.fit_opt=self.optimizer.minimize(self.error)


# stacked_frames=deque([np.zeros((80,80),dtype=np.float32) for i in range(stack_size)],maxlen=4)
# def preprocess(image):
# # prepro 210x160x3 uint8 frame into 6400 (80x80) 2D float array 
#   image = image[35:195] # crop
#   image = image[::2,::2,0] # downsample by factor of 2
#   image[image == 144] = 0 # erase background (background type 1)
#   image[image == 109] = 0 # erase background (background type 2)
#   image[image != 0] = 1 # everything else (paddles, ball) just set to 1
#   return np.reshape(image.astype(np.float).ravel(), [80,80])
# def one_hot_encoding(action):
#   if action==2:
#     action_e=[0,1]
#   elif action==3:
#     action_e=[1,0]
#   return action_e
##prepare last 4 processed frames to be fed to conv net
# def stack_frames(stacked_frames,state,is_new_episode):
#   frame=preprocess(state)
#   if is_new_episode:
#     stacked_frames=deque([np.zeros((80,80),dtype=np.float32) for i in range(stack_size)],maxlen=4)
#     stacked_frames.append(frame)
#     stacked_frames.append(frame)
#     stacked_frames.append(frame)
#     stacked_frames.append(frame)

#     stacked_state=np.stack(stacked_frames,axis=2)
#   else:
#     stacked_frames.append(frame)
#     stacked_state=np.stack(stacked_frames,axis=2)
#   return stacked_state,stacked_frames

def discount_rewards(r,gamma=0.95,normalization=False):
  discounted_r=np.zeros_like(r)
  running_add=0
  for t in reversed(range(0,len(r))):
    running_add=running_add*gamma+r[t]
    discounted_r[t]=running_add
  if normalization:
    mean=np.mean(discounted_r)
    std=np.std(discounted_r)
    discounted_r=(discounted_r-mean)/std
  return discounted_r
    


def make_batch(batch_size):
  num_episode=0
  states,actions,reward_of_batch,reward_of_episode,discounted_rewards,y_sum,y=[],[],[],[],np.array([]),np.array([]),[]
  state=env.reset()
  batch=0
  while True:
    action_probability_distribution=sess.run(PGNetwork.action_distribution,feed_dict={PGNetwork.inputs_:[state]})
   # print(action_probability_distribution)
    action=np.random.choice(range(action_probability_distribution.shape[1]),p=action_probability_distribution.ravel())
    
    next_state,reward,done,info=env.step(action)
    states.append(state)
    action_onehot=np.zeros(action_size)
    action_onehot[action]=1
    actions.append(action_onehot)
    reward_of_episode.append(reward)
    #next_state,stacked_frames=stack_frames(stacked_frames,next_state,False)

    if not done:
      
      next_state_value=sess.run(PGNetwork.value,feed_dict={PGNetwork.inputs_:[next_state]})
      y.append(next_state_value[0]+reward)
      state=next_state
    else:
      
      state=env.reset()
      y.append(reward)
      reward_of_batch+=reward_of_episode
      y_sum=np.concatenate((y_sum,np.array(y)))
      discounted_rewards=np.concatenate((discounted_rewards,discount_rewards(reward_of_episode,gamma=0.95,normalization=True)))
      #state,stacked_frames=stack_frames(stacked_frames,state,True)
      
      batch+=len(reward_of_episode)
      reward_of_episode=[]
      y=[]
      num_episode += 1   
      if batch>batch_size:
        break

  return states,actions,reward_of_batch,np.array(discounted_rewards),np.array(y_sum),num_episode
  


  # for i in range(batch_size):
  #   action_probability_distribution=sess.run(PGNetwork.action_distribution,feed_dict={PGNetwork.inputs_:[state]})
  #   action=np.random.choice(range(action_probability_distribution.shape[1]),p=action_probability_distribution.ravel())
    # if action==0:
    #   action=2
    # elif action==1:
    #   action=3
#     next_state,reward,done,info=env.step(action)
#     states.append(state)
#     action_=np.zeros(action_size)
#     action_[action]=1
#     actions.append(action_)
#     rewards_of_batch.append(reward)
#     # next_state,stacked_frames=stack_frames(stacked_frames,next_state,False)
#     state=next_state
#     if done:
#       state=env.reset()
#       # state,stacked_frames=stack_frames(stacked_frames,state,True)
#       episode_num+=1   
#       break
#   discounted_rewards=discount_rewards(rewards_of_batch,gamma=0.95,normalization=False)
#      # if len(np.concatenate(rewards_of_batch))> batch_size:
#      #   break
#      # rewards_of_episode=[]
#      #  episode_num+=1
#     #  state=env.reset()
#     #  state,stacked_frames=stack_frames(stacked_frames,state,True)
#    # else:
#   # next_state,stacked_frames=stack_frames(stacked_frames,next_state,False)
#   #  state=next_state
#   ##store the discounted reward of each episode
#  # discounted_reward_episode=[];
# #  for i in range(episode_num):
#  #   discounted_reward_episode.append(discounted_rewards[i][0])
#   return states, actions,rewards_of_batch,discounted_rewards, state, episode_num


##reset the graph
tf.reset_default_graph()
##initialize the networks
PGNetwork=PGNetwork(state_size,action_size,learning_rate_train,learning_rate_fit)
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
allRewards=[]
total_rewards=0
maximumRewardRecorded=0
epochs_mean_reward=[]
total_episode_reward=[]



saver=tf.train.Saver()
if training:
  while epoch<num_epochs+1:
    #state=env.reset()
    # state,stacked_frames=stack_frames(stacked_frames,state,True)
    #episode_num=0
    # rewards_of_episode=[]
    # total_reward_of_batch=[]
 
    states_mb,actions_mb,rewards_of_batch,discounted_rewards_mb,y_mb,num_episode=make_batch(batch_size)
   # values=sess.run([PGNetwork.value],feed_dict={PGNetwork.inputs_: states_mb})

    #print(np.shape(discounted_rewards_mb)[0])
    #mse=tf.reduce_mean(tf.square(error),name="mse")
    sess.run([PGNetwork.fit_opt],feed_dict={PGNetwork.inputs_:states_mb,PGNetwork.values_: y_mb})
    new_fitted_values=sess.run(PGNetwork.value,feed_dict={PGNetwork.inputs_:states_mb})

    #print(np.shape(new_fitted_values)[0])
    advantage=np.zeros_like(discounted_rewards_mb)
    advantage=discounted_rewards_mb-new_fitted_values.ravel()
    #print(advantage)
    #print 
    #print(action_probability_distribution)
   # print(rewards_of_batch)
    mean_reward_epoch=sum(rewards_of_batch)/num_episode
    epochs_mean_reward.append(mean_reward_epoch)
     # total_reward_of_batch.append(np.sum(rewards_of_batch))
    ess.run([PGNetwork.train_opt],feed_dict={PGNetwork.inputs_: states_mb,PGNetwork.actions: actions_mb,PGNetwork.advantage_: advantage})
    # episode_reward=np.sum(rewards_of_episode)
    # total_episode_reward.append(episode_reward)
    epoch+=1 
     #  allRewards.append(total_reward_of_that_batch)

     # mean_reward_of_that_batch=np.divide(total_reward_of_that_batch,batch_size)
     # mean_reward_total.append(mean_reward_of_that_batch)

   # average_reward_of_all_training=np.divide(np.sum(mean_reward_total),epoch)

    #maximumRewardRecorded=np.max(allRewards)


    print("===========")
    print("Epoch: ",epoch,"/",num_epochs)
    print("-----")
   # print("Number of training episodes: {}".format(nb_episodes_mb))
    print("Mean Reward of that batch {}".format(mean_reward_epoch))
   # print("Average Reward of all training: {}".format(average_reward_of_all_training))
  #  print("Max reward for a batch so far: {}".format(maximumRewardRecorded))

   # print("Training Loss: {}".format(loss_))
   # print("Cross Entropy: {}".format(cross))

t=range(num_epochs)
plt.plot(t,epochs_mean_reward)
plt.xlabel("training epochs")
plt.ylabel("average episode reward")
plt.savefig('fdg.pdf')




