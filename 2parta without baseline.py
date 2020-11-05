import tensorflow.compat.v1 as tf
import random

from collections import deque 
import time
import matplotlib.pyplot as plt
import numpy as np
#import tensorflow_probability as tfp
import gym
import os
#os.environ['COLAB_SKIP_TPU_AUTH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()
env = gym.make("CartPole-v0")
state_size=4
action_size = 2
print(env.action_space.n)

possible_actions=np.identity(action_size,dtype=int).tolist()
learning_rate=0.0001
num_epochs=300
batch_size=800
training=True
epoch=1
##environment hyperparameters

class PGNetwork:
  def __init__(self,state_size,action_size,learning_rate,name="PGNetwork"):
    self.state_size=state_size
    self.action_size=action_size
    self.learning_rate=learning_rate
    with tf.variable_scope("name"):
      with tf.name_scope("input"):
        self.inputs_=tf.placeholder(tf.float32,[None,state_size],name="inputs_")
        self.actions=tf.placeholder(tf.int32,[None,action_size],name="actions")
        self.discounted_episode_rewards_=tf.placeholder(tf.float32,[None], name="discounted_episode_rewards_")
      with tf.name_scope("conv1"):
        self.fc1 = tf.layers.dense(inputs = self.inputs_, units = 32,activation = tf.nn.relu, kernel_initializer =  tf.initializers.glorot_uniform(),name = 'fc1')
        self.fc2 = tf.layers.dense(inputs = self.fc1, units = 64, activation = tf.nn.relu,kernel_initializer = tf.initializers.glorot_uniform(), name = 'fc2')
        self.fc3 = tf.layers.dense(inputs = self.fc2, units = 16, activation = tf.nn.relu,kernel_initializer = tf.initializers.glorot_uniform(),name = 'fc3')
        
        self.logits = tf.layers.dense(inputs = self.fc3, units = action_size, activation = None,kernel_initializer = tf.initializers.glorot_uniform(),name = 'logits')

      with tf.name_scope("softmax"):
        self.action_distribution = tf.nn.softmax(self.logits)
      #   inputs=Input(shape=self.state_size)
    
      #   self.x=tf.keras.layers.conv2d(1,4,activation='relu',input_shape=input_shape[1:])(inputs)
      #   self.y=tf.keras.layers.Dense(24,activation='relu')(self.x)
      #   self.logits =tf.keras.layers.Dense(2,activation='softmax')(self.y)
        
      # with tf.name_scope("softmax"):
      #   self.action_distribution=tf.nn.softmax(self.logits)
      with tf.name_scope("loss"):
        self.cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.actions)
        self.weighted_negative_likelihoods=tf.multiply(self.cross_entropy,self.discounted_episode_rewards_)
        self.loss=tf.reduce_mean(self.weighted_negative_likelihoods)
      with tf.name_scope("train"):
        self.optimizer=tf.train.RMSPropOptimizer(self.learning_rate)
        self.train_opt=self.optimizer.minimize(self.loss)


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
  states,actions,reward_of_batch,reward_of_episode,discounted_rewards=[],[],[],[],np.array([])
  state=env.reset()
  batch=0
  while True:
    action_probability_distribution=sess.run(PGNetwork.action_distribution,feed_dict={PGNetwork.inputs_:[state]})
    #print(action_probability_distribution)
    action=np.random.choice(range(action_probability_distribution.shape[1]),p=action_probability_distribution.ravel())
     
    next_state,reward,done,info=env.step(action)
    states.append(state)
    action_onehot=np.zeros(action_size)
    action_onehot[action]=1
    actions.append(action_onehot)
    reward_of_episode.append(reward)
    #next_state,stacked_frames=stack_frames(stacked_frames,next_state,False)

    if not done:
      state=next_state
    else:

      state=env.reset()
      reward_of_batch+=reward_of_episode
      discounted_rewards=np.concatenate((discounted_rewards,discount_rewards(reward_of_episode,gamma=0.95,normalization=False)))
      #state,stacked_frames=stack_frames(stacked_frames,state,True)
      batch+=len(reward_of_episode)
      reward_of_episode=[]
      num_episode += 1   
      if batch>batch_size:
        break

  return states,actions,reward_of_batch,np.array(discounted_rewards),num_episode
  


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
PGNetwork=PGNetwork(state_size,action_size,learning_rate)
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
 
    states_mb,actions_mb,rewards_of_batch,discounted_rewards_mb,num_episode=make_batch(batch_size)
   # print(rewards_of_batch)
    mean_reward_epoch=sum(rewards_of_batch)/num_episode
    epochs_mean_reward.append(mean_reward_epoch)
     # total_reward_of_batch.append(np.sum(rewards_of_batch))
    loss_,cross,_=sess.run([PGNetwork.loss,PGNetwork.cross_entropy,PGNetwork.train_opt],feed_dict={PGNetwork.inputs_: states_mb,PGNetwork.actions: actions_mb,PGNetwork.discounted_episode_rewards_: discounted_rewards_mb})
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
plt.savefig('b.pdf')




