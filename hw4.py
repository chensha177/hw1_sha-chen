
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
def sample_jester_data(file_name, context_dim = 32, num_actions = 8, num_contexts = 19181,
shuffle_rows=True, shuffle_cols=False):
# =============================================================================
# """Samples bandit game from (user, joke) dense subset of Jester dataset.
# Args:
# file_name: Route of file containing the modified Jester dataset.
# context_dim: Context dimension (i.e. vector with some ratings from a user).
# num_actions: Number of actions (number of joke ratings to predict).
# num_contexts: Number of contexts to sample.
# shuffle_rows: If True, rows from original dataset are shuffled.
# shuffle_cols: Whether or not context/action jokes are randomly shuffled.
# Returns:
# dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
# opt_vals: Vector of deterministic optimal (reward, action) for each context.
# """
# =============================================================================
  np.random.seed(0)
  with tf.gfile.Open(file_name, 'rb') as f:
    dataset = np.load(f)
  if shuffle_cols:
    dataset = dataset[:, np.random.permutation(dataset.shape[1])]
  if shuffle_rows:
    np.random.shuffle(dataset)
  dataset = dataset[:num_contexts, :]

  assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'
  opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
  opt_rewards = np.array([dataset[i, context_dim + a] for i, a in enumerate(opt_actions)])
  return dataset, opt_rewards, opt_actions


k=32
n_a=8
D=np.random.random((n_a,k))-0.5

#theta*context=reward

# =============================================================================
# print(data_sampled)
# for i in range(n):
#     dat[i]=dataset[i][32:end]
#     md[i]=np.max(dat[i])
# =============================================================================
# =============================================================================
# print(data_sampled[0])
# print(data_sampled[1].shape)
# print(data_sampled[2].shape)
# =============================================================================
n=18000
choices=np.zeros(n)
rewards=np.zeros(n)
b=np.zeros((n_a,k))
A=np.zeros((n_a,k,k))
for a in range(0,n_a):
	A[a]=np.identity(k)
th_hat=np.zeros((n_a,k))
p=np.zeros(n_a)

tn=19181-18000
reg=[]

alph=0.2;


##x[t,a] is the context of the person who pulls the arm a,
##reward=x[t,a]*theta_a
##theta=inverse(D_a'D_a+I)D_a'y_a

dataset, opt_rewards, opt_actions=sample_jester_data("jester_data_40jokes_19181users.npy",context_dim = 32, num_actions = 8, num_contexts = 19181,
shuffle_rows=True, shuffle_cols=False)
print(len(dataset))
reg=[];
for i in range(n):
  x_i=dataset[i][0:k]

  for a in range(0,n_a):
    A_inv=np.linalg.inv(A[a])
    th_hat[a]=A_inv.dot(b[a])##ontext x_[t,a]*reward
    ta=x_i.dot(A_inv).dot(x_i)
    a_upper_ci=alph*np.sqrt(ta)
    a_mean=th_hat[a].dot(x_i)
    p[a]=a_mean+a_upper_ci ##x_{t,a}*theta_hat[a]+alpha*sqrt(x_{t,a}inv(A_a)x_{t,a}), where A_a=D_a'*D_a+I
  p=p+(np.random.random(len(p))*0.00000001)
  recd=p.argmax()
  choices[i]=recd
  rewards[i]=dataset[i][k+recd]##real reward
  A[int(choices[i])]+=np.outer(x_i,x_i)
  b[int(choices[i])]+=rewards[i]*x_i
##evaluation
  t_p=np.zeros(n_a)
  regret=0;
  t_choices=np.zeros(tn)
  t_rewards=np.zeros(tn)
  for j in range(tn):
    x_i=dataset[18000+j][0:k]
    for a in range(0,n_a):
      A_inv=np.linalg.inv(A[a])
      ta=x_i.dot(A_inv).dot(x_i)
      a_upper_ci=alph*np.sqrt(ta)
      a_mean=th_hat[a].dot(x_i)
      t_p[a]=a_mean+a_upper_ci ##x_{t,a}*theta_hat[a]+alpha*sqrt(x_{t,a}inv(A_a)x_{t,a}), where A_a=D_a'*D_a+I
    t_p=t_p+(np.random.random(len(t_p))*0.00000001)
    recd=t_p.argmax()
    t_choices[j]=recd
    t_rewards[j]=dataset[18000+j][k+recd]##real reward    
    regret+=opt_rewards[18000+j]-t_rewards[j]
  reg.append(regret)

  







t=range(len(reg))
plt.plot(t,reg)
plt.xlabel('training trial')
plt.ylabel("regret")
plt.savefig('hw41.pdf')

    
 
    

    


