


import numpy as np
import matplotlib.pyplot as plt

K=15;cf=100;ch=2;
N=200;
state=[];
T=500
thre=0.0000001;
discount_factor=0.95
a=["dispatch","not dispatch"];
p=[];


for i in range(N+1):
    state.append(i)
    p.append([]);
    for j in range(2):
        p[i].append([])
for i in state:
    for j in a:
        if j=="not dispatch":
            
            for l in range(1,6,1):
                if i+l<=N:
                    nextstate=i+l
                    prob=1.0/5
                    reward=(i+float(l)/2)*ch##for customers arriving in the coming interal, we count the  expected waiting time as 1/2 of the length of the interval
                elif i+l>N:
                    nextstate=N
                    prob=1.0/5
                    reward=(i+float(N-i)/2)*ch
             
                p[i][0].append([nextstate, prob,reward]);
        elif j=="dispatch":         
            for l in range(1,6,1):
                nextstate=max(i-K,0)+l;
                prob=1.0/5
                reward=(max(i-K,0)+float(l)/2)*ch+cf
                p[i][1].append([nextstate, prob,reward])



                    
            

def one_step_lookahead(state,state_index, V):
    A=np.zeros(len(a))
    for action in range(len(a)):
        for nextstate,prob,reward in p[state_index][action]:
            for i in range(len(state)):
                if nextstate==state[i]:
                    A[action]+=prob*(reward+discount_factor*V[i])
            
    return A

##enumeration:
def enumeration():
    V_t=np.zeros(len(state))
    V_tplus=np.zeros(len(state))
    for t in range(T,-1,-1):
        for s in state:
            A=one_step_lookahead(state,s,V_tplus)
            V_t[s]=np.min(A)
        V_tplus=V_t
    return V_t
V=enumeration()
print("optimal value function by enumeration:")
plt.plot(state,V)
###value iteration:       
def value_iteration():
    V=np.zeros(len(state))
    
    while True:
        delta=0
        for s in state:
            A=one_step_lookahead(state,s,V)
            delta=max(delta,np.abs(np.min(A)-V[s]))
            V[s]=np.min(A)
        if delta<thre:
            break
        
        
    policy=np.zeros([len(state),len(a)])
    for s in state:
        A=one_step_lookahead(state,s,V)
        best_action=np.argmin(A)
        policy[s,best_action]=1.0
    return policy,V

policy,v=value_iteration()
print("optimal value function by value iteration:")
plt.plot(state,v)

##policy iteration
def policy_eval_fn(policy):
    
    V=np.zeros(len(state))
    while True:
        delta=0
        for s in state:
            v=0
            for a, action_prob in enumerate(policy[s]):
                for nextstate,prob,reward in p[s][a]:
                    v+= action_prob*prob*(reward+discount_factor*V[nextstate])
            delta=max(delta,np.abs(v-V[s]))
            V[s]=v
        if delta<thre:
            break
    return np.array(V)


def policy_improvement():
    policy=np.ones([len(state),len(a)])/len(a)

    while True:
        
        V=policy_eval_fn(policy)
        policy_stable=True
        for s in range(len(state)):
            chosen_a=np.argmax(policy[s])
            best_action_value=one_step_lookahead(state,s,V)
            best_a=np.argmin(best_action_value)
            if chosen_a!=best_a:
                policy_stable=False
            policy[s]=np.eye(len(a))[best_a]
        if policy_stable:
            return policy,V
policy,v=policy_improvement()
besta=[];
for s in range(len(state)):
    if policy[s][0]==np.max(policy[s]):
        besta.append(0);
    elif policy[s][1]==np.max(policy[s]):
        besta.append(1);
##plot the optimal action
plt.plot(state,besta)



            
            

        

    
    
        
                 
    