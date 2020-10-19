

import numpy as np
import matplotlib.pyplot as plt
ch1=np.array([1,1.5,2,2.5,3])
N=100;
K=15;cf=100;
staten=[];

discount_factor=0.95
a=["dispatch","not dispatch"];
thre=0.0000001;
T=500


        
for i in range(N+1):
    for j in range(N+1):
        for k in range(N+1):
            for l in range(N+1):
                for m in range(N+1):
                    staten.append([i,j,k,l,m]);
## given state and action, store next state, the probability of reaching it and the associated reward
print(staten);
p=[];
#initialize p
for i in range(len(staten)):
    p.append([]);
    for j in range(2):
        p[i].append([])

for i in range(len(staten)):
    for j in a:
        if j=="not dispatch":
            
            for l in range(1,6,1):
                for k in range(1,6,1):
                    for m in range(1,6,1):
                        for n in range(1,6,1):
                            for o in range(1,6,1):
                                reward=0;
                                nextstaten=[];
                                for t in range(5):
                                    nextstaten.append(0)
                                if staten[i][0]+l<=N:
                                    nextstaten[0]=staten[i][0]+l  
                                    reward+=(staten[i][0]+float(l)/2)*ch1[0]
                                elif staten[i][0]+l>N:#given the number of customers in a class cannot exceed N
                                    nextstaten[0]=N 
                                    reward+=(staten[i][0]+float(N-staten[i][0])/2)*ch1[0]
                                    
                                if staten[i][1]+k<=N:
                                    nextstaten[1]=staten[i][1]+k
                                    
                                    reward+=(staten[i][1]+float(k)/2)*ch1[1]
                                elif staten[i][1]+k>N:
                                    nextstaten[1]=N
                                    
                                    reward+=(staten[i][1]+float(N-staten[i][1])/2)*ch1[1]
                                    
                                if staten[i][2]+m<=N:
                                    nextstaten[2]=staten[i][2]+k
                                    
                                    reward+=(staten[i][2]+float(m)/2)*ch1[2]
                                elif staten[i][2]+m>N:
                                    nextstaten[2]=N
                                    
                                    reward+=(staten[i][2]+float(N-staten[i][2])/2)*ch1[2]
                                    
                                if staten[i][3]+n<=N:
                                    nextstaten[3]=staten[i][3]+n
                                    
                                    reward+=(staten[i][3]+float(n)/2)*ch1[3]
                                elif staten[i][3]+n>N:
                                    nextstaten[3]=N
                                    
                                    reward+=(staten[i][3]+float(N-staten[i][3])/2)*ch1[3]
                                    
                                if staten[i][4]+o<=N:
                                    nextstaten[4]=staten[i][4]+o
                                    
                                    reward+=(staten[i][4]+float(o)/2)*ch1[4]
                                elif staten[i][4]+o>N:
                                    nextstaten[4]=N
                                    
                                    reward+=(staten[i][4]+float(N-staten[i][4])/2)*ch1[4]
                                prob=pow(1.0/5,5)   
                                p[i][0].append([nextstaten, prob,reward]);

             
               
        elif j=="dispatch":
            s_s=[];
            for t in range(5):
                s_s.append(0)
        ## because the objective is to minimize the cost, it is optimal to first assign the customers that has the highest waiting cost to the shuttle                            
            s_s[4]=max(0,staten[i][4]-K);
            s_s[3]=max(0,staten[i][3]-max(K-staten[i][4],0)); #if K-staten[i][4]<0, then it means that there is no available capacity for the classes that are lower than class 4
            s_s[2]=max(0,staten[i][2]-max(K-staten[i][4]-staten[i][3],0));
            s_s[1]=max(0,staten[i][1]-max(K-staten[i][4]-staten[i][3]-staten[i][2],0));
            s_s[0]=max(0,staten[i][0]-max(K-staten[i][4]-staten[i][3]-staten[i][2]-staten[i][1],0));
            
            for l in range(1,6,1):
                for k in range(1,6,1):
                    for m in range(1,6,1):
                        for n in range(1,6,1):
                            for o in range(1,6,1):
                                reward=0;
                                nextstaten=[];
                                for t in range(5):
                                    nextstaten.append(0)
                                nextstaten[0]=s_s[0]+l;
                                reward+=(s_s[0]+float(l)/2)*ch1[0]
                                if s_s[1]+k<=N:
                                    nextstaten[1]=s_s[1]+k         
                                    reward+=(s_s[1]+float(k)/2)*ch1[1]
                                elif s_s[1]+k>N:
                                    nextstaten[1]=N  
                                    reward+=(s_s[1]+float(N-s_s[1])/2)*ch1[1]
                                if s_s[2]+m<=N:
                                    nextstaten[2]=s_s[2]+m         
                                    reward+=(s_s[2]+float(m)/2)*ch1[2]
                                elif s_s[2]+m>N:
                                    nextstaten[2]=N
                                    reward+=(s_s[2]+float(N-s_s[2])/2)*ch1[2]
                                if s_s[3]+n<=N:
                                    nextstaten[3]=s_s[3]+n           
                                    reward+=(s_s[3]+float(n)/2)*ch1[3]
                                elif s_s[3]+n>N:
                                    nextstaten[3]=N
                                    reward+=(s_s[3]+float(N-s_s[3])/2)*ch1[3]
                                if s_s[4]+o<=N:
                                    nextstaten[4]=s_s[4]+o           
                                    reward+=(s_s[4]+float(o)/2)*ch1[4]
                                elif s_s[4]+o>N:
                                    nextstaten[4]=N
                                    reward+=(s_s[4]+float(N-s_s[4])/2)*ch1[4]
                                    
                                reward+=cf;
                                prob=pow(1.0/5,5)
                                p[i][1].append([nextstaten, prob,reward])
print(p)
#                            
def one_step_lookahead(state,state_index, V):
    A=np.zeros(len(a))
    for action in range(len(a)):
        for nextstate,prob,reward in p[state_index][action]:
            for i in range(len(state)):
                if nextstate==state[i]:
                    A[action]+=prob*(reward+discount_factor*V[i])
            
    return A
                   
def enumeration(state):
    V_t=np.zeros(len(state))
    V_tplus=np.zeros(len(state))
    for t in range(T,-1,-1):
        for s in range(len(state)):
            A=one_step_lookahead(state,s,V_tplus)
            V_t[s]=np.min(A)
        V_tplus=V_t
    return V_t
V=enumeration(staten)
print("optimal value function by enumeration:")
##plot the optimal value function when fixing the number of customers of the class whose ch>1
n=3;
v_state=[];
for i in range(N+1):
    v_state.append(0);
for i in range(len(staten)):
    if staten[i][1]==n & staten[i][2]==n & staten[i][3]==n & staten[i][4]==n:
        k=staten[i][0];
        v_state[k]=V[k];
i_state=[];       
for s in range(N+1):
    i_state.append(s);
plt.plot(i_state,v_state) 


def value_iteration(state):
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
n=3;
v_state=[];
for i in range(N+1):
    v_state.append(0);
for i in range(len(staten)):
    if staten[i][1]==n & staten[i][2]==n & staten[i][3]==n & staten[i][4]==n:
        k=staten[i][0];
        v_state[k]=v[k];
i_state=[];       
for s in range(N+1):
    i_state.append(s);

print("optimal value function by value iteration:")
plt.plot(i_state,v_state) 

#policy iteration
def policy_eval_fn(state,policy):
    
    V=np.zeros(len(state))
    while True:
        delta=0
        for s in range(len(state)):
            v=0
            for a, action_prob in enumerate(policy[s]):
                for nextstate,prob,reward in p[s][a]:
                    v+= action_prob*prob*(reward+discount_factor*V[nextstate])
            delta=max(delta,np.abs(v-V[s]))
            V[s]=v
        if delta<thre:
            break
    return np.array(V)


def policy_improvement(state):
    policy=np.ones([len(state),len(a)])/len(a)

    while True:
        
        V=policy_eval_fn(staten,policy)
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
##find the best action at each state 
for s in range(len(staten)):
    if policy[s][0]==np.max(policy[s]):
        besta.append(0);
    elif policy[s][1]==np.max(policy[s]):
        besta.append(1);

##plot the best action given the number of customers whose ch>1
n=3;
v_state=[];
for i in range(N+1):
    v_state.append(0);
for i in range(len(staten)):
    if staten[i][1]==n & staten[i][2]==n & staten[i][3]==n & staten[i][4]==n:
        k=staten[i][0];
        v_state[k]=besta[k];
i_state=[];       
for s in range(N+1):
    i_state.append(s);
plt.plot(i_state,v_state)

            



         