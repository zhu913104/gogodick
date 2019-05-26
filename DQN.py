import grabscreen
import re
from time import sleep
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import pyautogui
import tensorflow as tf
import datetime


hwnd = grabscreen.FindWindow_bySearch("envs")



np.random.seed(2)
tf.set_random_seed(2)  # reproducible



OUTPUT_GRAPH = False
# DISPLAY_REWARD_THRESHOLD = 1000  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 100   # maximum time step in one episode
MAX_TIME = 86400 #a day 
RENDER = True  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error

# env = gym.make('CartPole-v0')
# env.seed(1)  # reproducible
# env = env.unwrapped

N_F = 80
N_A = 5

# for i in range(4,0,-1):
#     print(i)
#     time.sleep(1)


nums = 1
frame_muti = True

class env(object):
    def __init__(self, hwnd, zoom,num,frame_muti):
        self.hwnd = hwnd
        self.zoom = zoom
        self.num = num
        self.stack = np.array([0])
        self.frame = 0


    def normalization_fram(self,x,maxpix):    
        # y = np.array((x**m)/(255**m)*255,dtype=np.uint8)
        y= np.array((x/maxpix)*255)
        y[y>255]=255
        y = np.array(y,dtype=np.uint8)
        return y

    def forword(self):
        pyautogui.keyUp('d')
        pyautogui.keyUp('a')
        pyautogui.keyUp('s')
        pyautogui.keyDown('w')

    def left_forword(self):
        pyautogui.keyUp('d')
        pyautogui.keyUp('s')
        pyautogui.keyDown('a')
        pyautogui.keyDown('w')

    def right_forword(self):
        pyautogui.keyUp('a')
        pyautogui.keyUp('s')
        pyautogui.keyDown('d')
        pyautogui.keyDown('w')
    
    def reset(self):
        pyautogui.keyDown('r')
        pyautogui.keyUp('s')
        pyautogui.keyUp('d')
        pyautogui.keyUp('a')
        pyautogui.keyUp('w')
        pyautogui.keyUp('r')
    
    def left(self):
        pyautogui.keyUp('w')
        pyautogui.keyUp('d')
        pyautogui.keyDown('a')
        pyautogui.keyDown('s')
        

    def right(self):
        pyautogui.keyUp('w')
        pyautogui.keyUp('a')
        pyautogui.keyDown('d')
        pyautogui.keyDown('s')
        
        
    

        
    def _state(self):
        self.frame = grabscreen.getWindow_Img(self.hwnd)
        self.frame = self.frame[28:,:1600]
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_RGB2GRAY)
        self.frame  =  self.normalization_fram(self.frame,150)
        self.frame = cv2.resize(self.frame,(self.zoom,self.zoom))
        return self.frame

    def get_reword(self):
        rewordpix = self.frame[0,-1]
        if rewordpix == 255:
            self.reword = 1
            return self.reword 
        elif rewordpix ==0:
            # _stop()
            self.reword =  -1
            return self.reword 
        else:
            self.reword = 0
            return self.reword 

    def stack_state(self):
        self._state()
        if self.num ==1:
            pass
        else :
            if self.stack.any()==0:
                self.stack = np.dstack((self.frame,self.frame))
                if self.num>2:
                    for i in range(self.num-2):
                        self.stack = np.dstack((self.stack,self.frame))
            else:
                self.stack = np.dstack((self.frame,self.stack[:,:,:(self.num-1)]))

    def get_state(self):
        self.stack_state()
        if self.num==1:
            return self.frame
        elif frame_muti:
            return self.stack

        else:
            self.stack_m = np.sum(self.stack,axis=2 )
            self.stack_m =self.stack_m/(self.num)
            self.stack_m= np.array(self.stack_m,dtype = np.uint8)
            return self.stack_m



    def action(self,act):
        if act ==0:
            self.forword()
        elif act ==1:
            self.left_forword()
        elif act ==2:
            self.right_forword() 
        elif act==3:
            self.left()
        elif act==4:
            self.right()





class Critic(object):
    def __init__(self, 
                sess, 
                n_features, 
                n_actions=5,
                lr=0.01,
                nums=1,
                frame_muti=False,
                memory_size=3000,
                replace_target_iter=200,
                batch_size=32,
                ):
        self.sess = sess
        self.frame_muti = frame_muti
        if self.frame_muti !=True:
            self.nums = 1
        else:
            self.nums=   nums
        self.n_features = n_features*n_features
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, (n_features**2)*2+2))
        self.s = tf.placeholder(tf.float32, [None, n_features,n_features,self.nums], "state")
        self.s_ = tf.placeholder(tf.float32, [None, n_features,n_features,self.nums], "state")
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.eval = self.build_layers(self.s ,'eval',True)
        self.target = self.build_layers(self.s_ ,'target',False)
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        
        
        t_params = tf.get_collection('target')
        e_params = tf.get_collection('eval')
        self.replace_traning_op =  [tf.assign(t, e) for t, e in zip(t_params, e_params)]


        with tf.variable_scope('squared_TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.eval))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(lr).minimize(self.loss)

            
    def build_layers(self,s, scope, trainable):
        with tf.variable_scope(scope):
            c1 = tf.layers.conv2d(
                inputs = s,
                filters = 16,
                kernel_size = (8,8),
                strides=(4,4),
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='c1',
                trainable=trainable
            )
            c2 = tf.layers.conv2d(
                inputs = c1,
                filters = 32,
                kernel_size = (4,4),
                strides=(2,2),
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='c2',
                trainable=trainable
            )
            fl = tf.layers.flatten(
                inputs = c2,
                name= 'fl'
                
                )
            l1 = tf.layers.dense(
                inputs=fl,
                units=256,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1',
                trainable=trainable
            )
            out = tf.layers.dense(
                inputs=l1,
                units=5,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V',
                trainable=trainable  
            )
            return out
        
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        s,s_=s.reshape(-1),s_.reshape(-1)
        transition = np.hstack((s, [a, r], s_))  #shape=(12802,)
        index = self.memory_counter % self.memory_size   
        self.memory[index, :] = transition
        self.memory_counter += 1

    
    def choose_action(self, s):
        if self.nums==1:
            s = s[np.newaxis, :,:,np.newaxis]
        else :
            s = s[np.newaxis, :]
        probs = self.sess.run(self.eval, {self.s: s})   # get probabilities for all actions
        # print(probs.ravel(),np.argmax(probs.ravel()),end='\r')
        if np.random.uniform() >0.9:
            chooseact =  np.random.choice(np.arange(probs.shape[1]))   # return a int
        else:
            chooseact =  np.argmax(probs.ravel())
        # print((probs.ravel()),chooseact,end='\r')
        # print("action",chooseact,,end='\r')
        return chooseact


    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_traning_op)
            # print('\ntarget_params_replaced')
        
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :] 
        next_observation = batch_memory[:, -self.n_features:].reshape(batch_memory.shape[0],int(self.n_features**0.5),int(self.n_features**0.5),1)  #把1D變4D
        now_observation= batch_memory[:, :self.n_features].reshape(batch_memory.shape[0],int(self.n_features**0.5),int(self.n_features**0.5),1)  #把1D變4D
        q_next, q_eval4next = self.sess.run(
            [self.target, self.eval],
            feed_dict={self.s_: next_observation,    # next observation
                       self.s: next_observation})    # next observation

        q_eval = self.sess.run(self.eval, {self.s: now_observation})
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        

        max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + GAMMA * selected_q_next

        # print("----------------------------")
        # print(q_eval[0],q_next[0],q_target[0])
        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.s:now_observation,
                                                self.q_target: q_target})
        self.learn_step_counter = self.learn_step_counter+1
        return self.cost





# sess = tf.Session()


ENVS = env(hwnd, N_F,nums,frame_muti = frame_muti)

# actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A,nums=nums,frame_muti=frame_muti)
# critic = Critic(sess, n_features=N_F, lr=LR_C,nums=nums,frame_muti=frame_muti)     # we need a good teacher, so the teacher should learn faster than the actor
# sess.run(tf.global_variables_initializer())



t = time.time()



reword_log = np.array([0,0])
loss_log = np.array([0,0])
act_log = np.array([0,0,0,0,0,0])
i_episode=0
date = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
LR_A = 0.00001    # learning rate for actor
LR_C = 0.00001    # learning rate for critic

sess = tf.Session()
critic = Critic(sess, n_features=N_F, lr=LR_C,nums=nums,frame_muti=frame_muti)     # we need a good teacher, so the teacher should  faster than the actor
sess.run(tf.global_variables_initializer())



saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state("saved_networks/dqn10e-5/")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old network weights")



while True:
    s =  ENVS.get_state()
    s=s/255
    track_r = []
    time.sleep(0.5)
    ENVS.action(0)
    act_log_ = np.array([0,0,0,0,0,0])
    while True:
        i_episode+=1
        a=critic.choose_action(s)
        ENVS.action(a)
        s_, r = ENVS.get_state() ,ENVS.get_reword()

        
        s_=s_/255


        if r==-1: 
            if a==3 or a==4:
                r = -0.1
            else:
                r = -1
        elif r ==0:
            if a==3 or a==4:
                r=0
            else:
                r=1
        elif r ==1:
            if a==3 or a==4:
                r=0
            else:
                r=1



        print("action:{}  reword:{:+.2f}".format(a,r),end='\r')
        critic.store_transition(s,a,r,s_)
        # print("--------------------------------------")
        # print(a,r)
        if a==0:
            act_log_[1] +=1 
        elif a==1:
            act_log_[2] +=1 
        elif a==2:
            act_log_[3] +=1
        elif a==3:
            act_log_[4] +=1
        elif a==4:
            act_log_[5] +=1


        track_r.append(r)
        td_error = critic.learn()  # gradient = grad[r + gamma * V(s_) - V(s)]
        loss_log = np.vstack([loss_log,np.array([i_episode,td_error])])
        np.save("log/dqn10e-4/loss/"+date,loss_log)
        # print('td_error:',td_error[0][0],"REWORD:",r)
        s = s_

        if  i_episode % MAX_EP_STEPS ==0:
            ENVS.reset()
            date_ = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
            ep_rs_sum = sum(track_r)
            
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            
            # if running_reward > DISPLAY_REWARD_THRESHOLD: 
            #     RENDER = True  # rendering
            
            reword_log = np.vstack([reword_log,np.array([i_episode,ep_rs_sum])])
            act_log = np.vstack([act_log,act_log_])
            np.save("log/dqn10e-5/action/"+date,act_log)
            np.save("log/dqn10e-5/reword/"+date,reword_log)
            saver.save(sess, 'saved_networks/dqn10e-5/'+date,global_step=i_episode)
            
            print("episode:", i_episode, "  reward:", int(running_reward),"now  reward:",ep_rs_sum,"---",date_)
            print(act_log_[1:])
            break