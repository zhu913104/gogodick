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

log = np.array([0,0,0])

np.random.seed(2)
tf.set_random_seed(2)  # reproducible



OUTPUT_GRAPH = False
MAX_EP_STEPS = 10   # maximum time step in one episode
RENDER = True  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error


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






class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001,nums=1,frame_muti=False):
        self.frame_muti = frame_muti
        if self.frame_muti !=True:
            self.nums = 1
        else:
            self.nums=   nums                       
        self.sess = sess
        self.n_actions = n_actions
        self.s = tf.placeholder(tf.float32, [None, n_features,n_features,self.nums], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.reword = tf.placeholder(tf.float32, None, "reword")  # TD_error
        self.ep_ss,self.ep_as,self.ep_rs =[],[],[] 

        with tf.variable_scope('Actor'):

            c1= tf.layers.conv2d(
                inputs = self.s,
                filters = 16,
                kernel_size = (8,8),
                strides=(4,4),
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='c1',
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
            )


            fl = tf.layers.flatten(
                inputs = c2,
                name= 'fl'
                )

            l1 = tf.layers.dense(
                inputs=fl,
                units=256,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=self.n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            # neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.acts_prob, labels=self.a)   # this is negative log of chosen action
            # log_prob = tf.log(self.acts_prob[0, self.a])
            neg_log_prob = tf.reduce_sum(-tf.log(self.acts_prob)*tf.one_hot(self.a, self.n_actions), axis=1)
            self.exp_v = tf.reduce_mean(neg_log_prob * self.reword)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(lr).minimize(self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self):
        nl_reword = self.normalization_reword()
        
        

        # if self.nums==1:
        #     self.ep_ss = self.ep_ss[np.newaxis, :,:,np.newaxis]
        # else :
        #     self.ep_ss = self.ep_ss[np.newaxis, :]
        self.ep_ss = self.ep_ss[:,:,:,np.newaxis]

        feed_dict = {self.s: self.ep_ss, self.a: self.ep_as, self.reword: nl_reword}
        # print(feed_dict)
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        self.ep_ss,self.ep_as,self.ep_rs =[],[],[] 
        return exp_v

    def choose_action(self, s):
        if self.nums==1:
            s = s[np.newaxis, :,:,np.newaxis]
        else :
            s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        # print(probs.ravel(),np.argmax(probs.ravel()),end='\r')
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
    
    def store_transition(self,s,a,r):
        if self.ep_ss == []:
            self.ep_ss=np.array([s])
            # print()
        else:
            self.ep_ss = np.concatenate([self.ep_ss,[s]])
        self.ep_as.append(a)
        
        self.ep_rs.append(r)

    def normalization_reword(self):
        nl_reword = np.array(self.ep_rs)
        nl_reword[:] = np.mean(nl_reword)
        return nl_reword




ENVS = env(hwnd, N_F,nums,frame_muti = frame_muti)
t = time.time()
value_log = np.array([0,0])
loss_log = np.array([0,0])
act_log = np.array([0,0,0,0,0,0])
i_episode=0
date = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
LR_A = 0.00001    # learning rate for actor
LR_C = 0.0001    # learning rate for critic

sess = tf.Session()
actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A,nums=nums,frame_muti=frame_muti)
sess.run(tf.global_variables_initializer())


#讀取之前儲存的參數
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state("saved_networks/PG/")
if checkpoint and checkpoint.model_checkpoint_path:
    # saver.restore(sess, checkpoint.model_checkpoint_path)
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
        a = actor.choose_action(s)

        ENVS.action(a)
        s_, r = ENVS.get_state() ,ENVS.get_reword()
        s_=s_/255

        if r==-1: 
            if a==3 or a==4:
                r = 0.1
            else:
                r = -1
        elif r ==0:
            if a==3 or a==4:
                r=0
            else:
                r=0.5
        elif r ==1:
            if a==3 or a==4:
                r=0
            else:
                r=1

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
        actor.store_transition(s,a,r)

        s = s_

        print("action:{}  reword:{:+.2f}                ".format(a,r),end='\r')
        # if RENDER:
        #     cv2.imshow("s1", s)
        #     k = cv2.waitKey(30)&0xFF #64bits! need a mask
        #     if k ==27:
        #         cv2.destroyAllWindows()
        #         break
        if i_episode%10==0:
            ex_v = actor.learn()     # true_gradient = grad[logPi(s,a) * td_error]
            print("action:{}  reword:{:+.2f}  learning".format(a,r),end='\r')


        if  i_episode % 100 ==0:
            date_ = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
            ep_rs_sum = sum(track_r)
            # print(actor.normalization_reword())
            
            
            ENVS.reset()
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            
            # if running_reward > DISPLAY_REWARD_THRESHOLD: 
            #     RENDER = True  # rendering
            loss_log = np.vstack([loss_log,np.array([i_episode,ex_v])])
            value_log = np.vstack([value_log,np.array([i_episode,ep_rs_sum])])
            act_log = np.vstack([act_log,act_log_])
            np.save("log/PG/action/"+date,act_log)
            np.save("log/PG/reword/"+date,value_log)
            np.save("log/PG/loss/"+date,loss_log)
            saver.save(sess, 'saved_networks/PG/'+date_,global_step=i_episode)
            
            print("episode:", i_episode, "  reward:", int(running_reward),"now  reward:",ep_rs_sum,"---",date_)
            print(act_log_[1:])
            break