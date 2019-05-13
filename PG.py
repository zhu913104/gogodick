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
# DISPLAY_REWARD_THRESHOLD = 1000  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 100   # maximum time step in one episode
RENDER = True  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error

# env = gym.make('CartPole-v0')
# env.seed(1)  # reproducible
# env = env.unwrapped

N_F = 80
N_A = 3

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
        pyautogui.keyDown('w')

    def left(self):
        pyautogui.keyUp('d')
        pyautogui.keyDown('a')
        pyautogui.keyDown('w')

    def right(self):
        pyautogui.keyUp('a')
        pyautogui.keyDown('d')
        pyautogui.keyDown('w')
    
    def reset(self):
        pyautogui.keyDown('r')
        pyautogui.keyUp('d')
        pyautogui.keyUp('a')
        pyautogui.keyUp('w')
        pyautogui.keyUp('r')
        
        
    

        
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
            self.left()
        elif act ==2:
            self.right() 



class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001,nums=1,frame_muti=False):
        self.frame_muti = frame_muti
        if self.frame_muti !=True:
            self.nums = 1
        else:
            self.nums=   nums                       
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features,n_features,self.nums], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

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
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            # log_prob = tf.log(self.acts_prob[0, self.a])
            # exp_v = log_prob * tf.stop_gradient(self.td_error)
            # entropy = -tf.reduce_sum(self.acts_prob * tf.log(self.acts_prob + 1e-5),
            #                                  axis=1, keep_dims=True)  # encourage exploration
            
            # print("self.a",self.a)
            # print("self.acts_prob:",self.acts_prob)
            # print("self.acts_prob[0, self.a]",self.acts_prob[0, self.a])
            # self.exp_v =tf.reduce_mean(ENTROPY_BETA * entropy + exp_v)
            
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        if self.nums==1:
            s = s[np.newaxis, :,:,np.newaxis]
        else :
            s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        if self.nums==1:
            s = s[np.newaxis, :,:,np.newaxis]
        else :
            s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        print(probs.ravel(),np.argmax(probs.ravel()),end='\r')
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


# class Critic(object):
#     def __init__(self, sess, n_features, lr=0.01,nums=1,frame_muti=False):
#         self.sess = sess
#         self.frame_muti = frame_muti
#         if self.frame_muti !=True:
#             self.nums = 1
#         else:
#             self.nums=   nums        

#         self.s = tf.placeholder(tf.float32, [1, n_features,n_features,self.nums], "state")
#         self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
#         self.r = tf.placeholder(tf.float32, None, 'r')

#         with tf.variable_scope('Critic'):
#             c1 = tf.layers.conv2d(
#                 inputs = self.s,
#                 filters = 16,
#                 kernel_size = (8,8),
#                 strides=(4,4),
#                 activation=tf.nn.relu,
#                 kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='c1',
#             )


 


#             c2 = tf.layers.conv2d(
#                 inputs = c1,
#                 filters = 32,
#                 kernel_size = (4,4),
#                 strides=(2,2),
#                 activation=tf.nn.relu,
#                 kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='c2',
#             )


#             fl = tf.layers.flatten(
#                 inputs = c2,
#                 name= 'fl'
#                 )


#             l1 = tf.layers.dense(
#                 inputs=fl,
#                 units=256,  # number of hidden units
#                 activation=tf.nn.relu,  # None
#                 # have to be linear to make sure the convergence of actor.
#                 # But linear approximator seems hardly learns the correct Q.
#                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='l1'
#             )

#             self.v = tf.layers.dense(
#                 inputs=l1,
#                 units=1,  # output units
#                 activation=None,
#                 kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
#                 bias_initializer=tf.constant_initializer(0.1),  # biases
#                 name='V'
#             )

#         with tf.variable_scope('squared_TD_error'):
#             self.td_error = self.r + GAMMA * self.v_ - self.v
#             self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
#         with tf.variable_scope('train'):
#             self.train_op = tf.train.RMSPropOptimizer(lr).minimize(self.loss)

#     def learn(self, s, r, s_):
#         if self.nums==1:
#             s, s_ = s[np.newaxis, :,:,np.newaxis], s_[np.newaxis, :,:,np.newaxis]
#         else:
#             s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
#         v_ = self.sess.run(self.v, {self.s: s_})
#         td_error, _ = self.sess.run([self.td_error, self.train_op],
#                                           {self.s: s, self.v_: v_, self.r: r})
#         return td_error





# sess = tf.Session()


ENVS = env(hwnd, N_F,nums,frame_muti = frame_muti)

# actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A,nums=nums,frame_muti=frame_muti)
# critic = Critic(sess, n_features=N_F, lr=LR_C,nums=nums,frame_muti=frame_muti)     # we need a good teacher, so the teacher should learn faster than the actor
# sess.run(tf.global_variables_initializer())



t = time.time()



value_log = np.array([0,0])
i_episode=0
date = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
LR_A = 0.00001    # learning rate for actor
LR_C = 0.0001    # learning rate for critic

sess = tf.Session()
actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A,nums=nums,frame_muti=frame_muti)
# critic = Critic(sess, n_features=N_F, lr=LR_C,nums=nums,frame_muti=frame_muti)     # we need a good teacher, so the teacher should learn faster than the actor
sess.run(tf.global_variables_initializer())



saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state("saved_networks/PG")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old network weights")



while True:
    s =  ENVS.get_state()
    s=s/255
    count=0
    track_r = []

    
    time.sleep(0.5)
    ENVS.action(0)
    i_episode+=1
    while True:
        
        a = actor.choose_action(s)

        ENVS.action(a)
        s_, r = ENVS.get_state() ,ENVS.get_reword()
        s_=s_/255


        if r==-1: 
            r = -5

        elif r ==0:
            r=0.8

        elif r ==1:
            r=1


        track_r.append(r)
        # td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        ex_v = actor.learn(s, a, r)     # true_gradient = grad[logPi(s,a) * td_error]
        # print('td_error:',td_error[0][0],"REWORD:",r)
        s = s_
        count += 1

        if RENDER:
            cv2.imshow("s1", s)
            k = cv2.waitKey(30)&0xFF #64bits! need a mask
            if k ==27:
                cv2.destroyAllWindows()
                break



        if  count >= MAX_EP_STEPS:
            date_ = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
            ep_rs_sum = sum(track_r)
            ENVS.reset()
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            
            # if running_reward > DISPLAY_REWARD_THRESHOLD: 
            #     RENDER = True  # rendering
            
            value_log = np.vstack([value_log,np.array([i_episode,running_reward])])
            np.save("log/PG"+date,value_log)
            saver.save(sess, 'saved_networks/PG'+date_,global_step=count)
            
            print("episode:", i_episode, "  reward:", int(running_reward),"now  reward:",ep_rs_sum,"---",date)

            break