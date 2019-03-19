import grabscreen
import re
from time import sleep
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import pyautogui
import tensorflow as tf



hwnd = grabscreen.FindWindow_bySearch("envs")

log = np.array([0,0,0])

np.random.seed(10)
tf.set_random_seed(10)  # reproducible


t = time.time()
OUTPUT_GRAPH = False
MAX_EPISODE = 3000000
DISPLAY_REWARD_THRESHOLD = 1000  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 20   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

# env = gym.make('CartPole-v0')
# env.seed(1)  # reproducible
# env = env.unwrapped

N_F = 80
N_A = 3
act = "forword"
for i in range(4,0,-1):
    print(i)
    time.sleep(1)


t= 0
nums = 4
zoom= 500


class env(object):
    def __init__(self, hwnd, zoom,num):
        self.hwnd = hwnd
        self.zoom = zoom
        self.num = num
        self.stack = np.array([0])
        self.frame = 0


    def normalization_fram(self,x,m):    
        y = np.array((x**m)/(255**m)*255,dtype=np.uint8)
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
    
    def _stop(self):
        pyautogui.keyUp('d')
        pyautogui.keyUp('a')
        pyautogui.keyUp('w')
        time.sleep(1)
        pyautogui.keyDown('w')
        
    def _state(self):
        self.frame = grabscreen.getWindow_Img(self.hwnd)
        self.frame = self.frame[28:,:1600]
        self.frame = cv2.cvtColor(self.frame,cv2.COLOR_RGB2GRAY)
        self.frame  =  self.normalization_fram(self.frame,0.6)
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
                self.stack = np.stack((self.frame,self.frame))
                if self.num>2:
                    for i in range(self.num-2):
                        self.stack = np.vstack((self.stack,[self.frame]))
            else:
                self.stack = np.vstack(([self.frame],self.stack[:(self.num-1),:,:]))

    def get_state(self):
        self.stack_state()
        if self.num==1:
            return self.frame
        else:
            self.stack_m = np.sum(self.stack,axis=0 )
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
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features,n_features,1], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):

            c1 = tf.layers.conv2d(
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
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :,:,np.newaxis]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :,:,np.newaxis]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess


        self.s = tf.placeholder(tf.float32, [1, n_features,n_features,1], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            c1 = tf.layers.conv2d(
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
                units=256,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :,:,np.newaxis], s_[np.newaxis, :,:,np.newaxis]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error





sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

ENVS = env(hwnd, N_F,nums)



##################################################################################################




##########################################################################################################################
for i_episode in range(MAX_EPISODE):
    s =  ENVS.get_state()
    s=s/255
    t = 0
    track_r = []
    done =False

    
    print("set")
    time.sleep(2)
    ENVS.action(0)
    print("ON")
    while True:
        
        a = actor.choose_action(s)
        # a = np.random.randint(3)
        ENVS.action(a)
        s_, r = ENVS.get_state() ,ENVS.get_reword()
        s_=s_/255
        if a ==0:
            act = 'forward'
        elif a ==1:
            act = 'left'
        elif a==2:
            act = 'right'
        print("action: ",act)


        if r==-1: 
            r = -20
            done = True
        elif r ==0:
            r=-1
            done =False
        else:
            r=2
            done =False

        track_r.append(r)
        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if RENDER:
            cv2.imshow("s", ENVS.frame)
            k = cv2.waitKey(30)&0xFF #64bits! need a mask
            if k ==27:
                cv2.destroyAllWindows()
                break



        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward),"now  reward:",ep_rs_sum)
            break




