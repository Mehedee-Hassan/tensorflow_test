
%reset -f 
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set a prediction array place holder , I will set value in run time 
# first place is None ,so I can set any [N dim X 1 ] array
predictor = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# set a prediction array place holder , I will set value in run time 
target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        

# the slope of straight line is A array
A = tf.Variable(tf.zeros(shape=[1,1]))

# the intercept of Y axis is b array
b = tf.Variable(tf.ones(shape=[1,1]))

# the traight line equation 
out= tf.add(tf.matmul(predictor,A),b)

# the loss funcion ,the metrics we will follow when minimizing the error 
loss = tf.reduce_mean(tf.abs(target-out))

# the optimizer function setting learning rate 0.001 as too small will fit properly
opt = tf.train.GradientDescentOptimizer(0.001)

# commanding what to do
train_step = opt.minimize(loss)

# creating tensorflow session
sess = tf.Session()

# initialize all the variable
init = tf.global_variables_initializer()


# run session to initialize the variable 
sess.run(init)


# taking randmom array as test and target array
#randarrayX = np.transpose([np.random.randint(10,40,3)])
#randarrayY = np.transpose([np.random.randint(10,40,3)])

# selecting constant test and target array
randarrayX = np.transpose([[24,36,50]]) 
randarrayY = np.transpose([[24,36,50]])

#print(randarrayX,randarrayY)

# the loss funcion array
lossArray =[]

# taking 100000 training steps ...
for i in range(0,100000):
    
    
    sess.run(train_step,feed_dict={predictor:randarrayX,target:randarrayY})
    losaaray = sess.run(loss,feed_dict={predictor:randarrayX,target:randarrayY})
    
    if i % 10000 == 0:
        print('Step Number' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print(losaaray) 
    
    lossArray.append(losaaray)

# the optimal slope for A
[slope] = sess.run(A)

#the optimal interceptor for b
[y_intercept] = sess.run(b)

# plotting the real x,y 
plt.plot(randarrayX,randarrayY,'o',label="main")

# for every real x we setup our slope and intercept
# this is the predition for every X value ,
# we have trained our model and found the optimal slope and y axis intercept
test_fit =[]
for i in randarrayX:
    test_fit.append(slope*i + y_intercept)

#plot real points Vs trained(fitted) line 
    
plt.plot(randarrayX,test_fit,'r-',label='prediction')
plt.legend(loc='lower right')
plt.show()


# Plot loss over time
plt.plot(lossArray, 'r-')
plt.title('L1 Loss per loop')
plt.xlabel('Loop')
plt.ylabel('L1 Loss')
plt.show()

