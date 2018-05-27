from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import time
import datetime

path='drink_data/'
#要训练的图片保存的地址
#模型保存地址

#将所有的图片resize成w*c,然后为RGB三色彩通道
w=200
h=150
c=3


#读取图片
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    #获取各类数据的名字,cate为所有该路径下的文件夹名字
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        #print(idx,folder)
        for im in glob.glob(folder+'/*.jpg'):
            #输出读取了的模型的图片名
            #print('reading the images:%s'%(im))
            img=io.imread(im)
            tf.image.flip_left_right(img) #进行了旋转，
            img=transform.resize(img,(w,h))
            #改变其尺寸的大小
            imgs.append(img)
            #将图片与标签分开加入到数组当中
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data,label=read_img(path)


#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
#把顺序打乱
data=data[arr]
label=label[arr]


#将所有数据分为训练集和验证集,按照比例
ratio=0.8
s=np.int(num_example*ratio) #s为训练数据的大小
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

#-----------------构建网络----------------------
#占位符,tensorflow中是通过可以通过后面喂数组的时候选择开多大的数组，这里只需要占位符先占着
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')


def inference(input_tensor, train, regularizer):
    #第一层5*5大小的卷积核，把高度从3 -> 32
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"): #池化到一半
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):#再进行一遍缩小
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#38*50

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope("layer6-pool3"):#再缩小一次,总共用2*2池化了4次
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')#19*25

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight",[3,3,128,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
    #最后这里得到的为6*6*128的网络结构
    with tf.name_scope("layer8-pool4"):#缩小总共4次,10*13
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        nodes = 10*13*128
        reshaped = tf.reshape(pool4,[-1,nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    #通过全连接降下来
    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)
    #这里为最后一层，输出结果为7个
    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 7],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [7], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit
#激活函数relu改成tanh,可能对你的网络是个优化方法
#---------------------------网络结束---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.001)
logits = inference(x,False,regularizer)

#一个为了以后调用方便
#(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')

#这里定义损失函数
loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) #这里调整每次的步长,步长太小，验证集达不到效果,很快就过拟合
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=True):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        #if shuffle:
        excerpt = indices[start_idx:start_idx + batch_size]
        # else:
        #     excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


#训练的次数，太多会过拟合的
Times=5 #这里是调整训练次数
batch_size=64  #这里上batch_size
saver=tf.train.Saver()
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#下面两句话为调用已有的模型，去再次训练
#saver.restore(sess, model_path)
start_time = datetime.datetime.now()
print('开始训练时间为:  ' ,start_time.strftime('%Y-%m-%d %H:%M:%S '))

for epoch in range(Times):
    model_path = "../model/model.ckpt"

    # 加下面2行是调用之前的模型
    #saver.restore(sess, model_path)
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    #这里是把所有的数据都训练一遍，然后输出一次结果
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    current_time = datetime.datetime.now()
    print('现在时间是:',current_time.strftime('%Y-%m-%d %H:%M:%S '),'第',epoch,'次的结果为 : '  )
    print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))

    #saver = tf.train.Saver()
    #SAMEation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   SAMEation loss: %f" % (np.sum(val_loss)/ n_batch))
    print("   SAMEation acc: %f" % (np.sum(val_acc)/ n_batch))
    saver.save(sess, model_path)
    print('保存模型')
sess.close()