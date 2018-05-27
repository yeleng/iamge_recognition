from skimage import io,transform
import tensorflow as tf
import numpy as np
image_size=26
#C:\Users\Administrator\Desktop\ML\bottle\drink_data\fenda
#这里为图片的地址
path = "drink_data/"

#flower_dict = {0:'dasiy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}
#
classes = { 0:'fenda',1:'lemon_tea',2:'milk_deluxe',3:'mizone',4:'nongfu_spring',5:'red_bull',6:'youyue'}

w=200
h=150
c=3

def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)
data = []
#str函数的作用是把数字转换成一个字符串
for i in range(image_size):
    all_path = path+str(i+1)+'.jpg'
    #注意这个循环中数组从0开始的
    data1 = read_one_image(all_path)
    #得到一个数据
    data.append(data1)
    #然后把数组传给一个数组中
with tf.Session() as sess:
    #data = []
    #data1 = read_one_image(path1)
    #data2 = read_one_image(path2)
    #data3 = read_one_image(path3)
    #data.append(data1)
    #data.append(data2)
    #data.append(data3)
    #这里调用meta数据
    saver = tf.train.import_meta_graph('model/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('model/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)

    #打印出预测矩阵
    #print(classification_result)
    #打印出预测矩阵每一行最大值的索引
    #print(tf.argmax(classification_result,1).eval())
    #根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("第",i+1,"个瓶子预测结果是:"+classes[output[i]])