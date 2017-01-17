
import tensorflow as tf
import pdb
import numpy as np
import BasicConvLSTMCell
import layer_def as ld
from data_handler import *
from params import *
from loss_functions import *
def sample_function(GT_past,GT_future,G_future):
    label = []
    input_ = []
    for i in range(GT_past.shape[0]):
        neg_or_pos = np.random.random_integers(0,1,1)
        if neg_or_pos == 0: #negative
            label.append([0.0,1.0])
            input_.append(np.concatenate([GT_past[i,:,:,:,:],G_future[i,:,:,:,:]],0))
        else:
            label.append([1.0,0.0])
            input_.append(np.concatenate([GT_past[i,:,:,:,:],GT_future[i,:,:,:,:]],0))

    input_ = np.asarray(input_)
    label = np.asarray(label,dtype=np.float32)
    label = np.squeeze(label)
    return input_,label

class D_scale_CNN:
    def __init__ (self,scope,scale_index,height,width,length,batch_size,layer_num_cnn,kernel_size,kernel_num,pool_kernel_size
    ,layer_num_full,full_size,scope_string):
        """ initialize the network
        @param scale_index: The index for the scale of temporal pyramid
        @height: height of the input (scaler)
        @width: width of the inputs (scaler)
        @length: length of the sequence (scaler)
        @batch_size: batch_size of LSTM (scaler)
        @layer_num_lstm: number of stacked conv_lstm layer(scaler)
        @kernel_size: list of kernel, should be consistent with layer_num
        @layer_num_full: number of fully connected layer
        @full_size: list of fully connected layer neuron, should be consistent with layer_num_full
        """
        self.scope = scope
        self.scale_index = scale_index
        self.height = height
        self.width = width
        self.length =length
        self.batch_size = batch_size
        self.layer_num_cnn= layer_num_cnn
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.layer_num_full = layer_num_full
        self.full_size = full_size
        self.pool_kernel_size = pool_kernel_size
	self.scope_string = scope_string
        assert len(kernel_size) == layer_num_cnn
        assert len(full_size) == layer_num_full
        self.define_graph()

    def define_graph(self):
        """ define a Bidirectional LSTM and concatenated feature for MLP
        """


        with tf.name_scope('input'):
            self.input_frames = tf.placeholder(
                tf.float32, shape=[None, self.length, self.height, self.width,1])
            # use variable batch_size for more flexibility
            self.GT_label = tf.placeholder(tf.float32,shape = [self.batch_size,2])

        conv_W = []
        conv_b = []
        full_connect_w=[]
        full_connect_b=[]
        with tf.variable_scope("D_scale_{}".format(self.scale_index)):
            for layer_id_,kernel_,kernel_num_ in zip(xrange(self.layer_num_cnn),self.kernel_size,self.kernel_num):
	 	with tf.variable_scope("conv_{}".format(layer_id_)):
                    conv_W.append(tf.get_variable("matrix", shape = kernel_+[kernel_num_], initializer = tf.random_uniform_initializer(-0.01, 0.01)))
                    conv_b.append(tf.get_variable("bias",shape = [kernel_num_],initializer=tf.constant_initializer(0.01)))

            for layer_id_ in xrange(self.layer_num_full-1):
                with tf.variable_scope("full_connect_{}".format(layer_id_)):
                    full_connect_w.append(tf.get_variable("matrix", [self.full_size[layer_id_],self.full_size[layer_id_+1]],initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32)))

                    full_connect_b.append(tf.get_variable("bias", [self.full_size[layer_id_+1]],initializer=tf.constant_initializer(0.01)))



        def forward():
            """Make conv pass """
            input_ = self.input_frames[:,:,:,:,:]
            for layer_id_,kernel_ in zip(xrange(self.layer_num_cnn),self.pool_kernel_size):
                input_ = tf.nn.conv3d(input_, conv_W[layer_id_], strides=[1, 1, 1, 1,1], padding='SAME') + conv_b[layer_id_]
                input_ = tf.nn.relu(input_)
		input_ = tf.nn.max_pool3d(input_,ksize = kernel_,strides =[1, 2, 2, 2,1], padding='SAME')
            preds  = tf.reshape(input_,[self.batch_size,-1])
            """Make mlp pass"""

            full_out_summary= []
            for layer_id in range(self.layer_num_full-1):
                histogram_name = "histogram:"+str(layer_id)
                preds = tf.matmul(preds,full_connect_w[layer_id])+full_connect_b[layer_id]
                temp = preds
                temp = tf.reshape(temp,[-1])
                full_out_summary.append(tf.summary.histogram(histogram_name,temp))
                if layer_id == self.layer_num_full-2:
                    preds = tf.sigmoid(preds)
                    preds_1 = tf.sub(tf.ones([self.batch_size,1]),preds)
                    preds = tf.concat(1,[preds,preds_1])
                    temp = preds
                    temp = tf.reshape(temp,[-1])
                    full_out_summary.append(tf.summary.histogram("output",temp))
                else:
                    preds = tf.nn.relu(preds)
            output = tf.clip_by_value(preds,0.05,0.95)
            output = tf.squeeze(output)
            return output,full_out_summary

	self.preds ,full_out_summary= forward()


        """ loss and training op """
        #self.loss = bce_loss(self.preds,self.GT_label)
        self.loss = adv_loss(self.preds,self.GT_label)
        temp_op = tf.train.AdamOptimizer(FLAGS.lr)
        variable_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self.scope_string)
        gvs = temp_op.compute_gradients(self.loss,var_list=variable_collection)
        capped_gvs = [(tf.clip_by_norm(grad, FLAGS.clip), var) for grad, var in gvs]
        self.train_op = temp_op.apply_gradients(capped_gvs)
        #gvs = temp_op.compute_gradients(self.loss)

        """ summary """

        loss_summary = tf.summary.scalar('loss_D', self.loss)
        self.summary = tf.summary.merge([loss_summary]+full_out_summary)
