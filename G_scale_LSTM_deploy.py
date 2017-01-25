
import tensorflow as tf
import pdb
import numpy as np
import BasicConvLSTMCell
import layer_def as ld
from data_handler import *
from loss_functions import *
from params import *


class G_scale_LSTM:
    def __init__ (self,scope,scale_index,height,width,length,batch_size,layer_num_lstm,kernel_size,kernel_num
        ,future_seq_length,flag_for_future,scope_string):
        """ initialize the network
        @param scale_index: The index for the scale of temporal pyramid
        @height: height of the input (scaler)
        @width: width of the inputs (scaler)
        @length: length of the sequence (scaler)
        @batch_size: batch_size of LSTM (scaler)
        @layer_num_lstm: number of stacked conv_lstm layer(scaler)
        @kernel_size: list of kernel, should be consistent with layer_num
        @kernel_num: list of kernel number, should be consistent with layer_num
        @future_seq_length: length of predicted frames
        @flag_for_future: flags of whether condi
        """

	self.scope = scope
        self.scale_index = scale_index
        self.height = height
        self.width = width
        self.length =length
        self.batch_size = batch_size
        self.layer_num_lstm = layer_num_lstm
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.future_seq_length = future_seq_length
	self.scope_string = scope_string
        self.define_graph()

    def define_graph(self):
        """ define a Bidirectional LSTM and concatenated feature for MLP
        """
        lstm_encode = []
        lstm_predict = []
        lstm_decode = []
	lstm_encode_state = []
        lstm_predict_state = []
	lstm_decode_state = []
        with tf.name_scope('input'):
            self.input_frames = tf.placeholder(
                tf.float32, shape=[None, self.length, self.height, self.width,1])

            # use variable batch_size for more flexibility
            self.D_label = tf.placeholder(tf.float32,shape=[self.batch_size,2])
            self.future_frames = tf.placeholder(
                tf.float32, shape=[None, self.future_seq_length, self.height, self.width,1])

        with tf.variable_scope("G_scale_{}".format(self.scale_index)):

            for layer_id_, kernel_, kernel_num_ in zip(xrange(self.layer_num_lstm),self.kernel_size,self.kernel_num):
                layer_name_encode = "conv_lstm_encode_{}".format(layer_id_)
                #with tf.variable_scope('conv_lstm_encode', initializer = tf.random_uniform_initializer(-.01, 0.1)):
                temp_cell= BasicConvLSTMCell.BasicConvLSTMCell([self.height,self.width],
                [kernel_,kernel_],kernel_num_,layer_name_encode)
                temp_state = temp_cell.zero_state(self.batch_size,tf.float32)
                lstm_encode.append(temp_cell)
                lstm_encode_state.append(temp_state)

            for layer_id_, kernel_,kernel_num_ in zip(xrange(self.layer_num_lstm),self.kernel_size,self.kernel_num):
                layer_name_predict = "conv_lstm_predict_{}".format(layer_id_)
                #with tf.variable_scope('conv_lstm_predict', initializer = tf.random_uniform_initializer(-.01, 0.1)):
                temp_cell= BasicConvLSTMCell.BasicConvLSTMCell([self.height,self.width],
                [kernel_,kernel_],kernel_num_,layer_name_predict)
                temp_state = temp_cell.zero_state(self.batch_size,tf.float32)
                lstm_predict.append(temp_cell)
                lstm_predict_state.append(temp_state)
	    
	    for layer_id_, kernel_,kernel_num_ in zip(xrange(self.layer_num_lstm),self.kernel_size,self.kernel_num):
                layer_name_predict = "conv_lstm_decode_{}".format(layer_id_)
                #with tf.variable_scope('conv_lstm_predict', initializer = tf.random_uniform_initializer(-.01, 0.1)):
                temp_cell= BasicConvLSTMCell.BasicConvLSTMCell([self.height,self.width],
                [kernel_,kernel_],kernel_num_,layer_name_predict)
                temp_state = temp_cell.zero_state(self.batch_size,tf.float32)
                lstm_decode.append(temp_cell)
                lstm_decode_state.append(temp_state)
		
        input_ = self.input_frames[:,0,:,:,:]
        for lstm_layer_id in xrange(self.layer_num_lstm):
            input_,lstm_encode_state[lstm_layer_id]=lstm_encode[lstm_layer_id](input_,lstm_encode_state[lstm_layer_id])
	
	input_ = self.input_frames[:,0,:,:,:]
        lstm_pyramid = []
        for lstm_layer_id in xrange(self.layer_num_lstm):
            input_,lstm_predict_state[lstm_layer_id]=lstm_predict[lstm_layer_id](input_,lstm_predict_state[lstm_layer_id])
            lstm_pyramid.append(input_)
	y_cat = tf.concat(3,lstm_pyramid)
        temp = ld.transpose_conv_layer(y_cat,1,1,1,"predict")

	input_ = self.input_frames[:,0,:,:,:]	
	lstm_pyramid_de = []
	for lstm_layer_id in xrange(self.layer_num_lstm):
            input_,lstm_decode_state[lstm_layer_id]=lstm_decode[lstm_layer_id](input_,lstm_decode_state[lstm_layer_id])
            lstm_pyramid_de.append(input_)
	y_cat_de = tf.concat(3,lstm_pyramid_de)
        temp_de = ld.transpose_conv_layer(y_cat_de,1,1,1,"decode")
	
	self.scope.reuse_variables()
	def forward():
            """Make forward pass """
            for frame_id in xrange(self.length):
                input_ = self.input_frames[:,frame_id,:,:,:]
		for lstm_layer_id in xrange(self.layer_num_lstm):
                    input_,lstm_encode_state[lstm_layer_id]=lstm_encode[lstm_layer_id](input_,lstm_encode_state[lstm_layer_id])

            for i in xrange(self.layer_num_lstm):
                lstm_predict_state[i]=lstm_encode_state[i]
                lstm_decode_state[i] =lstm_encode_state[i]
	    predicts = []
            for frame_id in xrange(self.future_seq_length):
		if frame_id ==0:
                    input_ = self.input_frames[:,-1,:,:,:]
		else:
                    input_ = y_out
	        # adding all layer predictions together
                lstm_pyramid = []
                for lstm_layer_id in xrange(self.layer_num_lstm):
                    input_,lstm_predict_state[lstm_layer_id]=lstm_predict[lstm_layer_id](input_,lstm_predict_state[lstm_layer_id])
                    lstm_pyramid.append(input_)
                y_cat = tf.concat(3,lstm_pyramid)
		y_out = ld.transpose_conv_layer(y_cat,1,1,1,"predict")
                predicts.append(y_out)
            # swap axis
            x_unwrap_gen = tf.pack(predicts)
            predicts = tf.transpose(x_unwrap_gen, [1,0,2,3,4])
            
	    decodes= []
            for frame_id in range(self.length,0,-1):
                if frame_id ==self.length:
                    input_ = self.future_frames[:,0,:,:,:]
                else:
                    input_ = self.input_frames[:,frame_id,:,:,:]
	        # adding all layer predictions together
                lstm_pyramid = []
                for lstm_layer_id in xrange(self.layer_num_lstm):
                    input_,lstm_decode_state[lstm_layer_id]=lstm_decode[lstm_layer_id](input_,lstm_decode_state[lstm_layer_id])
                    lstm_pyramid.append(input_)
                y_cat = tf.concat(3,lstm_pyramid)
                y_out = ld.transpose_conv_layer(y_cat,1,1,1,"decode")
                decodes.append(y_out)

            x_unwrap_de = tf.pack(decodes)
            decodes = tf.transpose(x_unwrap_de, [1,0,2,3,4])
	    
	    return predicts, decodes
	   # return predicts
	#self.preds = forward()
        self.preds,self.decodes = forward()

        """ loss and training op """
        mean_loss = l2_loss(self.preds,self.future_frames)/self.future_seq_length
	mean_loss_de = l2_loss(self.decodes,self.input_frames)/self.length
	GT_label = tf.concat(1,[tf.ones([self.batch_size,1]),tf.zeros([self.batch_size,1])])	
	entropy_loss = adv_loss(self.D_label,GT_label)
	self.loss = mean_loss+entropy_loss+mean_loss_de
 	#self.loss = mean_loss+entropy_loss
	#self.loss = combined_loss(self.preds,self.future_frames,self.D_label)
	temp_op = tf.train.AdamOptimizer(FLAGS.lr)
	variable_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self.scope_string)
	gvs = temp_op.compute_gradients(self.loss,var_list=variable_collection)
	capped_gvs = [(tf.clip_by_norm(grad, FLAGS.clip), var) for grad, var in gvs]
        self.train_op = temp_op.apply_gradients(capped_gvs)

	mean_loss_summary = tf.summary.scalar('loss_mean_pre',mean_loss)
	mean_loss_de_summary = tf.summary.scalar('loss_mean_dec',mean_loss_de)
	entropy_loss_summary = tf.summary.scalar('loss_entropy',entropy_loss)
	loss_summary = tf.summary.scalar('loss_G', self.loss)
        self.summary = tf.summary.merge([loss_summary,mean_loss_summary,entropy_loss_summary,mean_loss_de_summary])
	#self.summary = tf.summary.merge([loss_summary,mean_loss_summary,entropy_loss_summary])
