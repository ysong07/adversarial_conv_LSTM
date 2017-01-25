from data_handler import *
import os.path
import time
import scipy.io as scipy_io
import numpy as np
import tensorflow as tf
#import cv2

#import bouncing_balls as b
import layer_def as ld
import BasicConvLSTMCell
import pdb
import scipy.io as scipy_io

from G_scale_LSTM_deploy import *
from G_scale_LSTM_high_deploy import *
#from G_scale_LSTM_temp import *
#from D_scale_LSTM import *
from D_scale_CNN import *
from loss_functions import *
from params import *

def train():
    data_handler = VideoPatchDataHandler(FLAGS.seq_length,FLAGS.batch_size,10,'valid')

    with tf.Graph().as_default():
        with tf.variable_scope("G_0") as scope:
            G_model_0 = G_scale_LSTM(scope=scope,scale_index=0,height=20,width=18,length=4,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[5,5],kernel_num=[128,128],future_seq_length=4,flag_for_future =True,scope_string = "G_0")

        with tf.variable_scope("D_0") as scope:
            #D_model = D_scale_LSTM(scope=scope,scale_index =0,height=20,width=18,length=20,batch_size =FLAGS.batch_size,layer_num_lstm=2, kernel_size=[5,5],kernel_num=[32,16],layer_num_full=3,full_size =[16*18*20*2,1000,1],scope_string = "D_0")
	    conv_kernel = []
	    kernel_num = [32,32,32]
	    conv_kernel.append([3,3,3,1])
	    conv_kernel.append([3,3,3,32])
	    conv_kernel.append([3,3,3,32])
	    pool_kernel_size = []
	    pool_kernel_size.append([1,2,2,2,1])
	    pool_kernel_size.append([1,2,2,2,1])
	    pool_kernel_size.append([1,2,2,2,1])
    	    D_model_0 = D_scale_CNN(scope=scope,scale_index =0,height=20,width=18,length=8,batch_size =FLAGS.batch_size,layer_num_cnn=3, kernel_size=conv_kernel,kernel_num=kernel_num,pool_kernel_size = pool_kernel_size,layer_num_full=3,full_size =[3*3*1*32,10,1],scope_string = "D_0")
        # Build an initialization operation to run below.
        with tf.variable_scope("G_1") as scope:
            G_model_1 = G_scale_LSTM_high(scope=scope,scale_index=0,height=20,width=18,length=8,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[5,5],kernel_num=[128,128],future_seq_length=8,flag_for_future =True,scope_string = "G_1")   
	
	with tf.variable_scope("D_1") as scope:
            #D_model = D_scale_LSTM(scope=scope,scale_index =0,height=20,width=18,length=20,batch_size =FLAGS.batch_size,layer_num_lstm=2, kernel_size=[5,5],kernel_num=[32,16],layer_num_full=3,full_size =[16*18*20*2,1000,1],scope_string = "D_0")
            conv_kernel = []
            kernel_num = [32,32,32]
            conv_kernel.append([3,3,3,1])
            conv_kernel.append([3,3,3,32])
            conv_kernel.append([3,3,3,32])
            pool_kernel_size = []
            pool_kernel_size.append([1,2,2,2,1])
            pool_kernel_size.append([1,2,2,2,1])
            pool_kernel_size.append([1,2,2,2,1])
            D_model_1 = D_scale_CNN(scope=scope,scale_index =0,height=20,width=18,length=16,batch_size =FLAGS.batch_size,layer_num_cnn=3, kernel_size=conv_kernel,kernel_num=kernel_num,pool_kernel_size = pool_kernel_size,layer_num_full=3,full_size =[3*3*2*32,20,1],scope_string = "D_1")
    
        with tf.variable_scope("G_2") as scope:
            G_model_2 = G_scale_LSTM_high(scope=scope,scale_index=0,height=20,width=18,length=16,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[5,5],kernel_num=[128,128],future_seq_length=16,flag_for_future =True,scope_string = "G_2")  
         
        with tf.variable_scope("D_2") as scope:
            #D_model = D_scale_LSTM(scope=scope,scale_index =0,height=20,width=18,length=20,batch_size =FLAGS.batch_size,layer_num_lstm=2, kernel_size=[5,5],kernel_num=[32,16],layer_num_full=3,full_size =[16*18*20*2,1000,1],scope_string = "D_0")
            conv_kernel = []
            kernel_num = [32,32,32]
            conv_kernel.append([3,3,3,1])
            conv_kernel.append([3,3,3,32])
            conv_kernel.append([3,3,3,32])
            pool_kernel_size = []
            pool_kernel_size.append([1,2,2,2,1])
            pool_kernel_size.append([1,2,2,2,1])
            pool_kernel_size.append([1,2,2,2,1])
            D_model_2 = D_scale_CNN(scope=scope,scale_index =0,height=20,width=18,length=32,batch_size =FLAGS.batch_size,layer_num_cnn=3, kernel_size=conv_kernel,kernel_num=kernel_num,pool_kernel_size = pool_kernel_size,layer_num_full=3,full_size =[3*3*4*32,40,1],scope_string = "D_2")
        
        init = tf.initialize_all_variables()
        sess = tf.Session()

	sess.run(init)
        
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=graph_def)
    	"""saver """
	saver = tf.train.Saver(tf.all_variables()) 
	saver.restore(sess,"/scratch/ys1297/ecog/adversarial_lstm/source/checkpoints/model.ckpt-20000")
	for step in xrange(FLAGS.max_step):
            data_handler.Set_id(1000)
	    #dat = data_handler.GetBatch()
    	    dat = data_handler.Get_ordered_Batch()
            # training hyper parameters
            t = time.time()
            """ forward pass for G model """
	    dat_down_2_past = temporal_down_scale(dat[:,0:FLAGS.seq_start,:,:,:])
	    dat_down_4_past = temporal_down_scale(dat_down_2_past)
	    dat_down_2_fut = temporal_down_scale(dat[:,FLAGS.seq_start:,:,:,:])
	    dat_down_4_fut = temporal_down_scale(dat_down_2_fut)
	    # forward pass of G_0
	    G_feed_dict_0 = {G_model_0.input_frames:dat_down_4_past,G_model_0.future_frames:dat_down_4_fut}
            g_predict_0 = sess.run([G_model_0.preds],feed_dict=G_feed_dict_0)[0]
   	    # forward pass of G_1
	    G_feed_dict_1 = {G_model_1.input_frames:dat_down_2_past, G_model_1.future_frames:dat_down_2_fut,\
	    G_model_1.input_frames_low_scale:temporal_up_scale(dat_down_4_past),G_model_1.future_frames_low_scale:temporal_up_scale(g_predict_0)}
	    g_predict_1 = sess.run([G_model_1.preds],feed_dict=G_feed_dict_1)[0] 
	    # forward pass of G_2
	    G_feed_dict_2 = {G_model_2.input_frames:dat[:,0:FLAGS.seq_start,:,:,:], G_model_2.future_frames:\
	    dat[:,FLAGS.seq_start:,:,:,:],G_model_2.input_frames_low_scale:temporal_up_scale(dat_down_2_past),G_model_2.future_frames_low_scale: temporal_up_scale(g_predict_1)}
            g_predict_2 = sess.run([G_model_2.preds],feed_dict=G_feed_dict_2)[0]     
	
#	    if step %5==0: 
#                """training for D model """
#		# training D_0
#		D_input_0, D_input_label_0 = sample_function(dat_down_4_past,dat_down_4_fut,g_predict_0)
#		D_feed_dict_0 = {D_model_0.input_frames:D_input_0,D_model_0.GT_label:D_input_label_0}
#		d_summary_0,_,d_loss_0,d_predicts_0 = sess.run([D_model_0.summary,D_model_0.train_op,D_model_0.loss,D_model_0.preds],feed_dict=D_feed_dict_0)
#	        summary_writer.add_summary(d_summary_0,step)
#
#	        # training D_1
#                D_input_1, D_input_label_1 = sample_function(dat_down_2_past,dat_down_2_fut,g_predict_1)
#                D_feed_dict_1 = {D_model_1.input_frames:D_input_1,D_model_1.GT_label:D_input_label_1}
#                d_summary_1,_,d_loss_1,d_predicts_1 = sess.run([D_model_1.summary,D_model_1.train_op,D_model_1.loss,D_model_1.preds],feed_dict=D_feed_dict_1)
#                summary_writer.add_summary(d_summary_1,step)
#
#	        # training D_2
#                D_input_2, D_input_label_2 = sample_function(dat[:,0:FLAGS.seq_start,:,:,:],dat[:,FLAGS.seq_start:,:,:,:],g_predict_2)
#                D_feed_dict_2 = {D_model_2.input_frames:D_input_2,D_model_2.GT_label:D_input_label_2}
#                d_summary_2,_,d_loss_2,d_predicts_2 = sess.run([D_model_2.summary,D_model_2.train_op,D_model_2.loss,D_model_2.preds],feed_dict=D_feed_dict_2)
#                summary_writer.add_summary(d_summary_2,step)
#
#            """ training for G model """
#	    # traininig for G_0
#    	    D_input_0 = {D_model_0.input_frames:np.concatenate([dat_down_4_past,g_predict_0],1)}
#     	    d_predict_0 = sess.run([D_model_0.preds],feed_dict= D_input_0)[0]	
#	    G_feed_dict_0 = {G_model_0.input_frames:dat_down_4_past,G_model_0.future_frames:dat_down_4_fut,G_model_0.D_label:d_predict_0}
#	    g_summary_0,_,g_loss_0 =  sess.run([G_model_0.summary,G_model_0.train_op,G_model_0.loss],feed_dict=G_feed_dict_0)
#	    summary_writer.add_summary(g_summary_0, step)
#
#	    # training for G_1
#	    D_input_1 = {D_model_1.input_frames:np.concatenate([dat_down_2_past,g_predict_1],1)}
#            d_predict_1 = sess.run([D_model_1.preds],feed_dict= D_input_1)[0]  
#	    G_feed_dict_1 = {G_model_1.input_frames:dat_down_2_past, G_model_1.future_frames:dat_down_2_fut,G_model_1.input_frames_low_scale:temporal_up_scale(dat_down_4_past),G_model_1.future_frames_low_scale:temporal_up_scale(dat_down_4_fut),G_model_1.D_label:d_predict_1}
#	    g_summary_1,_,g_loss_1 =  sess.run([G_model_1.summary,G_model_1.train_op,G_model_1.loss],feed_dict=G_feed_dict_1) 
#            summary_writer.add_summary(g_summary_1, step)
# 
# 	    # training for G_2
#	    D_input_2 = {D_model_2.input_frames:np.concatenate([dat[:,0:FLAGS.seq_start,:,:,:],g_predict_2],1)}
#            d_predict_2 = sess.run([D_model_2.preds],feed_dict= D_input_2)[0]  
#	    G_feed_dict_2 = {G_model_2.input_frames:dat[:,0:FLAGS.seq_start,:,:,:], G_model_2.future_frames:dat[:,FLAGS.seq_start:,:,:,:],G_model_2.input_frames_low_scale:temporal_up_scale(dat_down_2_past),G_model_2.future_frames_low_scale:temporal_up_scale(dat_down_2_fut),G_model_2.D_label:d_predict_2}
#
#	    g_summary_2,_,g_loss_2 =  sess.run([G_model_2.summary,G_model_2.train_op,G_model_2.loss],feed_dict=G_feed_dict_2)
#            summary_writer.add_summary(g_summary_2, step)
    	    
	    #if step %100 ==0 :
		#print d_predict
		#""" generate sequence """
	    for i in range(FLAGS.batch_size):
		output_file = './imgs'+'/Ecog_'+'scale_0_'+str(i)+'.pdf'
                data_handler.DisplayData_Ecog(temporal_down_scale(temporal_down_scale(dat)),rec=dat_down_4_past[:,dat_down_4_past.shape[1]::-1,:,:,:],fut=g_predict_0,case_id=i,output_file =output_file)
	 	output_file = './imgs'+'/Ecog_'+'scale_1_'+str(i)+'.pdf'
                data_handler.DisplayData_Ecog(temporal_down_scale(dat),rec=dat_down_2_past[:,dat_down_2_past.shape[1]::-1,:,:],fut=g_predict_1,case_id=i,output_file =output_file)
		output_file = './imgs'+'/Ecog_'+'scale_2_'+str(i)+'.pdf'
                data_handler.DisplayData_Ecog(dat,rec=dat[:,FLAGS.seq_start-1::-1,:,:,:],fut=g_predict_2,case_id=i,output_file =output_file)    	
	    """ adding summary """
    	    #summary_writer.add_summary(g_summary_, step)  
    	    elapsed = time.time() - t
	    sys.exit()    
            print("time per batch is " + str(elapsed))
            print(step)
            if step %100==0:
	    #print(d_predict)
	        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
	        saver.save(sess, checkpoint_path, global_step=step)  
	
def main(argv=None):  # pylint: disable=unused-argument
  with tf.device('gpu:0') :
    #if tf.gfile.Exists(FLAGS.train_dir):
    #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
    #tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
  tf.app.run()

  data_handler = VideoPatchDataHandler(20,80,1,'train')
  data_handler.Set_id(700000)
  data = data_handler.Get_ordered_Batch()
  output_file = '/scratch/ys1297/ecog/conv_LSTM/Convolutional-LSTM-in-Tensorflow/imgs/'+'Ecog_{}.pdf'.format(0)
  data_handler.DisplayData_Ecog(data,output_file=output_file)
  print 1
