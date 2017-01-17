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
#from D_scale_LSTM import *
from D_scale_CNN import *
from loss_functions import *
from params import *

def valid():
    data_handler = VideoPatchDataHandler(FLAGS.seq_length,FLAGS.batch_size,10,'valid')

    with tf.Graph().as_default():
        with tf.variable_scope("G_0") as scope:
            G_model = G_scale_LSTM(scope=scope,scale_index=0,height=20,width=18,length=10,batch_size = FLAGS.batch_size,layer_num_lstm=2,kernel_size=[5,5],kernel_num=[128,128],future_seq_length=10,flag_for_future =True,scope_string = "G_0")

        with tf.variable_scope("D_0") as scope:
        #    #D_model = D_scale_LSTM(scope=scope,scale_index =0,height=20,width=18,length=20,batch_size =FLAGS.batch_size,layer_num_lstm=2, kernel_size=[5,5],kernel_num=[32,16],layer_num_full=3,full_size =[16*18*20*2,1000,1],scope_string = "D_0")
	    conv_kernel = []
	    kernel_num = [32,32,32]
	    conv_kernel.append([3,3,3,1])
	    conv_kernel.append([3,3,3,32])
	    conv_kernel.append([3,3,3,32])
	    pool_kernel_size = []
	    pool_kernel_size.append([1,2,2,2,1])
	    pool_kernel_size.append([1,2,2,2,1])
	    pool_kernel_size.append([1,2,2,2,1])
    	    D_model = D_scale_CNN(scope=scope,scale_index =0,height=20,width=18,length=20,batch_size =FLAGS.batch_size,layer_num_cnn=3, kernel_size=conv_kernel,kernel_num=kernel_num,pool_kernel_size = pool_kernel_size,layer_num_full=3,full_size =[3*3*3*32,1000,1],scope_string = "D_0")
        ## Build an initialization operation to run below.

        init = tf.initialize_all_variables()
        sess = tf.Session()

	sess.run(init)

        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=graph_def)
    	"""saver """
	saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess,"/scratch/ys1297/ecog/adversarial_lstm/source/checkpoints/model.ckpt-24000")

        for step in xrange(FLAGS.max_step):
            data_handler.Set_id(1000)
	    #dat = data_handler.GetBatch()
	    dat = data_handler.Get_ordered_Batch()

            # training hyper parameters
            t = time.time()
            """ forward pass for G model """
            #_, loss_r = sess.run([train_op,loss],feed_dict={G_model.input_frames:dat[:,0:FLAGS.seq_start,:,:,:],G_model.future_frames:dat[:,FLAGS.seq_start:,:,:,:],ground_holder:dat[:,FLAGS.seq_start:,:,:,:]})
            G_feed_dict = {G_model.input_frames:dat[:,0:FLAGS.seq_start,:,:,:],G_model.future_frames:dat[:,FLAGS.seq_start:,:,:,:]}
            g_predict = sess.run([G_model.preds],feed_dict=G_feed_dict)[0]

        #     """training for D model """
        #     D_input_,D_input_label = sample_function(dat[:,0:FLAGS.seq_start,:,:,:],dat[:,FLAGS.seq_start:,:,:,:],g_predict)
        #     D_feed_dict = {D_model.input_frames:D_input_,D_model.GT_label:D_input_label}
        #     d_summary_,_,d_loss,d_predicts = sess.run([D_model.summary,D_model.train_op,D_model.loss,D_model.preds],feed_dict=D_feed_dict)
    	#     summary_writer.add_summary(d_summary_,step)
    	#
   	 #    #if step%5 == 0 :
        #     """ training for G model """
    	    D_input_ = {D_model.input_frames:np.concatenate([dat[:,0:FLAGS.seq_start,:,:,:],g_predict],1)}
            d_predict = sess.run([D_model.preds],feed_dict= D_input_)[0]
	    pdb.set_trace() 
       #     G_feed_dict = {G_model.input_frames:dat[:,0:FLAGS.seq_start,:,:,:],G_model.future_frames:dat[:,FLAGS.seq_start:,:,:,:],G_model.D_label:d_predict}
        #     g_summary_,_,g_loss =  sess.run([G_model.summary,G_model.train_op,G_model.loss],feed_dict=G_feed_dict)
        #     summary_writer.add_summary(g_summary_, step)

	    #if step %100 ==0 :
		#print d_predict
		#""" generate sequence """
            #for i in range(FLAGS.batch_size):
	    #    output_file = './imgs'+'/Ecog_'+str(step)+'_'+str(i)+'.pdf'
#		data_handler.DisplayData_Ecog(dat,rec=dat[:,FLAGS.seq_start-1::-1,:,:,:],fut=g_predict,case_id=i,output_file =output_file)
	#
	    """ adding summary """
    	    #summary_writer.add_summary(g_summary_, step)
    	    #summary_writer.add_summary(d_summary_, step)
    	    #elapsed = time.time() - t

            print("time per batch is " + str(elapsed))
            print(step)
            if step %2000==0:
	    #print(d_predict)
	        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
	        saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):  # pylint: disable=unused-argument
  with tf.device('gpu:0') :
    #if tf.gfile.Exists(FLAGS.train_dir):
    #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
    #tf.gfile.MakeDirs(FLAGS.train_dir)
    valid()

if __name__ == '__main__':
  tf.app.run()

  data_handler = VideoPatchDataHandler(20,80,1,'train')
  data_handler.Set_id(700000)
  data = data_handler.Get_ordered_Batch()
  output_file = '/scratch/ys1297/ecog/conv_LSTM/Convolutional-LSTM-in-Tensorflow/imgs/'+'Ecog_{}.pdf'.format(0)
  data_handler.DisplayData_Ecog(data,output_file=output_file)
  print 1
