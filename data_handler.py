import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()
import tensorflow as tf
import cPickle as pkl
import sys
import pdb
import h5py
import time
from params import *
def temporal_down_scale(tensor):
    # tensor: batch_size,length,height,width,1
    # note: the downsampler is mannually set to 2
    if type(tensor).__module__ == np.__name__: #if numpy array
        out_tensor = np.zeros([tensor.shape[0],tensor.shape[1]/2,tensor.shape[2],tensor.shape[3],tensor.shape[4]])
	for i in range(0,tensor.shape[1],2):
            out_tensor[:,i/2,:,:,:]= np.mean(tensor[:,i:i+2,:,:,:],axis=1) 
    else:
	original_shape = [FLAGS.batch_size,tensor.get_shape()[1].value,FLAGS.width,FLAGS.height,1]
	#original_shape = [tensor.get_shape()[0].value,tensor.get_shape()[1].value,tensor.get_shape()[2].value,\
        #tensor.get_shape()[3].value,tensor.get_shape()[4].value]
	tensor = tf.reshape(tensor,original_shape[0:4])
	out_tensor = tf.nn.avg_pool(tensor,ksize=[1,2,1,1],strides=[1,2,1,1],padding ='SAME')
	out_tensor = tf.reshape(out_tensor,[original_shape[0],original_shape[1]/2,original_shape[2],original_shape[3],\
	original_shape[4]])
    return out_tensor

def temporal_up_scale(tensor):
    # tensor: batch_size,length,height,width,1
    # note: the upsampler is mannually set to 2
    if type(tensor).__module__ == np.__name__ : #if numpy array
	tensor = np.swapaxes(tensor,1,4)
        original_shape = [tensor.shape[0],tensor.shape[1],tensor.shape[2],tensor.shape[3],tensor.shape[4]]

        up_scale_ = get_up_scale_matrix(tensor.shape[4]) 
	tensor = np.dot(tensor.reshape([-1,tensor.shape[4]]),up_scale_)
        tensor = np.reshape(tensor,original_shape[0:4]+[original_shape[4]*2])
	tensor = np.swapaxes(tensor,1,4)
    else:
        tensor = tf.transpose(tensor,[0,4,2,3,1])
	original_shape = [FLAGS.batch_size,1,FLAGS.width,FLAGS.height,tensor.get_shape()[4].value]
	#original_shape = [tensor.get_shape()[0].value,tensor.get_shape()[1].value,tensor.get_shape()[2].value,\
	#tensor.get_shape()[3].value,tensor.get_shape()[4].value]
	up_scale_ = tf.constant(get_up_scale_matrix(original_shape[4]).astype('float32'))
	tensor = tf.matmul(tf.reshape(tensor,[-1,original_shape[4]]),up_scale_)
	tensor = tf.reshape(tensor,original_shape[0:4]+[original_shape[4]*2])
	tensor = tf.transpose(tensor,[0,4,2,3,1])
	 

    return tensor
def get_up_scale_matrix(n):
    # get n *2n bilinear interpolation numpy matrix
    out_matrix = np.zeros([n,2*n])
    temp_array = np.array([0.25,0.75,0.75,0.25])
    for temp in range(1,n-1):
	out_matrix[temp,2*temp-1:2*temp+4-1] = temp_array
    out_matrix[0,0:3] = np.array([1.25,0.75,0.25])
    out_matrix[1,0:3] = np.array([-.25,0.25,0.75])
    out_matrix[-1,-1:-4:-1] = np.array([1.25,0.75,0.25])
    out_matrix[-2,-1:-4:-1] = np.array([-.25,0.25,0.75])
    return out_matrix
    
	
class VideoPatchDataHandler(object):
  def __init__(self,sequence_length=20,batch_size=80,down_sample_rate_=1,dataset_name='train'):
    stats = pkl.load(open('/scratch/ys1297/ecog/data/ECOG_4041_mean.pkl','r'))
    self.data_file_ = h5py.File('/scratch/ys1297/ecog/data/ECOG_40_41.h5','r')[dataset_name]
    # data sampler sliding window step size
    self.down_sample_rate_ = down_sample_rate_
    # sequence length
    self.seq_length_ = sequence_length
    
    self.batch_size_ = batch_size
    self.image_size_ = [20,18]
   
     
    try:
      self.data_ = np.float32(self.data_file_[:]*stats['scale_factor']) #data 36 min:-120 max:103
      #self.data_ = (self.data_-stats['min'])/(stats['max']-stats['min'])
      self.data_ = 2*((self.data_-stats['min'])/(stats['max']-stats['min'])-0.5) 
      #self.data_ = np.float32(self.data_file_[:]*10000) #data 41
      #self.data_ = (self.data_+10)/(20+10)
        # self.data_ min: -150 max: 103
      self.data_[self.data_<-1]=-1
      self.data_[self.data_>1]=1
    except:
      print 'Please set the correct path to the dataset'
      sys.exit()
    self.dataset_size_ = self.data_.shape[0]
    self.id_= 0
    self.indices_ = np.arange((self.dataset_size_-self.seq_length_)/self.down_sample_rate_)
    self.is_color_ = False
    np.random.seed(100)
    np.random.shuffle(self.indices_)
  def GetBatch(self, verbose=False):
    minibatch = np.zeros((self.batch_size_, self.seq_length_, self.image_size_[0],self.image_size_[1],1), dtype=np.float32)

    for j in range(self.batch_size_):
      temp =  self.data_[self.indices_[self.id_]*self.down_sample_rate_:self.indices_[self.id_]*self.down_sample_rate_+self.seq_length_,:]
      minibatch[j,:,:,:,0]  = temp.reshape(-1,20,18)
      self.id_ += 1
      if self.id_ >= (self.dataset_size_-self.seq_length_-1)/self.down_sample_rate_:
        self.id_ = 0
        self.indices_ = np.arange((self.dataset_size_-self.seq_length_)/self.down_sample_rate_)
        np.random.shuffle(self.indices_)

    return minibatch
  
  def Set_id(self,id):
    self.id_ = id
    pass
  
  def Get_ordered_Batch(self):
    minibatch = np.zeros((self.batch_size_, self.seq_length_, self.image_size_[0],self.image_size_[1],1), dtype=np.float32)

    for j in range(self.batch_size_):  
      #temp =  self.data_[self.indices_[self.id_]*self.down_sample_rate_:self.indices_[self.id_]*self.down_sample_rate_+self.seq_length_,:]
      temp = self.data_[self.id_*self.down_sample_rate_:self.id_*self.down_sample_rate_+self.seq_length_,:]
      minibatch[j,:,:,:,0]  = temp.reshape(-1,20,18)
      self.id_ += 1
      if self.id_ >= (self.dataset_size_-self.seq_length_-1)/self.down_sample_rate_:
        self.id_ = 0
        self.indices_ = np.arange((self.dataset_size_-self.seq_length_)/self.down_sample_rate_)
        np.random.shuffle(self.indices_)

    return minibatch
  def DisplayData_Ecog(self, data, rec=None, fut=None, fig=1, case_id=0, output_file=None):
    output_file1 = None
    output_file2 = None
    data = data/2.0+0.5
    if fut is not None:
      fut = fut/2.0+0.5
    if rec is not None:
      rec = rec/2.0+0.5
    # to adjust to multi scale
    self.seq_length_ = data.shape[1] 

    if output_file is not None:
      name, ext = os.path.splitext(output_file)
      output_file1 = '%s_original%s' % (name, ext)
      output_file2 = '%s_recon%s' % (name, ext)
      output_file3 = '%s_original_norm_color%s' % (name, ext)
    # get data
    if self.is_color_:
      data = data[case_id, :]
      data[data>1.] = 1.
      data[data<0.] = 0.
      data = data.reshape(-1, 3, self.image_size_, self.image_size_)
      data = data.transpose(0, 2, 3, 1)
    else:
      #data = data[case_id, :].reshape(-1, 20, 18)
      data = np.squeeze(data[case_id,:,:,:,0])
      data = np.swapaxes(data,1,2)

    # get reconstruction and future sequences if they exist
    if rec is not None:
      if self.is_color_:
        rec = rec[case_id, :]
        rec[rec>1.] = 1.
        rec[rec<0.] = 0.
        rec = rec.reshape(-1, 3, self.image_size_, self.image_size_)
        rec = rec.transpose(0, 2, 3, 1)
      else:
        #rec = rec[case_id, :].reshape(-1, 20,18)
        rec = np.squeeze(rec[case_id,:,:,:,0]) 
	rec = np.swapaxes(rec,1,2)
      enc_seq_length = rec.shape[0]
    else:
      rec = data[20:0:-1,:,:]
      enc_seq_length = rec.shape[0]	
    
    if fut is not None:
      if self.is_color_:
        fut = fut[case_id, :]
        fut[fut>1.] = 1.
        fut[fut<0.] = 0.
        fut = fut.reshape(-1, 3, self.image_size_, self.image_size_)
        fut = fut.transpose(0, 2, 3, 1)
      else:
        #fut = fut[case_id, :].reshape(-1, 20,18)
        fut = np.squeeze(fut[case_id,:,:,:,0])
	fut = np.swapaxes(fut,1,2)
    else:
      fut = data[9:-1,:,:]

    if rec is None:
      enc_seq_length = self.seq_length_ - fut.shape[0]
    else:
      assert enc_seq_length == self.seq_length_ - fut.shape[0]

    """ new plot method """
    num_rows = 3
    plt.figure(2*fig, figsize=(self.seq_length_, 3))
    plt.clf()
    #plt.figure(1)
    for i in xrange(self.seq_length_):
      plt.subplot(num_rows, self.seq_length_, i+1)
      if self.is_color_:
        plt.imshow(data[i])
      else:
        plt.imshow(-data[i, :, :], cmap=plt.cm.jet, interpolation="nearest")
        plt.clim(-0.8,-0.3)
      plt.axis('off')

    # create figure for reconstuction and future sequences
    for i in xrange(self.seq_length_):
      if rec is not None and i < enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1 + self.seq_length_)
        if self.is_color_:
          plt.imshow(rec[rec.shape[0] - i - 1])
        else:
          plt.imshow(-rec[rec.shape[0] - i - 1, :, :], cmap=plt.cm.jet , interpolation="nearest")
          plt.clim(-0.8,-0.3)
      if fut is not None and i >= enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1+self.seq_length_)
        if self.is_color_:
          plt.imshow(fut[i - enc_seq_length])
        else:
          plt.imshow(-fut[i - enc_seq_length, :, :], cmap=plt.cm.jet,  interpolation="nearest")
          plt.clim(-0.8,-0.3)
      plt.axis('off')

    error_sum = 0
    for i in xrange(self.seq_length_):
      if rec is not None and i < enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1 + self.seq_length_*2)
        if self.is_color_:
          plt.imshow(rec[rec.shape[0] - i - 1])
        else:
          plt.imshow(-data[i, :, :]+rec[rec.shape[0] - i - 1, :, :], cmap=plt.cm.jet , interpolation="nearest")
          plt.clim(-0.4,0.4)
      if fut is not None and i >= enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1+self.seq_length_*2)
        if self.is_color_:
          plt.imshow(fut[i - enc_seq_length])
        else:
          plt.imshow(-data[i, :, :]+fut[i - enc_seq_length, :, :], cmap=plt.cm.jet,  interpolation="nearest")
          plt.clim(-0.4,0.4)
          error_sum += np.sum(np.abs(-data[i, :, :]+fut[i - enc_seq_length, :, :]))
      plt.axis('off')
    plt.draw()
    print output_file3
    plt.savefig(output_file3, bbox_inches='tight')
    return error_sum

if __name__ == '__main__':
    from params import *
    data_handler = VideoPatchDataHandler(FLAGS.seq_length,FLAGS.batch_size,10,'valid')
    data_handler.Set_id(1000)
    dat = data_handler.Get_ordered_Batch()
    tensor = tf.constant(dat)
  
    down_tensor = temporal_down_scale(tensor)
    down_tensor = temporal_down_scale(down_tensor)
    up_tensor = temporal_up_scale(down_tensor)
    up_tensor = temporal_up_scale(up_tensor)
    
    sess = tf.Session()
    t1 = time.time()
    d_predict = sess.run([up_tensor])[0]
    t2 = time.time()
    print t2-t1
    output_file = './imgs'+'/Ecog_test_tensorflow'+'.pdf'
    data_handler.DisplayData_Ecog(dat,rec=d_predict[:,20:0:-1,:,:,],fut=d_predict[:,20:,:,:,:],case_id=14,output_file =output_file)

    t1 = time.time()
    down_dat = temporal_down_scale(dat)
    down_dat = temporal_down_scale(down_dat)
    up_dat = temporal_up_scale(down_dat)
    up_dat = temporal_up_scale(up_dat)
    t2 = time.time()
    print t2-t1
    output_file = './imgs'+'/Ecog_test_numpy'+'.pdf'
    data_handler.DisplayData_Ecog(dat,rec=up_dat[:,20:0:-1,:,:,],fut=up_dat[:,20:,:,:,:],case_id=14,output_file =output_file)
