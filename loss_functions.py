import tensorflow as tf
import numpy as np
import pdb
from tfutils import log10





def combined_loss(gen_frames, gt_frames, d_preds, lam_adv=1, lam_lp=1, lam_gdl=1, l_num=2, alpha=2):
    """
    Calculates the sum of the combined adversarial, lp and GDL losses in the given proportion. Used
    for training the generative model.

    @param gen_frames: A list of tensors of the generated frames at each scale.
    @param gt_frames: A list of tensors of the ground truth frames at each scale.
    @param d_preds: A list of tensors of the classifications made by the discriminator model at each
                    scale.
    @param lam_adv: The percentage of the adversarial loss to use in the combined loss.
    @param lam_lp: The percentage of the lp loss to use in the combined loss.
    @param lam_gdl: The percentage of the GDL loss to use in the combined loss.
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).
    @param alpha: The power to which each gradient term is raised in GDL loss.

    @return: The combined adversarial, lp and GDL losses.
    """
    batch_size = tf.shape(gen_frames)[0]  # variable batch size as a tensor
    loss = lam_lp * l2_loss(gen_frames, gt_frames)
    #loss += lam_gdl * gdl_loss(gen_frames, gt_frames, alpha)
    loss += lam_adv * adv_loss(d_preds, tf.concat(1,[tf.ones([batch_size,1]),tf.zeros([batch_size,1])]))

    return loss


def bce_loss(preds, targets):
    """
    Calculates the sum of binary cross-entropy losses between predictions and ground truths.

    @param preds: A 1xN tensor. The predicted classifications of each frame.
    @param targets: A 1xN tensor The target labels for each frame. (Either 1 or -1). Not "truths"
                    because the generator passes in lies to determine how well it confuses the
                    discriminator.

    @return: The sum of binary cross-entropy losses.
    """
    return tf.squeeze(-1 * (tf.matmul(targets, log10(preds), transpose_a=True) +
                            tf.matmul(1 - targets, log10(1 - preds), transpose_a=True)))
def l2_loss(preds,targets):
    """Calculate the summary of l2 loss  """
    return tf.nn.l2_loss(preds-targets)

def adv_loss(preds,labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(preds,labels))
#def adv_loss(preds, labels):
#    """
#    Calculates the sum of BCE losses between the predicted classifications and true labels.
#
#    @param preds: The predicted classifications at each scale.
#    @param labels: The true labels. (Same for every scale).
#
#    @return: The adversarial loss.
#    """
#    # calculate the loss for each scale
#    scale_losses = []
#    for i in xrange(preds.size):
#        loss = bce_loss(preds[i], labels)
#        scale_losses.append(loss)
#
#    # condense into one tensor and avg
#    return tf.reduce_mean(tf.pack(scale_losses))
