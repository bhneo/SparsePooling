"""
License: Apache 2.0
Author: Ashley Gritzman
E-mail: ashley.gritzman@za.ibm.com

Credits: 
  1.  Jonathan Hui's blog, "Understanding Matrix capsules with EM Routing 
      (Based on Hinton's Capsule Networks)" 
      https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-
      Capsule-Network/
  2.  Questions and answers on OpenReview, "Matrix capsules with EM routing" 
      https://openreview.net/forum?id=HJWLfGWRb
  3.  Suofei Zhang's implementation on GitHub, "Matrix-Capsules-EM-Tensorflow" 
      https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow
  4.  Guang Yang's implementation on GitHub, "CapsulesEM" 
      https://github.com/gyang274/capsulesEM
"""

import numpy as np
# Public modules
import tensorflow as tf


def em_routing(votes_ij, activations_i, beta_a, beta_v, iter_routing, softmax_in, temper, final_lambda, epsilon, spatial_routing_matrix, log=None):
  """The EM routing between input capsules (i) and output capsules (j).
  
  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description 
  of EM routing.
  
  Author:
    Ashley Gritzman 19/10/2018
  Definitions:
    N -> number of samples in batch
    i -> number of input capsules, also called "child_caps"
    o -> number of output capsules, also called "parent_caps"
    child_space -> spatial dimensions of input capsule layer i
    parent_space -> spatial dimensions of output capsule layer j
    n_channels -> number of channels in pose matrix (usually 4x4=16)
  Args: 
    votes_ij: 
      votes from capsules in layer i to capsules in layer j
      For FC layer:
        The kernel dimensions are equal to the spatial dimensions of the input 
        layer i, and the spatial dimensions of the output layer j are 1x1.
        (N, i, o, 4x4)
    activations_i: 
      activations of capsules in layer i (L)
      (N, i, 1)
    batch_size: 
    spatial_routing_matrix: 
  Returns:
    poses_j: 
      poses of capsules in layer j (L+1)
      (N, o, 4x4)
    activations_j: 
      activations of capsules in layer j (L+1)
      (N, o, 1)
  """
  
  #----- Dimensions -----#
  
  # Get dimensions needed to do conversions
  votes_shape = votes_ij.get_shape().as_list()
  caps_in = int(votes_shape[1])
  caps_out = int(votes_shape[2])
  atoms = int(votes_shape[3])

  #----- Reshape Inputs -----#
  votes_ij = tf.reshape(votes_ij, [-1, caps_in, caps_out, atoms])
  activations_i = tf.reshape(activations_i, [-1, caps_in, 1, 1])

  with tf.name_scope("em_routing") as scope:
    # rr = 1 / (caps_out + 1e-9)
    # rr = np.reshape(rr, [1, 1, 1, 1])
    # rr = np.tile(rr, [1, caps_in, caps_out, 1])
    # # Convert rr from np to tf
    # rr = tf.constant(rr, dtype=tf.float32)

    b = tf.zeros_like(votes_ij, name='b')
    b = tf.reduce_sum(b, -1, keepdims=True)
    rr = tf.nn.softmax(b, axis=-2)

    poses = []
    probs = []
    cs = []
    
    for it in range(iter_routing):
      # AG 17/09/2018: modified schedule for inverse_temperature (lambda) based
      # on Hinton's response to questions on OpenReview.net: 
      # https://openreview.net/forum?id=HJWLfGWRb
      # "the formula we used for lambda is:
      # lambda = final_lambda * (1 - tf.pow(0.95, tf.cast(i + 1, tf.float32)))
      # where 'i' is the routing iteration (range is 0-2). Final_lambda is set 
      # to 0.01."
      # final_lambda = 0.01
      cs.append(rr)

      inverse_temperature = (final_lambda * 
                             (1 - tf.pow(0.95, tf.cast(it + 1, tf.float32))))

      # AG 26/06/2018: added var_j
      activations_j, mean_j, stdv_j, var_j = m_step(
        rr, 
        votes_ij, 
        activations_i, 
        beta_v, beta_a, 
        inverse_temperature=inverse_temperature,
        epsilon=epsilon
      )

      # We skip the e_step call in the last iteration because we only need to 
      # return the a_j and the mean from the m_stp in the last iteration to 
      # compute the output capsule activation and pose matrices  
      if it < iter_routing - 1:
        rr = e_step(votes_ij, 
                    activations_j, 
                    mean_j, 
                    stdv_j, 
                    var_j, 
                    spatial_routing_matrix,
                    epsilon,
                    softmax_in,
                    temper)


      # pose: (N, OH, OW, o, 4 x 4) via squeeze mean_j (24, 6, 6, 32, 16)
      poses_j = tf.squeeze(mean_j, axis=-3, name="poses")

      # activation: (N, OH, OW, o, 1) via squeeze o_activation is
      # [24, 6, 6, 32, 1]
      activations_j = tf.squeeze(activations_j, axis=-3, name="activations")

      poses.append(poses_j)
      probs.append(activations_j)

  return poses, probs, cs


def m_step(rr, votes, activations_i, beta_v, beta_a, inverse_temperature, epsilon):
  """The m-step in EM routing between input capsules (i) and output capsules 
  (j).
  
  Compute the activations of the output capsules (j), and the Gaussians for the
  pose of the output capsules (j).
  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description 
  of m-step.
  
  Author:
    Ashley Gritzman 19/10/2018
    
  Args: 
    rr: 
      assignment weights between capsules in layer i and layer j
      (N, i, o, 1)
    votes_ij: 
      votes from capsules in layer i to capsules in layer j
      For FC layer:
        The kernel dimensions are equal to the spatial dimensions of the input 
        layer i, and
        the spatial dimensions of the output layer j are 1x1.
        (N, i, o, 4x4)
    activations_i: 
      activations of capsules in layer i (L)
      (N, i, o, 1)
    beta_v: 
      Trainable parameters in computing cost 
      (1, 1, 32, 1)
    beta_a: 
      Trainable parameters in computing next level activation 
      (1, 1, 32, 1)
    inverse_temperature: lambda, increase over each iteration by the caller
    
  Returns:
    activations_j: 
      activations of capsules in layer j (L+1)
      (N, 1, o, 1)
    mean_j: 
      mean of each channel in capsules of layer j (L+1)
      (N, 1, o, n_channels)
    stdv_j: 
      standard deviation of each channel in capsules of layer j (L+1)
      (N, 1, o, n_channels)
    var_j: 
      variance of each channel in capsules of layer j (L+1)
      (N, 1, o, n_channels)
  """

  with tf.name_scope("m_step") as scope:
    
    rr_prime = rr * activations_i
    rr_prime = tf.identity(rr_prime, name="rr_prime")

    # rr_prime_sum: sum over all input capsule i
    rr_prime_sum = tf.reduce_sum(rr_prime, 
                                 axis=-3, 
                                 keepdims=True, 
                                 name='rr_prime_sum')
    
    # AG 13/12/2018: normalise amount of information
    # The amount of information given to parent capsules is very different for 
    # the final "class-caps" layer. Since all the spatial capsules give output 
    # to just a few class caps, they receive a lot more information than the 
    # convolutional layers. So in order for lambda and beta_v/beta_a settings to 
    # apply to this layer, we must normalise the amount of information.
    # activ from convcaps1 to convcaps2 (64*5*5, 144, 16, 1) 144/16 = 9 info
    # (N*OH*OW, kh*kw*i, o, 1)
    # activ from convcaps2 to classcaps (64, 1, 1, 400, 5, 1) 400/5 = 80 info
    # (N, 1, 1, IH*IW*i, n_classes, 1)
    child_caps = float(rr_prime.get_shape().as_list()[-3])
    parent_caps = float(rr_prime.get_shape().as_list()[-2])
    ratio_child_to_parent = child_caps/parent_caps
    layer_norm_factor = 100/ratio_child_to_parent
    # logger.info("ratio_child_to_parent: {}".format(ratio_child_to_parent))
    # rr_prime_sum = rr_prime_sum/ratio_child_to_parent

    # mean_j: (24, 6, 6, 1, 32, 16)
    mean_j_numerator = tf.reduce_sum(rr_prime * votes, 
                                     axis=-3, 
                                     keepdims=True, 
                                     name="mean_j_numerator")
    mean_j = tf.divide(mean_j_numerator,
                       rr_prime_sum + epsilon,
                       name="mean_j")
    
    #----- AG 26/06/2018 START -----#
    # Use variance instead of standard deviation, because the sqrt seems to 
    # cause NaN gradients during backprop.
    # See original implementation from Suofei below
    var_j_numerator = tf.reduce_sum(rr_prime * tf.square(votes - mean_j), 
                                    axis=-3, 
                                    keepdims=True, 
                                    name="var_j_numerator")
    var_j = tf.divide(var_j_numerator,
                      rr_prime_sum + epsilon,
                      name="var_j")
    
    # Set the minimum variance (note: variance should always be positive)
    # This should allow me to remove the FLAGS.epsilon safety from log and div 
    # that follow
    #var_j = tf.maximum(var_j, FLAGS.epsilon)
    #var_j = var_j + FLAGS.epsilon
    
    ###################
    #var_j = var_j + 1e-5
    var_j = tf.identity(var_j + 1e-9, name="var_j_epsilon")
    ###################
    
    # Compute the stdv, but it shouldn't actually be used anywhere
    # stdv_j = tf.sqrt(var_j)
    stdv_j = None
    
    ######## layer_norm_factor
    cost_j_h = (beta_v + 0.5*tf.math.log(var_j)) * rr_prime_sum * layer_norm_factor
    cost_j_h = tf.identity(cost_j_h, name="cost_j_h")
    
    # ----- END ----- #
    
    """
    # Original from Suofei (reference [3] at top)
    # stdv_j: (24, 6, 6, 1, 32, 16)
    stdv_j = tf.sqrt(
      tf.reduce_sum(
        rr_prime * tf.square(votes - mean_j), axis=-3, keepdims=True
      ) / rr_prime_sum,
      name="stdv_j"
    )
    # cost_j_h: (24, 6, 6, 1, 32, 16)
    cost_j_h = (beta_v + tf.log(stdv_j + FLAGS.epsilon)) * rr_prime_sum
    """
    
    # cost_j: (24, 6, 6, 1, 32, 1)
    # activations_j_cost = (24, 6, 6, 1, 32, 1)
    # yg: This is done for numeric stability.
    # It is the relative variance between each channel determined which one 
    # should activate.
    cost_j = tf.reduce_sum(cost_j_h, axis=-1, keepdims=True, name="cost_j")
    #cost_j_mean = tf.reduce_mean(cost_j, axis=-2, keepdims=True)
    #cost_j_stdv = tf.sqrt(
    #  tf.reduce_sum(
    #    tf.square(cost_j - cost_j_mean), axis=-2, keepdims=True
    #  ) / cost_j.get_shape().as_list()[-2]
    #)
    
    # AG 17/09/2018: trying to remove normalisation
    # activations_j_cost = beta_a + (cost_j_mean - cost_j) / (cost_j_stdv)
    activations_j_cost = tf.identity(beta_a - cost_j, 
                                     name="activations_j_cost")

    # (24, 1, 32, 1)
    activations_j = tf.sigmoid(inverse_temperature * activations_j_cost,
                               name="sigmoid")
    
    # AG 26/06/2018: added var_j to return
    return activations_j, mean_j, stdv_j, var_j

  
# AG 26/06/2018: added var_j
def e_step(votes_ij, activations_j, mean_j, stdv_j, var_j, spatial_routing_matrix, epsilon, softmax_in, temper):
  """The e-step in EM routing between input capsules (i) and output capsules (j).
  
  Update the assignment weights using in routung. The output capsules (j) 
  compete for the input capsules (i).
  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description 
  of e-step.
  
  Author:
    Ashley Gritzman 19/10/2018
    
  Args: 
    votes_ij: 
      votes from capsules in layer i to capsules in layer j
        (N, i, o, 4x4)
    activations_j: 
      activations of capsules in layer j (L+1)
      (N, i, o, 1)
    mean_j: 
      mean of each channel in capsules of layer j (L+1)
      (N, 1, o, 16)
    stdv_j: 
      standard deviation of each channel in capsules of layer j (L+1)
      (N, 1, o, n_channels)
    var_j: 
      variance of each channel in capsules of layer j (L+1)
      (N, 1, o, n_channels)
    spatial_routing_matrix: ???
    
  Returns:
    rr: 
      assignment weights between capsules in layer i and layer j
      (N, i, o, 1)
  """
  
  with tf.name_scope("e_step") as scope:
    
    # AG 26/06/2018: changed stdv_j to var_j
    o_p_unit0 = - tf.reduce_sum(
      tf.square(votes_ij - mean_j, name="num") / (2 * var_j), 
      axis=-1, 
      keepdims=True, 
      name="o_p_unit0")
    
    o_p_unit2 = - 0.5 * tf.reduce_sum(
      tf.math.log(2*np.pi * var_j),
      axis=-1, 
      keepdims=True, 
      name="o_p_unit2"
    )

    # (24, 288, 32, 1)
    o_p = o_p_unit0 + o_p_unit2
    zz = tf.math.log(activations_j + epsilon) + o_p
    
    # AG 13/11/2018: New implementation of normalising across parents
    #----- Start -----#
    zz_shape = zz.get_shape().as_list()
    caps_in = zz_shape[1]
    caps_out = zz_shape[2]
    
    zz = tf.reshape(zz, [-1, caps_in, caps_out])
    
    with tf.name_scope("softmax_across_parents") as scope:
      zz_softmax = softmax_across_parents(zz, spatial_routing_matrix, softmax_in, temper)
      
    rr = tf.reshape(zz_softmax, [-1, caps_in, caps_out, 1])
    #----- End -----#

    # AG 02/11/2018
    # In response to a question on OpenReview, Hinton et al. wrote the 
    # following:
    # "The gradient flows through EM algorithm. We do not use stop gradient. A 
    # routing of 3 is like a 3 layer network where the weights of layers are 
    # shared."
    # https://openreview.net/forum?id=HJWLfGWRb&noteId=S1eo2P1I3Q
    return rr


def softmax_across_parents(probs_sparse, spatial_routing_matrix, softmax_in, temper=1.0):
  """Softmax across all parent capsules including spatial and depth.

  Consider a sparse matrix of probabilities (1, 5, 5, 49, 8, 32)
  (batch_size, parent_space, parent_space, child_space*child_space, child_caps,   parent_caps)

  For one child capsule, we need to normalise across all parent capsules that
  receive output from that child. This includes the depth of parent capsules,
  and the spacial dimension od parent capsules. In the example matrix of
  probabilities above this would mean normalising across [1, 2, 5] or
  [parent_space, parent_space, parent_caps]. But the softmax function
  `tf.nn.softmax` can only operate across one axis, so we need to reshape the
  matrix such that we can combine paret_space and parent_caps into one axis.

  Author:
    Ashley Gritzman 05/11/2018

  Args:
    probs_sparse:
      the sparse representation of the probs matrix, in log
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 49, 8, 32)

  Returns:
    rr_updated:
      softmax across all parent capsules, same shape as input
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 49, 8, 32)
  """

  # e.g. (1, 8, 32)
  # (batch_size, child_caps, parent_caps)
  # Perform softmax across parent capsule dimension
  if softmax_in:
    parent_softmax = tf.nn.softmax(probs_sparse * temper, axis=-2)
  else:
    parent_softmax = tf.nn.softmax(probs_sparse * temper, axis=-1)

  # weights
  rr_updated = parent_softmax

  return rr_updated


def to_sparse(probs, spatial_routing_matrix, sparse_filler=tf.math.log(1e-20)):
  """Convert probs tensor to sparse along child_space dimension.

  Consider a probs tensor of shape (64, 6, 6, 3*3, 32, 16).
  (batch_size, parent_space, parent_space, kernel*kernel, child_caps,
  parent_caps)
  The tensor contains the probability of each child capsule belonging to a
  particular parent capsule. We want to be able to sum the total probabilities
  for a single child capsule to all the parent capsules. So we need to convert
  the 3*3 spatial locations have been condensed, into a sparse format across
  all child spatial location e.g. 14*14.

  Since we are working in log space, we must replace the zeros that come about
  during sparse with log(0). The 'sparse_filler' option allows us to specify the
  number to use to fill.

  Author:
    Ashley Gritzman 01/11/2018

  Args:
    probs:
      tensor of log probabilities of each child capsule belonging to a
      particular parent capsule
      (batch_size, parent_space, parent_space, kernel*kernel, child_caps,
      parent_caps)
      (64, 5, 5, 3*3, 32, 16)
    spatial_routing_matrix:
      binary routing map with children as rows and parents as columns
    sparse_filler:
      the number to use to fill in the sparse locations instead of zero

  Returns:
    sparse:
      the sparse representation of the probs tensor in log space
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 7*7, 32, 16)
  """

  # Get shapes of probs
  shape = probs.get_shape().as_list()
  batch_size = 128
  child_caps = shape[-2]
  parent_caps = shape[-1]
  # Unroll the probs along the spatial dimension
  # e.g. (64, 6, 6, 3*3, 8, 32) -> (64, 6*6, 3*3, 8, 32)
  probs_unroll = tf.reshape(
    probs,
    [-1, child_caps, parent_caps])

  # Create an index mapping each capsule to the correct sparse location
  # Each element of the index must contain [batch_position,
  # parent_space_position, child_sparse_position]
  # E.g. [63, 24, 49] maps image 63, parent space 24, sparse position 49
  child_sparse_idx = np.array([[0]])
  child_sparse_idx = child_sparse_idx[np.newaxis, ...]
  child_sparse_idx = np.tile(child_sparse_idx, [batch_size, 1, 1])

  parent_idx = np.arange(1)
  parent_idx = np.reshape(parent_idx, [-1, 1])
  parent_idx = np.repeat(parent_idx, 1)
  parent_idx = np.tile(parent_idx, batch_size)
  parent_idx = np.reshape(parent_idx, [batch_size, 1, 1])

  batch_idx = np.arange(batch_size)
  batch_idx = np.reshape(batch_idx, [-1, 1])
  batch_idx = np.tile(batch_idx, 1 * 1)
  batch_idx = np.reshape(batch_idx, [batch_size, 1, 1])

  # Combine the 3 coordinates
  indices = np.stack((batch_idx, parent_idx, child_sparse_idx), axis=3)
  indices = tf.constant(indices)

  # Convert each spatial location to sparse
  shape = [-1, child_caps, parent_caps]
  sparse = tf.scatter_nd(indices, probs_unroll, shape)

  # scatter_nd pads the output with zeros, but since we are operating
  # in log space, we need to replace 0 with log(0), or log(1e-9)
  zeros_in_log = tf.ones_like(sparse, dtype=tf.float32) * sparse_filler
  sparse = tf.where(tf.equal(sparse, 0.0), zeros_in_log, sparse)

  # Reshape
  # (64, 5*5, 7*7, 8, 32) -> (64, 6, 6, 14*14, 8, 32)
  sparse = tf.reshape(sparse, [-1, child_caps, parent_caps])

  return sparse

def test_to_sparse():
  inputs = tf.random.uniform([64, 1, 1, 1, 16, 5], 100, 200)
  res = to_sparse(inputs, [[1]])
  print(res)


def to_dense(sparse, spatial_routing_matrix):
  """Convert sparse back to dense along child_space dimension.

  Consider a sparse probs tensor of shape (64, 5, 5, 49, 8, 32).
  (batch_size, parent_space, parent_space, child_space*child_space, child_caps,
  parent_caps)
  The tensor contains all child capsules at every parent spatial location, but
  if the child does not route to the parent then it is just zero at that spot.
  Now we want to get back to the dense representation:
  (64, 5, 5, 49, 8, 32) -> (64, 5, 5, 9, 8, 32)

  Author:
    Ashley Gritzman 05/11/2018
  Args:
    sparse:
      the sparse representation of the probs tensor
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 49, 8, 32)
    spatial_routing_matrix:
      binary routing map with children as rows and parents as columns

  Returns:
    dense:
      the dense representation of the probs tensor
      (batch_size, parent_space, parent_space, kk, child_caps, parent_caps)
      (64, 5, 5, 9, 8, 32)
  """

  # Get shapes of probs
  shape = sparse.get_shape().as_list()
  child_caps = shape[-2]
  parent_caps = shape[-1]

  # Apply boolean_mask on axis 1 and 2
  # sparse_unroll: (64, 5*5, 49, 8, 32)
  # spatial_routing_matrix: (49, 25) -> (25, 49)
  # dense: (64, 5*5, 49, 8, 32) -> (64, 5*5*9, 8, 32)
  dense = tf.boolean_mask(sparse, tf.transpose(spatial_routing_matrix), axis=1)

  # Reshape
  dense = tf.reshape(dense, [-1, child_caps, parent_caps])
  return dense


def group_children_by_parent(bin_routing_map):
  """Groups children capsules by parent capsule.

  Rearrange the bin_routing_map so that each row represents one parent capsule,   and the entries in the row are indexes of the children capsules that route to   that parent capsule. This mapping is only along the spatial dimension, each
  child capsule along in spatial dimension will actually contain many capsules,   e.g. 32. The grouping that we are doing here tell us about the spatial
  routing, e.g. if the lower layer is 7x7 in spatial dimension, with a kernel of
  3 and stride of 1, then the higher layer will be 5x5 in the spatial dimension.
  So this function will tell us which children from the 7x7=49 lower capsules
  map to each of the 5x5=25 higher capsules. One child capsule can be in several
  different parent capsules, children in the corners will only belong to one
  parent, but children towards the center will belong to several with a maximum   of kernel*kernel (e.g. 9), but also depending on the stride.

  Author:
    Ashley Gritzman 19/10/2018
  Args:
    bin_routing_map:
      binary routing map with children as rows and parents as columns
  Returns:
    children_per_parents:
      parents are rows, and the indexes in the row are which children belong to       that parent
  """

  tmp = np.where(np.transpose(bin_routing_map))
  children_per_parent = np.reshape(tmp[1], [np.array(bin_routing_map).shape[1], -1])

  return children_per_parent

def test_group_c():
  res = group_children_by_parent([[1]])
  print(res)


def create_routing_map(child_space, k, s):
  """Generate TFRecord for train and test datasets from .mat files.

  Create a binary map where the rows are capsules in the lower layer (children)
  and the columns are capsules in the higher layer (parents). The binary map
  shows which children capsules are connected to which parent capsules along the   spatial dimension.

  Author:
    Ashley Gritzman 19/10/2018
  Args:
    child_space: spatial dimension of lower capsule layer
    k: kernel size
    s: stride
  Returns:
    binmap:
      A 2D numpy matrix containing mapping between children capsules along the
      rows, and parent capsules along the columns.
      (child_space^2, parent_space^2)
      (7*7, 5*5)
  """

  parent_space = int((child_space - k) / s + 1)
  binmap = np.zeros((child_space ** 2, parent_space ** 2))
  for r in range(parent_space):
    for c in range(parent_space):
      p_idx = r * parent_space + c
      for i in range(k):
        # c_idx stand for child_index; p_idx is parent_index
        c_idx = r * s * child_space + c * s + child_space * i
        binmap[(c_idx):(c_idx + k), p_idx] = 1
  return binmap