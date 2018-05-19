from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.pad = filter_size // 2

        self.params['W1'] = np.random.randn(num_filters, 3, filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)

        size_after_filter = int(1 + (input_dim[1] + 2 * self.pad - filter_size))
        size_after_pool = size_after_filter // 2

        self.params['W2'] = np.random.randn(num_filters * size_after_pool ** 2, hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = X
        scores, conv_cache = conv_relu_forward(scores, W1, b1, conv_param)
        scores, pool_cache = max_pool_forward_fast(scores, pool_param)
        scores, af_relu_cache = affine_relu_forward(scores, W2, b2)
        scores, af_cache = affine_forward(scores, W3, b3)

        if y is None:
            return scores

        loss, grads = 0, {}


        loss, backward = softmax_loss(scores, y)

        backward, grads['W3'], grads['b3'] = affine_backward(backward, af_cache)
        backward, grads['W2'], grads['b2'] = affine_relu_backward(backward, af_relu_cache)
        backward = max_pool_backward_fast(backward, pool_cache)
        backward, grads['W1'], grads['b1'] = conv_relu_backward(backward, conv_cache)

        loss += 0.5 * self.reg * np.mean(self.params['W1'] ** 2)
        loss += 0.5 * self.reg * np.mean(self.params['W2'] ** 2)
        loss += 0.5 * self.reg * np.mean(self.params['W3'] ** 2)
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']
        grads['W3'] += self.reg * self.params['W3']

        return loss, grads
