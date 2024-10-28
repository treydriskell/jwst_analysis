import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.data.ops.iterator_ops import OwnedIterator, IteratorSpec
from tensorflow_probability.python.internal import dtype_util, tensor_util, parameter_properties, reparameterization
import gpflow as gpf

rng = np.random.default_rng()

half = np.float64(0.5)
one = np.float64(1.0)
two = np.float64(2.0)
pi = np.float64(math.pi)
sqrt_two = tf.math.sqrt(two)
sqrt_two_pi  = tf.math.sqrt(two*pi)


def linear_loss(model, data):
    # training_loss
    training_loss = model.training_loss(data)/2.0e5
    # print(training_loss)
    return tf.math.exp(training_loss)


def linear_loss_closure(model, data, compile=False, **closure_kwargs):
    training_loss = linear_loss
    if isinstance(data, OwnedIterator):
        if compile:
        # lambda because: https://github.com/GPflow/GPflow/issues/1929
            training_loss_lambda = lambda d: training_loss(model,d)
            input_signature = [data.element_spec]
            training_loss = tf.function(training_loss_lambda, input_signature=input_signature)
        def closure() -> tf.Tensor:
            assert isinstance(data, OwnedIterator)  # Hint for mypy.
            batch = next(data)
            return training_loss(model,batch)
    else:
        def closure() -> tf.Tensor:
            return training_loss(model,data)

        if compile:
            closure = tf.function(closure)
    return closure


class CustomDataLoader(OwnedIterator):
    def __init__(self, X, Y):
        m_min = int(np.amin(X[:,0]))
        m_max = int(round(np.amax(X[:,0])))
        lmhs = np.arange(m_min, m_max+1)
        self.nm = len(lmhs)-1
        zs = [8.0, 12.0, 16.0]
        self.nz = 3
        self.samples_per_bin = 100 # 500
        self.batch = self.nz*self.nm*self.samples_per_bin
        self.split_X = []
        self.split_Y = []
        self.length = len(X) // self.batch
        
        for i in range(self.nz):
            zidx = X[:,1]==zs[i]
            zX = X[zidx,:]
            zY = Y[zidx]
            self.split_X.append([])
            self.split_Y.append([])
            for j in range(self.nm):
                midx = (zX[:,0]>=lmhs[j]) & (zX[:,0]<lmhs[j+1])
                self.split_X[i].append(zX[midx,:]) 
                self.split_Y[i].append(zY[midx])
    
    @property
    def element_spec(self):
        return (tf.TensorSpec(shape=(None, 2), dtype=tf.float64, name=None), tf.TensorSpec(shape=(None, 1), dtype=tf.float64, name=None))
    
    def _type_spec(self):
        return IteratorSpec(self.element_spec)

    def __len__(self):
        return self.length
    
    def __next__(self):
        sample_X = np.zeros((self.batch,2))
        sample_Y = np.zeros((self.batch,1))
        for i in range(self.nz):
            for j in range(self.nm):
                samples = min(self.samples_per_bin, len(self.split_X[i][j]))
                i_idx = j*samples + self.nm*i*samples
                f_idx = (j+1)*samples + self.nm*i*samples
                idxs = rng.choice(np.arange(len(self.split_X[i][j])), size=samples, replace=False)
                # sample_X[i_idx:f_idx,:] = rng.choice(self.split_X[i][j], size=self.sample_per_bin,
                #     replace=False)
                sample_X[i_idx:f_idx,:] = self.split_X[i][j][idxs]
                sample_Y[i_idx:f_idx,:] = self.split_Y[i][j][idxs]
        return tf.convert_to_tensor(sample_X, dtype=tf.float64), tf.convert_to_tensor(sample_Y, dtype=tf.float64) # tf.convert_to_tensor(sample_X, sample_Y) #tf.convert_to_tensor(sample_X), tf.convert_to_tensor(sample_Y)
    
    def get_next(self):
        return next(self)

    def get_next_as_optional(self):
        return tf.experimental.Optional(self.get_next())


class TwoPieceNormalLikelihood(gpf.likelihoods.multilatent.MultiLatentTFPConditional):
    def __init__(self, **kwargs):
        distribution_class = tfp.distributions.TwoPieceNormal
        self.scale_transform = tfp.bijectors.Softplus() #tfp.bijectors.Softplus() #tfp.bijectors.Softplus() #tfp.bijectors.Softplus() #tfp.bijectors.Exp()
        # low = tf.constant(0., dtype=tf.float64) 
        # high = tf.constant(2., dtype=tf.float64) 
        # self.skew_transform = tfp.bijectors.Sigmoid(low, high) 
        self.skew_transform = tfp.bijectors.Exp() #tfp.bijectors.Softplus()
        # TODO: okay maybe not the right choice, want something that maps most stuff close to 1, not sharp transition around 1

        def conditional_distribution(F):
            loc = F[..., :1]
            scale = self.scale_transform(F[..., 1:2])
            skewness = self.skew_transform(F[..., 2:])
            return distribution_class(loc, scale, skewness)

        super().__init__(
            latent_dim=3,
            conditional_distribution=conditional_distribution,
            **kwargs,
        )


class JohnsonSULikelihood(gpf.likelihoods.multilatent.MultiLatentTFPConditional):
    def __init__(self, **kwargs):
        distribution_class = tfp.distributions.JohnsonSU
        low = tf.constant(1.0e-3, dtype=tf.float64)
        # self.skewness_transform = tfp.bijectors.Softplus(low=low)
        self.tailweight_transform = tfp.bijectors.Softplus(low=low)
        # self.loc_transform = tfp.bijectors.Softplus(low=low)
        self.scale_transform = tfp.bijectors.Softplus(low=low)
        # TODO: okay maybe not the right choice, want something that maps most stuff close to 1, not sharp transition around 1

        def conditional_distribution(F):
            s = F[..., :1] # self.skewness_transform(F[..., :1]) 
            t = self.tailweight_transform(F[..., 1:2]) 
            loc = F[..., 2:3] # self.loc_transform(F[..., 2:3])
            scale = self.scale_transform(F[..., 3:])
            return distribution_class(s, t, loc, scale)

        super().__init__(
            latent_dim=4,
            conditional_distribution=conditional_distribution,
            **kwargs,
        )


class SoftTruncatedNormal(tfp.distributions.Distribution):
    """Based on tfp.distributions.TruncatedNormal 
    https://github.com/tensorflow/probability/blob/v0.22.0/tensorflow_probability/python/distributions/truncated_normal.py#L72-L401
    
    Note: This is only a one-sided constraint!"""
    def __init__(self,
                 loc,
                 scale,
                 low,
                 eta=20,
                #  high,
                 validate_args=False,
                 allow_nan_stats=False, # True
                 name='SoftTruncatedNormal'):
        dtype = tf.float64
        self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
        self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
        self._low = tensor_util.convert_nonref_to_tensor(
          low, name='low', dtype=dtype)
        self._eta = tensor_util.convert_nonref_to_tensor(
          eta, name='eta', dtype=dtype)
        self._norm = None
        self._var = None
        self._m = None
        # self._high = tensor_util.convert_nonref_to_tensor(
        #   high, name='high', dtype=dtype)
        super(SoftTruncatedNormal, self).__init__(
          dtype=dtype,
          name=name,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats)

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for the scale."""
        return self._scale

    @property
    def low(self):
        return self._low

    @property
    def eta(self):
        """ Controls how sharp the cutoff is."""
        return self._eta
    

    def _loc_scale_low_eta(self, loc=None, scale=None, low=None, eta=None):
        loc = tf.convert_to_tensor(self.loc if loc is None else loc)
        scale = tf.convert_to_tensor(self.scale if scale is None else scale)
        low = tf.convert_to_tensor(self.low if low is None else low)
        eta = tf.convert_to_tensor(self.eta if eta is None else eta)
        return loc, scale, low, eta

    def _sample_n(self):
        pass
    
    @property
    def norm(self):
        if self._norm is None:
            loc, scale, low, eta = self._loc_scale_low_eta()
            norm_erf = tf.math.erf(eta*(loc-low)/tf.math.sqrt(
                two*(one+tf.square(eta*scale))))
            self._norm = scale * tf.math.sqrt(two*pi) * half * (one+norm_erf)
        return self._norm

    def _log_prob(self, x): # or _prob
        loc, scale, low, eta = self._loc_scale_low_eta()
        log_prob = -(half * tf.square((x - loc) / scale))
        log_prob += tf.math.log(half*(one+tf.math.erf(eta * (x-low)/sqrt_two)))
        return log_prob - tf.math.log(self.norm)

    def _event_shape(self):
        return tf.TensorShape([])
        
    # def _event_shape_tensor(self):
    #     pass

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            loc=parameter_properties.ParameterProperties(),
            scale=parameter_properties.ParameterProperties(),
            low=parameter_properties.ParameterProperties(),
            eta=parameter_properties.ParameterProperties()
            )

    def _mean(self):
        if self._m is None:
            loc, scale, low, eta = self._loc_scale_low_eta()
            b = eta*scale
            a = eta*(loc-low)
            t = tf.math.sqrt(one+tf.square(b))
            pdf_at = tf.math.exp(-half*tf.square(a/t))/sqrt_two_pi/scale
            cdf_at = loc * half * (one + tf.math.erf(a/t/sqrt_two)) 
            self._m = loc * cdf_at + b * pdf_at / t
        return self._m

    def _mode(self):
        return self.loc # by assumption low < loc, so loc is always the mode

    def _variance(self): # depression, hard to calculate numerically
        """ This is hard to calculate numerically. I am going to cheat
        and use the trunacated normal variance instead. :) """
        if self._var is None:
            loc, scale, low, eta = self._loc_scale_low_eta()
            # np_dtype = dtype_util.as_numpy_dtype(loc.dtype)
            # result = tf.math.exp(-half*tf.square((low-loc)/scale))*(low+loc)*scale/sqrt_two_pi
            # result += half*(tf.square(loc)+tf.square(scale))*(one-tf.math.erf(low-loc/scale/sqrt_two_pi))
            # truncated_mean = scale*tf.math.exp(-half*tf.square((low-loc)/scale))/sqrt_two_pi
            # truncated_mean += half*loc*(one-tf.math.erf((low-loc)/scale/sqrt_two_pi))
            # self._var = result - tf.square(truncated_mean)
            alpha = (low - loc) / scale
            phi_alpha = tf.math.exp(-half*tf.square(alpha))/sqrt_two_pi/scale
            Z = half * (one -  tf.math.erf(alpha / sqrt_two))
            self._var = tf.square(scale) * (one + alpha * phi_alpha / Z - tf.square(phi_alpha / Z))
        return self._var

    # def _log_cdf(self): #, _cdf, _survival_function, or _log_survival_function
    #     pass

    # etc.


class SoftTruncatedNormalLikelihood(gpf.likelihoods.multilatent.MultiLatentTFPConditional):
    def __init__(self, eta, **kwargs):
        distribution_class = SoftTruncatedNormal
        self.scale_transform = tfp.bijectors.Softplus() #tfp.bijectors.Softplus() #tfp.bijectors.Softplus() #tfp.bijectors.Softplus() #tfp.bijectors.Exp()
        self.eta = tf.constant(eta, dtype=tf.float64) # doesn't matter what this value is
        # self.loc_transform = tfp.bijectors.Softplus(low=tf.constant(1.0, dtype=tf.float64))
        self.low_transform = tfp.bijectors.Softplus()

        def conditional_distribution(F):
            loc = F[..., :1]
            scale = self.scale_transform(F[..., 1:2])
            low = loc - self.low_transform(F[..., 2:]) - 1.0e-1
            return distribution_class(loc=loc, scale=scale, low=low, eta=self.eta)

        super().__init__(
            latent_dim=3,
            conditional_distribution=conditional_distribution,
            **kwargs,
        )


class TruncatedNormalLikelihood(gpf.likelihoods.multilatent.MultiLatentTFPConditional):
    def __init__(self, high, **kwargs):
        distribution_class = tfp.distributions.TruncatedNormal
        self.scale_transform = tfp.bijectors.Softplus() #tfp.bijectors.Softplus() #tfp.bijectors.Softplus() #tfp.bijectors.Softplus() #tfp.bijectors.Exp()
        self.high = tf.constant(high, dtype=tf.float64) # doesn't matter what this value is
        self.loc_transform = tfp.bijectors.Softplus(low=tf.constant(1.0, dtype=tf.float64))
        self.low_transform = tfp.bijectors.Softplus(low=tf.constant(1.0, dtype=tf.float64))
        # self.loc_transform = tfp.bijectors.Chain([
        #     # tfp.bijectors.Shift(self.high),
        #     # tfp.bijectors.Scale(tf.constant(-1, dtype=tf.float64)),
        #     tfp.bijectors.Softplus(),
        #     ]
        # )
        # self.low_transform = tfp.bijectors.Chain([
        #     # tfp.bijectors.Scale(tf.constant(-1, dtype=tf.float64)),
        #     tfp.bijectors.Softplus(),
        #     ]
        # ) # low=tf.constant(5, dtype=tf.float64)

        def conditional_distribution(F):
            loc = self.high - self.loc_transform(F[..., :1])
            scale = self.scale_transform(F[..., 1:2])
            low = loc - self.low_transform(F[..., 2:])
            return distribution_class(loc=loc, scale=scale, low=low, high=self.high, validate_args=True, allow_nan_stats=False)

        super().__init__(
            latent_dim=3,
            conditional_distribution=conditional_distribution,
            **kwargs,
        )

def normal_pdf(x):
    return np.exp(-0.5*x**2)/np.sqrt(2.0*np.pi)

def TwoPieceNormalPDF(x, loc, scale, skewness):
    k = (2 * skewness) / ((1 + skewness**2) * scale)
    y = (x - loc) / scale
    idx = x < loc
    pdf = np.zeros_like(x)
    pdf[idx] = k * normal_pdf(y * skewness) 
    nidx = np.logical_not(idx)
    pdf[nidx] = k * normal_pdf(y / skewness) 
    return pdf
    
    