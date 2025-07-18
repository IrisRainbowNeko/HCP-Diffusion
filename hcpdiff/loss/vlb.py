import torch
from torch import nn
import numpy as np

class VLBLoss(nn.Module):
    need_sigma = True
    need_timesteps = True
    need_sampler = True
    var_pred = True

    def __init__(self, loss, weight: float = 1.):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.
        """

        return 0.5*(-1.0+logvar2-logvar1+(logvar1-logvar2).exp()+((mean1-mean2)**2)*(-logvar2).exp())

    def forward(self, input: torch.Tensor, target: torch.Tensor, sigma, timesteps: torch.Tensor, x_t: torch.Tensor, sampler):
        eps_pred, var_pred = input.chunk(2, dim=1)
        x0_pred = sampler.eps_to_x0(eps_pred, x_t, sigma)

        true_mean = sampler.sigma_scheduler.get_post_mean(timesteps, target, x_t)
        true_logvar = sampler.sigma_scheduler.get_post_log_var(timesteps, ndim=input.ndim)

        pred_mean = sampler.sigma_scheduler.get_post_mean(timesteps, x0_pred, x_t)
        pred_logvar = sampler.sigma_scheduler.get_post_log_var(timesteps, ndim=input.ndim, x_t_var=var_pred)

        kl = self.normal_kl(true_mean, true_logvar, pred_mean, pred_logvar)
        kl = kl.mean(dim=(1,2,3))/np.log(2.0)

        decoder_nll = -self.discretized_gaussian_log_likelihood(target, means=pred_mean, log_scales=0.5*pred_logvar)
        assert decoder_nll.shape == target.shape
        decoder_nll = decoder_nll.mean(dim=(1,2,3))/np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((timesteps == 0), decoder_nll, kl)

        return self.weight*output

    def approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal.
        """
        return 0.5*(1.0+torch.tanh(np.sqrt(2.0/np.pi)*(x+0.044715*torch.pow(x, 3))))

    def discretized_gaussian_log_likelihood(self, x, *, means, log_scales):
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.
        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
        assert x.shape == means.shape == log_scales.shape
        centered_x = x-means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv*(centered_x+1.0/255.0)
        cdf_plus = self.approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv*(centered_x-1.0/255.0)
        cdf_min = self.approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0-cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus-cdf_min
        log_probs = torch.where(
            x<-0.999,
            log_cdf_plus,
            torch.where(x>0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == x.shape
        return log_probs
