import torch
import torch.nn.functional as F
import torch.nn as nn


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer_ms(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, loss):
        super().__init__()
        eps = 1e-20
        self.model = model
        if loss == 'l2':
            self.loss = nn.MSELoss()
        elif loss == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar + eps))

    def forward(self, x_0, cond, wind):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        loss = F.mse_loss(self.model(x_t, t, cond, wind), noise, reduction='none')
        return loss


class GaussianDiffusionSampler_ms(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, num_threshold):
        super().__init__()
        eps = 1e-20
        self.model = model
        self.T = T
        self.percentile = num_threshold
        # schedules
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1.0)[:T]

        # forward-process helpers
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar',
                             torch.sqrt(1. - alphas_bar + eps))

        # posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar + eps)
        )

        self.register_buffer(
            'posterior_mean_coef1',
            self.betas * torch.sqrt(alphas_bar_prev) / (1. - alphas_bar + eps)
        )

        self.register_buffer(
            'posterior_mean_coef2',
            (1. - alphas_bar_prev) * torch.sqrt(alphas) / (1. - alphas_bar + eps)
        )

    # ----------------------------------------------------
    # x0 reconstruction (exact inversion of forward process)
    # ----------------------------------------------------
    def predict_x0_from_eps(self, x_t, t, eps):
        return (
            x_t - extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape) * eps
        ) / extract(self.sqrt_alphas_bar, t, x_t.shape)

    # ----------------------------------------------------
    # Imagen-style dynamic thresholding
    # ----------------------------------------------------
    def dynamic_thresholding(self, x0):
        B = x0.shape[0]
        x0_flat = x0.view(B, -1)

        s = torch.quantile(x0_flat.abs(), self.percentile, dim=1)
        s = torch.maximum(s, torch.ones_like(s))  # enforce s >= 1
        s = s.view(B, *([1] * (x0.dim() - 1))) #(B,C,H,W) => (B,1,1,1)

        return torch.clamp(x0, -s, s) / s

    # ----------------------------------------------------
    # p(x_{t-1} | x_t)
    # ----------------------------------------------------
    def p_mean_variance(self, x_t, t, cond, wind):
        eps = self.model(x_t, t, cond, wind)

        # 1. reconstruct x0
        x0_pred = self.predict_x0_from_eps(x_t, t, eps)

        # 2. dynamic thresholding
        x0_pred = self.dynamic_thresholding(x0_pred)

        # 3. posterior mean
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        var = extract(self.posterior_var, t, x_t.shape)
        return mean, var

    # ----------------------------------------------------
    # sampling loop
    # ----------------------------------------------------
    def forward(self, x_T, cond, wind):
        x_t = x_T

        for time_step in reversed(range(self.T)):
            t = x_t.new_full((x_t.shape[0],), time_step, dtype=torch.long)

            mean, var = self.p_mean_variance(x_t, t, cond, wind)

            if time_step > 0:
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(var) * noise
            else:
                x_t = mean

            assert torch.isnan(x_t).sum() == 0, "NaN detected"

        return x_t