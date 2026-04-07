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
    def __init__(self, model, beta_1, beta_T, T, num_threshold, ddim_steps=50, eta=0.0):
        super().__init__()
        eps = 1e-20
        self.model = model
        self.T = T
        self.percentile = num_threshold
        self.ddim_steps = ddim_steps
        self.eta = eta

        # schedules
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar',
                             torch.sqrt(1. - alphas_bar + eps))

    # ----------------------------------------------------
    # x0 reconstruction (UNCHANGED)
    # ----------------------------------------------------
    def predict_x0_from_eps(self, x_t, t, eps):
        return (
            x_t - extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape) * eps
        ) / extract(self.sqrt_alphas_bar, t, x_t.shape)

    # ----------------------------------------------------
    # dynamic thresholding (UNCHANGED)
    # ----------------------------------------------------
    def dynamic_thresholding(self, x0):
        B = x0.shape[0]
        x0_flat = x0.view(B, -1)

        s = torch.quantile(x0_flat.abs(), self.percentile, dim=1)
        s = torch.maximum(s, torch.ones_like(s))
        s = s.view(B, *([1] * (x0.dim() - 1)))

        return torch.clamp(x0, -s, s) / s

    # ----------------------------------------------------
    # DDIM sampling step (REPLACES p_mean_variance)
    # ----------------------------------------------------
    def ddim_step(self, x_t, t, t_prev, cond, wind):
        eps = self.model(x_t, t, cond, wind)

        # 1. predict x0
        x0_pred = self.predict_x0_from_eps(x_t, t, eps)
        x0_pred = self.dynamic_thresholding(x0_pred)

        alpha_bar_t = extract(self.alphas_bar, t, x_t.shape)
        alpha_bar_prev = extract(self.alphas_bar, t_prev, x_t.shape)

        # 2. compute sigma (controls stochasticity)
        if self.eta > 0:
            sigma = self.eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t)
            ) * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
            noise = torch.randn_like(x_t)
        else:
            sigma = 0.0
            noise = 0.0

        # 3. DDIM update
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps

        x_prev = (
            torch.sqrt(alpha_bar_prev) * x0_pred +
            dir_xt +
            sigma * noise
        )

        return x_prev

    # ----------------------------------------------------
    # sampling loop (DDIM)
    # ----------------------------------------------------
    def forward(self, x_T, cond, wind):
        x_t = x_T

        times = torch.linspace(0, self.T - 1, self.ddim_steps).long().flip(0)
        #timesteps = (np.linspace(0, np.sqrt(T), steps) ** 2).astype(int) (quadratic time step) or cosine time step ??
        for i in range(len(times) - 1):
            t = x_t.new_full((x_t.shape[0],), times[i], dtype=torch.long)
            t_prev = x_t.new_full((x_t.shape[0],), times[i + 1], dtype=torch.long)

            x_t = self.ddim_step(x_t, t, t_prev, cond, wind)

            assert torch.isnan(x_t).sum() == 0, "NaN detected"

        return x_t