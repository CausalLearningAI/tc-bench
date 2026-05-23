import torch
import math
class WeatherNormalizer:
    def __init__(self, wind_mean, wind_std, precip_mean, precip_std):
        self.wind_mean = wind_mean.view(1, 1, 1, 2).cuda()
        self.wind_std = wind_std.view(1, 1, 1, 2).cuda()
        self.precip_mean = precip_mean.cuda()
        self.precip_std = precip_std.cuda() #TODO: flexible device handling

    def __call__(self, x):
        x = x.clone()
        x[..., 0:2] = (x[..., 0:2] - self.wind_mean) / self.wind_std
        precip = torch.log1p(x[..., 2])
        x[..., 2] = (precip - self.precip_mean) / self.precip_std
        return x

    def inverse(self, x):
        x = x.clone().type_as(self.wind_mean)
        x[..., 0:2] = x[..., 0:2] * self.wind_std + self.wind_mean
        precip = x[..., 2] * self.precip_std + self.precip_mean
        x[..., 2] = torch.expm1(precip)
        return x
    
    def inverse_std(self, m, s):
        s = s.clone().type_as(self.wind_mean)
        s[..., 0:2] = s[..., 0:2] * self.wind_std
        
        s_precip = s[..., 2]
        m_precip = m[..., 2]
        var = (torch.exp(s_precip**2) - 1) * torch.exp(2 * m_precip + s_precip**2)
        s[..., 2] = torch.sqrt(var)
    
        return s

def save_normalization_stats(path, wind_mean, wind_std, precip_mean, precip_std):
    stats = {
        "wind_mean": wind_mean,
        "wind_std": wind_std,
        "precip_mean": precip_mean,
        "precip_std": precip_std
    }
    torch.save(stats, path)

def load_normalization_stats(path):
    stats = torch.load(path, weights_only=True)
    return (
        stats["wind_mean"],
        stats["wind_std"],
        stats["precip_mean"],
        stats["precip_std"]
    )


class LogMinMaxScaler(torch.nn.Module):
    """
    Precipitation normalizer:
      1. log-transform with epsilon
      2. min-max scale to [-1, 1]
    """

    def __init__(self, max_val=0.5436816, min_val=0.0, eps: float = 1e-6):
        """
        Args:
            data: reference tensor (full dataset or representative sample) used to fit min/max.
            eps: small constant to avoid log(0).
        """
        super().__init__()
        self.eps = eps

        # Compute reference min and max in log space
        log_min = math.log(min_val + eps) - math.log(eps)
        log_max = math.log(max_val + eps) - math.log(eps)
        self.register_buffer("min_val", torch.tensor(log_min))
        self.register_buffer("max_val", torch.tensor(log_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize: raw precip -> log space -> [-1, 1]
        """
        self.max_val = self.max_val.type_as(x)
        self.min_val = self.min_val.type_as(x)
        log_x = torch.log(x + self.eps) - torch.log(torch.tensor(self.eps, device=x.device))
        scaled = (log_x - self.min_val) / (self.max_val - self.min_val)
        return scaled * 2 - 1

    def inverse(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize: [-1, 1] -> log space -> raw precip
        """
        self.max_val = self.max_val.type_as(x_norm)
        self.min_val = self.min_val.type_as(x_norm)
        log_x = (x_norm + 1) / 2 * (self.max_val - self.min_val) + self.min_val
        res = torch.exp(log_x + torch.log(torch.tensor(self.eps, device=x_norm.device))) - self.eps
        return res*1000  # convert back to mm/day
