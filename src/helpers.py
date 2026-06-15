import torch


def add_actuator_jitter(commands, max_residual_error: float):
    """
    commands: list of torch.Tensor, each tensor shape (num_actuators,)
    returns: list of torch.Tensor with per-actuator uniform jitter added
    """
    if max_residual_error == 0:
        return commands

    out = []
    for cmd in commands:
        # uniform jitter in [-max_residual_error, +max_residual_error]
        jitter = (torch.rand_like(cmd) * 2.0 - 1.0) * max_residual_error
        out_cmd = cmd + jitter

        # clamp the commands so they do not exceed the max actuator stroke. 
        out_cmd = torch.clamp(out_cmd, -1.0, 1.0)
        out.append(out_cmd)

    return out

def add_shot_noise(measurement):
    # mu = measurement.clamp_min(0).cpu()
    
    # # Use PyTorch's native Poisson distribution generator for physically accurate shot noise
    # # This correctly handles both high intensity (where it approaches Gaussian)
    # # AND low intensity (where it correctly generates integer 0, 1, 2... counts)
    # noisy_measurement = torch.randn(mu).to(measurement.device)
    
    # return noisy_measurement
    mu = measurement.clamp_min(0)
    eps = 1e-12

    noise = torch.sqrt(mu.clamp_min(eps)) * torch.randn_like(mu)
    return (noise).clamp_min(0.0)