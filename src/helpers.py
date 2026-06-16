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
    
    noise_component = torch.sqrt(measurement) * torch.randn_like(measurement)
    gaussian_noisy = (measurement + noise_component).clamp_min(0.0)
    


    return gaussian_noisy


def FRC(reconstruction_A, reconstruction_B, grid_spacing_object, num_points):
    nx, ny = reconstruction_A.shape

    fft_A = torch.fft.fftshift(torch.fft.fft2(reconstruction_A))
    fft_B = torch.fft.fftshift(torch.fft.fft2(reconstruction_B))

    Fx = torch.fft.fftshift(torch.fft.fftfreq(nx, d=grid_spacing_object, device=reconstruction_A.device))
    Fy = torch.fft.fftshift(torch.fft.fftfreq(ny, d=grid_spacing_object, device=reconstruction_A.device))

    FX, FY = torch.meshgrid(Fx, Fy, indexing="xy")
    freq_magnitudes = torch.sqrt(FX**2 + FY**2)

    freq_max = freq_magnitudes.max()
    freq_bins = torch.linspace(0, freq_max, num_points + 1, device=reconstruction_A.device)

    results = []
    freq_centers = []
    for i in range(num_points):
        # Compute the correlation on a circle (of a small radius)
        mask = (freq_magnitudes >= freq_bins[i]) & (freq_magnitudes < freq_bins[i+1])
        
        # Extract only the elements inside the ring 
        fft_A_masked = fft_A[mask]
        fft_B_masked = fft_B[mask]

        numerator = torch.sum(fft_A_masked * torch.conj(fft_B_masked))
        denominator = torch.sum(torch.abs(fft_A_masked)**2) * torch.sum(torch.abs(fft_B_masked)**2)
        
        results.append( torch.real(numerator) / (torch.sqrt(denominator) + 1e-12))

        freq_centers.append((freq_bins[i] + freq_bins[i+1]) / 2)

    # Return both the frequencies at the center of each ring and the FRC values
    return torch.stack(freq_centers), torch.stack(results)