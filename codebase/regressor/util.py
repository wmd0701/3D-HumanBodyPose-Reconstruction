import torch
import numpy as np


@torch.no_grad()
def proj_vertices(points, images, fx, fy, cx, cy):
    """ Projects 3D points onto image plane.

    Args:
        points (torch.Tensor): 3D points (B, N, 3)
        images:  3D points (B, 3, H, W)
        fx: focal length (B,)
        fy: focal length (B,)
        cx: focal length (B,)
        cy: focal length (B,)

    Returns:
        rendered_img (torch.Tensor): Images with marked vertex locations (B, 3, H, W).
    """
    B, H, W = images.shape[0], images.shape[2], images.shape[3]
    device = images.device

    # compute pixel locations
    x = points[..., 0] / points[..., 2] * fx.view(-1, 1) + cx.view(-1, 1)
    y = points[..., 1] / points[..., 2] * fy.view(-1, 1) + cy.view(-1, 1)

    # round pixel locations
    x = (x+0.5).int().long()
    y = (y+0.5).int().long()

    # discard invalid pixels
    valid = ~((x < 0) | (y < 0) | (x >= W) | (y >= H))
    idx = (W*y + x) + W*H*torch.arange(0, B, device=device, dtype=torch.long).view(B, 1)
    idx = idx.view(-1)[valid.view(-1)]

    # mark selected pixels
    rendered_img = torch.clone(images).permute(0, 2, 3, 1)  # -> (B, H, W, 3)
    rendered_img = rendered_img.contiguous().view(-1, 3)
    rendered_img[idx] = torch.IntTensor([0, 0, 255]).to(device=device, dtype=rendered_img.dtype)
    rendered_img = rendered_img.view(B, H, W, 3).contiguous()

    rendered_img = rendered_img.permute(0, 3, 1, 2)

    return rendered_img

def load_smpl_mean_params(path):
    init_params = np.zeros(3 + 69 + 10, dtype=np.float)
    mean_values_dict = np.load(path, allow_pickle=True)['arr_0'].ravel()[0]

    init_params[:3] = mean_values_dict['root_orient_mean'][:]
    init_params[3:66] = mean_values_dict['pose_body_mean'][:]
    init_params[66:72] = mean_values_dict['pose_hand_mean'][:]
    init_params[72:82] = mean_values_dict['betas_mean'][:]

    return init_params