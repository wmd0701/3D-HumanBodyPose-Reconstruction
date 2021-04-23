from abc import ABC

from torch import nn
from .body_model import BodyModel


class BaseModel(nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # parameters
        self.bm_path = cfg['data']['bm_path']
        self.in_ch = cfg['model'].get('in_ch', 3)
        self.out_ch = cfg['model'].get('out_ch', 70)
        self.img_resolution = cfg['data']['resy'], cfg['data']['resx']

        self.device = cfg.get('device', 'cuda')
        self.batch_size = cfg['training']['batch_size']

        # body_model
        self.body_model = BodyModel(
            bm_path=self.bm_path,
            num_betas=10,
            batch_size=self.batch_size,
        ).to(device=self.device)

    @staticmethod
    def create_model(cfg):
        model_name = cfg['model']['name']
        if model_name == 'conv':
            model = ConvModel(cfg)
        else:
            raise Exception(f'Model `{model_name}` is not defined.')

        return model

    def get_vertices(self, root_loc, root_orient, betas, pose_body, pose_hand):
        """ Fwd pass through the parametric body model to obtain mesh vertices.

        Args:
            root_loc (torch.Tensor): Root location (B, 10).
            root_orient (torch.Tensor): Root orientation (B, 3).
            betas (torch.Tensor): Shape coefficients (B, 10).
            pose_body (torch.Tensor): Body joint rotations (B, 21*3).
            pose_hand (torch.Tensor): Hand joint rotations (B, 2*3).

        Returns:
            mesh vertices (torch.Tensor): (B, 6890, 3)
        """
        body = self.body_model(
            trans=root_loc,
            root_orient=root_orient,
            pose_body=pose_body,
            pose_hand=pose_hand,
            betas=betas
        )
        vertices = body.v
        return vertices


class ConvModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.backbone_f_len = cfg['model'].get('backbone_f_len', 500)
        self._build_net()

    def _build_net(self):
        """ Creates NNs. """
        fc_in_ch = 1*(self.img_resolution[0]//2**3)*(self.img_resolution[1]//2**3)
        self.backbone = nn.Sequential(
            nn.Conv2d(self.in_ch, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(5, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(5, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),
            nn.Conv2d(5, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.ReLU(),
            # flattening
            nn.Flatten(),
            nn.Linear(fc_in_ch, self.backbone_f_len)
        )

        self.nn_root_orient = nn.Linear(self.backbone_f_len, 3)
        self.nn_betas = nn.Linear(self.backbone_f_len, 10)
        self.nn_pose_body = nn.Linear(self.backbone_f_len, 63)
        self.nn_pose_hand = nn.Linear(self.backbone_f_len, 6)

    def forward(self, input_data):
        """ Fwd pass.

        Returns (dict):
            mesh vertices (torch.Tensor): (B, 6890, 3)
        """
        image_crop = input_data['image_crop']
        root_loc = input_data['root_loc']

        img_encoding = self.backbone(image_crop)

        # regress parameters
        root_orient = self.nn_root_orient(img_encoding)
        betas = self.nn_betas(img_encoding)
        pose_body = self.nn_pose_body(img_encoding)
        pose_hand = self.nn_pose_hand(img_encoding)

        # regress vertices
        vertices = self.get_vertices(root_loc, root_orient, betas, pose_body, pose_hand)
        predictions = {
            'vertices': vertices,
            'root_loc': root_loc,
            'root_orient': root_orient,
            'betas': betas,
            'pose_body': pose_body,
            'pose_hand': pose_hand
        }
        return predictions
