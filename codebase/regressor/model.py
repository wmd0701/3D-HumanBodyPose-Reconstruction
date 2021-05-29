from abc import ABC

from torch import nn
from .body_model import BodyModel
from .resnet import load_resnet
from .iterative_regressor import IterativeRegressor
from .util import load_smpl_init_params


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

        self._build_net(cfg)

    def _build_net(self, cfg):
        """ Creates NNs. """
        
        # Use resnet as feature extractor, replace last layer with identity
        self.backbone = load_resnet()
        self.backbone.fc = nn.Identity()


        # Define iterative regression inspired by HMR
        fc_layers = [2048 + 82, 1024, 1024, 82]
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]
        iterations = 3
        initialization = load_smpl_init_params()
    
        self.regressor = IterativeRegressor(
            fc_layers,
            use_dropout,
            drop_prob,
            use_ac_func,
            iterations,
            initialization,
            self.batch_size)
        
        # Layers to extract final outputs. Dimension + 10 + 63 + 6 = 82
        self.nn_root_orient = nn.Linear(82, 3)
        self.nn_betas = nn.Linear(82, 10)
        self.nn_pose_body = nn.Linear(82, 63)
        self.nn_pose_hand = nn.Linear(82, 6)

    def forward(self, input_data):
        """ Fwd pass.

        Returns (dict):
            mesh vertices (torch.Tensor): (B, 6890, 3)
        """
        image_crop = input_data['image_crop']
        root_loc = input_data['root_loc']

        img_encoding = self.backbone(image_crop)

        # regress parameters
        params = self.regressor(img_encoding)
        root_orient = self.nn_root_orient(params)
        betas = self.nn_betas(params)
        pose_body = self.nn_pose_body(params)
        pose_hand = self.nn_pose_hand(params)

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
