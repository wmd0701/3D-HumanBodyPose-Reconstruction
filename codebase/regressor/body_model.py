from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BodyModel(nn.Module):

    def __init__(self,
                 bm_path,
                 params=None,
                 num_betas=10,
                 batch_size=1, v_template=None,
                 num_dmpls=None, path_dmpl=None,
                 num_expressions=10,
                 use_posedirs=True,
                 dtype=torch.float32):

        super(BodyModel, self).__init__()

        '''
        :param bm_path: path to a SMPL model as pkl file
        :param num_betas: number of shape parameters to include.
                if betas are provided in params, num_betas would be overloaded with number of thoes betas
        :param batch_size: number of smpl vertices to get
        :param device: default on gpu
        :param dtype: float precision of the compuations
        :return: verts, trans, pose, betas
        '''

        self.dtype = dtype

        if params is None: params = {}

        # -- Load SMPL params --
        if '.npz' in bm_path:
            smpl_dict = np.load(bm_path, encoding='latin1')
        else:
            raise ValueError('bm_path should be either a .pkl nor .npz file')

        njoints = smpl_dict['posedirs'].shape[2] // 3
        self.model_type = {69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano'}[njoints]

        assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'mano'], ValueError(
            'model_type should be in smpl/smplh/smplx/mano.')

        self.use_dmpl = False
        if num_dmpls is not None:
            if path_dmpl is not None:
                self.use_dmpl = True
            else:
                raise (ValueError('path_dmpl should be provided when using dmpls!'))

        if self.use_dmpl and self.model_type in ['smplx', 'mano']: raise (
            NotImplementedError('DMPLs only work with SMPL/SMPLH models for now.'))

        # Mean codebase vertices
        if v_template is None:
            v_template = np.repeat(smpl_dict['v_template'][np.newaxis], batch_size, axis=0)
        else:
            v_template = np.repeat(v_template[np.newaxis], batch_size, axis=0)

        self.register_buffer('v_template', torch.tensor(v_template, dtype=dtype))

        self.register_buffer('f', torch.tensor(smpl_dict['f'].astype(np.int32), dtype=torch.int32))

        if len(params):
            if 'betas' in params.keys():
                num_betas = params['betas'].shape[1]
            if 'dmpls' in params.keys():
                num_dmpls = params['dmpls'].shape[1]

        num_total_betas = smpl_dict['shapedirs'].shape[-1]
        if num_betas < 1:
            num_betas = num_total_betas

        shapedirs = smpl_dict['shapedirs'][:, :, :num_betas]
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=dtype))

        if self.model_type == 'smplx':
            begin_shape_id = 300 if smpl_dict['shapedirs'].shape[-1] > 300 else 10
            exprdirs = smpl_dict['shapedirs'][:, :, begin_shape_id:(begin_shape_id + num_expressions)]
            self.register_buffer('exprdirs', torch.tensor(exprdirs, dtype=dtype))

            expression = torch.tensor(np.zeros((batch_size, num_expressions)), dtype=dtype, requires_grad=True)
            self.register_parameter('expression', nn.Parameter(expression, requires_grad=True))

        if self.use_dmpl:
            dmpldirs = np.load(path_dmpl)['eigvec']

            dmpldirs = dmpldirs[:, :, :num_dmpls]
            self.register_buffer('dmpldirs', torch.tensor(dmpldirs, dtype=dtype))

        # Regressor for joint locations given shape - 6890 x 24
        self.register_buffer('J_regressor', torch.tensor(smpl_dict['J_regressor'], dtype=dtype))

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        if use_posedirs:
            posedirs = smpl_dict['posedirs']
            posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
            self.register_buffer('posedirs', torch.tensor(posedirs, dtype=dtype))
        else:
            self.posedirs = None

        # indices of parents for each joints
        kintree_table = smpl_dict['kintree_table'].astype(np.int32)
        self.register_buffer('kintree_table', torch.tensor(kintree_table, dtype=torch.int32))

        # LBS weights
        # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
        weights = smpl_dict['weights']
        self.register_buffer('weights', torch.tensor(weights, dtype=dtype))

        if 'trans' in params.keys():
            trans = params['trans']
        else:
            trans = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('trans', nn.Parameter(trans, requires_grad=True))

        # root_orient
        # if self.model_type in ['smpl', 'smplh']:
        root_orient = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('root_orient', nn.Parameter(root_orient, requires_grad=True))

        # pose_body
        if self.model_type in ['smpl', 'smplh', 'smplx']:
            if 'pose_body' in params.keys():
                pose_body = params['pose_body']
            else:
                pose_body = torch.tensor(np.zeros((batch_size, 63)), dtype=dtype, requires_grad=True)
            self.register_parameter('pose_body', nn.Parameter(pose_body, requires_grad=True))

        # pose_hand
        if 'pose_hand' in params.keys():
            pose_hand = params['pose_hand']
        else:
            if self.model_type in ['smpl']:
                pose_hand = torch.tensor(np.zeros((batch_size, 1 * 3 * 2)), dtype=dtype, requires_grad=True)
            elif self.model_type in ['smplh', 'smplx']:
                pose_hand = torch.tensor(np.zeros((batch_size, 15 * 3 * 2)), dtype=dtype, requires_grad=True)
            elif self.model_type in ['mano']:
                pose_hand = torch.tensor(np.zeros((batch_size, 15 * 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('pose_hand', nn.Parameter(pose_hand, requires_grad=True))

        # face poses
        if self.model_type == 'smplx':
            pose_jaw = torch.tensor(np.zeros((batch_size, 1 * 3)), dtype=dtype, requires_grad=True)
            self.register_parameter('pose_jaw', nn.Parameter(pose_jaw, requires_grad=True))
            pose_eye = torch.tensor(np.zeros((batch_size, 2 * 3)), dtype=dtype, requires_grad=True)
            self.register_parameter('pose_eye', nn.Parameter(pose_eye, requires_grad=True))

        if 'betas' in params.keys():
            betas = params['betas']
        else:
            betas = torch.tensor(np.zeros((batch_size, num_betas)), dtype=dtype, requires_grad=True)
        self.register_parameter('betas', nn.Parameter(betas, requires_grad=True))

        if self.use_dmpl:
            if 'dmpls' in params.keys():
                dmpls = params['dmpls']
            else:
                dmpls = torch.tensor(np.zeros((batch_size, num_dmpls)), dtype=dtype, requires_grad=True)
            self.register_parameter('dmpls', nn.Parameter(dmpls, requires_grad=True))
        self.batch_size = batch_size

    def r(self):
        from human_body_prior.tools.omni_tools import copy2cpu as c2c
        return c2c(self.forward().v)

    def forward(self, root_orient=None, pose_body=None, pose_hand=None, pose_jaw=None, pose_eye=None, betas=None,
                trans=None, dmpls=None, expression=None, return_dict=False, v_template=None, clothed_v_template=None,
                **kwargs):
        """
        :param root_orient: Nx3
        :param pose_body:
        :param pose_hand:
        :param pose_jaw:
        :param pose_eye:
        :param kwargs:
        :return:
        """
        assert not (v_template is not None and betas is not None), ValueError(
            'vtemplate and betas could not be used jointly.')
        assert self.model_type in ['smpl', 'smplh', 'smplx', 'mano', 'mano'], ValueError(
            'model_type should be in smpl/smplh/smplx/mano')
        if root_orient is None:  root_orient = self.root_orient
        if self.model_type in ['smplh', 'smpl']:
            if pose_body is None:  pose_body = self.pose_body
            if pose_hand is None:  pose_hand = self.pose_hand
        elif self.model_type == 'smplx':
            if pose_body is None:  pose_body = self.pose_body
            if pose_hand is None:  pose_hand = self.pose_hand
            if pose_jaw is None:  pose_jaw = self.pose_jaw
            if pose_eye is None:  pose_eye = self.pose_eye
        elif self.model_type in ['mano', 'mano']:
            if pose_hand is None:  pose_hand = self.pose_hand

        if pose_hand is None:  pose_hand = self.pose_hand

        if trans is None: trans = self.trans
        if v_template is None: v_template = self.v_template
        if betas is None: betas = self.betas

        if v_template.size(0) != pose_body.size(0):
            v_template = v_template[:pose_body.size(0)]  # this is fine since actual batch size will
            # only be equal to or less than specified batch
            # size

        if self.model_type in ['smplh', 'smpl']:
            full_pose = torch.cat([root_orient, pose_body, pose_hand], dim=1)
        elif self.model_type == 'smplx':
            full_pose = torch.cat([root_orient, pose_body, pose_jaw, pose_eye, pose_hand],
                                  dim=1)  # orient:3, body:63, jaw:3, eyel:3, eyer:3, handl, handr
        elif self.model_type in ['mano', 'mano']:
            full_pose = torch.cat([root_orient, pose_hand], dim=1)

        if self.use_dmpl:
            if dmpls is None: dmpls = self.dmpls
            shape_components = torch.cat([betas, dmpls], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.dmpldirs], dim=-1)
        elif self.model_type == 'smplx':
            if expression is None: expression = self.expression
            shape_components = torch.cat([betas, expression], dim=-1)
            shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)
        else:
            shape_components = betas
            shapedirs = self.shapedirs

        verts, joints, bone_transforms, abs_bone_transforms, v_posed = lbs(betas=shape_components, pose=full_pose,
                                                                           v_template=v_template,
                                                                           clothed_v_template=clothed_v_template,
                                                                           shapedirs=shapedirs, posedirs=self.posedirs,
                                                                           J_regressor=self.J_regressor,
                                                                           parents=self.kintree_table[0].long(),
                                                                           lbs_weights=self.weights,
                                                                           dtype=self.dtype)

        Jtr = joints + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)
        v_posed = v_posed + trans.unsqueeze(dim=1)

        res = {}
        res['v'] = verts
        res['v_a_pose'] = v_posed
        res['f'] = self.f
        res['abs_bone_transforms'] = abs_bone_transforms
        res['bone_transforms'] = bone_transforms
        res['betas'] = betas
        res['Jtr'] = Jtr

        if self.model_type == 'smpl':
            res['pose_body'] = pose_body
        elif self.model_type == 'smplh':
            res['pose_body'] = pose_body
            res['pose_hand'] = pose_hand
        elif self.model_type == 'smplx':
            res['pose_body'] = pose_body
            res['pose_hand'] = pose_hand
            res['pose_jaw'] = pose_jaw
            res['pose_eye'] = pose_eye
        elif self.model_type in ['mano', 'mano']:
            res['pose_hand'] = pose_hand
        res['full_pose'] = full_pose

        if not return_dict:
            class result_meta(object):
                pass

            res_class = result_meta()
            for k, v in res.items():
                res_class.__setattr__(k, v)
            res = res_class

        return res


def lbs(betas, pose, v_template, clothed_v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, num_joints=23, dtype=torch.float32):
    """ Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The codebase mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        num_joints : int, optional
            The number of joints of the model. The default value is equal
            to the number of joints of the SMPL body model
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    """

    batch_size = betas.shape[0]
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)
    if clothed_v_template is not None:
        v_shaped = clothed_v_template

    rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])

    if posedirs is not None:
        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])

        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped
    else:
        v_posed = v_shaped

    # 4. Get the global joint location
    J_transformed, A, abs_A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).repeat([batch_size, 1, 1])
    num_joints = J_regressor.shape[0]
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, A, abs_A, v_posed


def vertices2joints(J_regressor, vertices):
    """ Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    """

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    """ Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(aa_rots):
    """
    convert batch of rotations in axis-angle representation to matrix representation
    :param aa_rots: Nx3
    :return: mat_rots: Nx3x3
    """

    dtype = aa_rots.dtype
    device = aa_rots.device

    batch_size = aa_rots.shape[0]

    angle = torch.norm(aa_rots + 1e-8, dim=1, keepdim=True)
    rot_dir = aa_rots / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    """ Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    batch_size = rot_mats.shape[0]
    num_joints = joints.shape[1]
    device = rot_mats.device

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = torch.cat([joints, torch.zeros([batch_size, num_joints, 1, 1], dtype=dtype, device=device)], dim=2)
    init_bone = torch.matmul(transforms, joints_homogen)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    rel_transforms = transforms - init_bone

    return posed_joints, rel_transforms, transforms
