import os
from glob import glob
from os.path import join, basename
import numpy as np
import h5py

from macros import cam_params

def compute_smpl_means_params(subjects):
        dataset_folder = "/cluster/project/infk/hilliges/lectures/mp21/project3/data/H36M_data_st"


        betas_sum = np.zeros(10)
        pose_body_sum = np.zeros(63)
        pose_hand_sum = np.zeros(6)
        root_orient_sum = np.zeros(3)
        counter = 0

        for subject in subjects:
            subject_dir = join(dataset_folder, subject)
            action_dirs = glob(join(subject_dir, '*'))
            actions = set()
            for action_dir in action_dirs:
                actions.add(basename(action_dir).split('.')[0])

            actions = sorted(list(actions))

            for action in actions:
                for cam_idx, cam in enumerate(cam_params):
                    #img_dir = join(self.img_folder, subject, 'Images', action + '.' + cam['id'])
                    points_dir = join(subject_dir, action + '.' + cam['id'])

                    points_files = sorted(glob(join(points_dir, '*.npz')))
                    # for f_idx, points_file in enumerate(points_files):
                    for points_file in points_files:
                        with np.load(points_file) as points_file_dict:
                            betas_sum += points_file_dict['betas']
                            pose_body_sum += points_file_dict['pose_body']
                            pose_hand_sum += points_file_dict['pose_hand']
                            root_orient_sum += points_file_dict['root_orient']
                            counter += 1
                        
                        if counter % 1e3 == 0:
                            print(f'Processed {int(counter * 1e3)} files')
                        
        
        betas_mean = betas_sum / counter
        pose_body_mean = pose_body_sum / counter
        pose_hand_mean = pose_hand_sum / counter
        root_orient_mean = root_orient_sum / counter

        smpl_mean_params_dict = {
            'betas_mean' : betas_mean,
            'pose_body_mean' : pose_body_mean,
            'pose_hand_mean' : pose_hand_mean,
            'root_orient_mean' : root_orient_mean
        }

        print(betas_mean)
        print(pose_body_mean)
        print(pose_hand_mean)

        outfile = 'smpl_mean_params_' + '_'.join(subjects) + '.npz'
        np.savez(outfile, smpl_mean_params_dict)
        

if __name__ == '__main__':
    compute_smpl_means_params(['S1','S5','S6','S7','S8'])
    #mean_values = h5py.File("neutral_smpl_mean_params.h5")
    #print(list(mean_values['pose']))
    #print(list(mean_values['shape']))