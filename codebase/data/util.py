import cv2
import numpy as np


# def get_transform_new(center, scale, res, rot=0, flip=False):
#     """Generate transformation matrix."""
#     h = 200 * scale
#     # Top-left corner of the original bounding box
#     x1 = center[0] - h / 2
#     y1 = center[1] - h / 2
#     # Set rotation center as the new origin
#     t1 = np.eye(3)
#     t1[:2, -1] = -center
#     # Rotate around the new origin and translate to new image coordinates
#     t2 = np.eye(3)
#     rot_rad = rot * np.pi / 180
#     sn, cs = np.sin(rot_rad), np.cos(rot_rad)
#     h_rot = int(np.ceil(h * np.abs(sn) + h * np.abs(cs)))
#     t2[0, :2] = [cs, -sn]
#     t2[1, :2] = [sn, cs]
#     t2[2, 2] = 1
#     trans = (h_rot - h) / 2
#     t2[:2, -1] = [trans + center[0] - x1, trans + center[1] - y1]
#     # Flip the image horizontally
#     t3 = np.eye(3)
#     if flip:
#         t3[0, 0] = -1
#         t3[0, 2] = h_rot - 1
#
#     t = np.dot(t3, np.dot(t2, t1))  # transformation except for final scaling
#
#     # Scale the image to specified resolution
#     t4 = np.eye(3)
#     t4[0, 0] = res[1] / h_rot
#     t4[1, 1] = res[0] / h_rot
#
#     return np.dot(t4, t), t, h_rot
#
#
# def crop_new(img, center, scale, res, rot=0, flip=False):
#     """Crop image according to the supplied bounding box."""
#     t, t_no_scale, h_rot = get_transform_new(center, scale, res, rot, flip)
#     img_crop = cv2.warpAffine(img, t[:2, :], res)
#
#     return img_crop, t_no_scale, h_rot


def transform(pt, center, scale, res, invert=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape, dtype=np.uint8)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]

    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    new_img = cv2.resize(new_img, res)

    return new_img


def get_transform(center, scale, res):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    return t
