import os
import math
import random
import trimesh
import pyrender
import numpy as np
import PIL

VIEWS=10
IMAGE_SIZE=64
FOCAL=512

Z_CENTER = 32
Z_RANGE = 32
Z_NEAR = 8
Z_FAR = 64

PITCH_RANGE = math.pi/6
YAW_RANGE = math.pi/6
ROLL_RANGE = math.pi/6

def get_colored_cube():
    side_faces = [[4,6], [10,11], [3,8], [0,2], [7,9], [1,5]]
    side_colors = [trimesh.visual.color.random_color() for _ in range(6)]
    cube = trimesh.primitives.Box()

    cube.visual.face_colors = [trimesh.visual.color.random_color()]*12
    for sf, sc in zip(side_faces, side_colors):
        cube.visual.face_colors[sf[0]] = sc
        cube.visual.face_colors[sf[1]] = sc

    return cube

def get_intrinsics():
    return pyrender.IntrinsicsCamera(
        fx=FOCAL,
        fy=FOCAL,
        cx=IMAGE_SIZE/2,
        cy=IMAGE_SIZE/2,
        znear=Z_NEAR,
        zfar=Z_FAR)

def euler_to_rotation_matrix(pitch=0.0, yaw=0.0, roll=0.0):
    R_x = np.array([[1., 0., 0.],
                    [0.,math.cos(pitch), -math.sin(pitch)],
                    [0., math.sin(pitch), math.cos(pitch)]])

    R_y = np.array([[math.cos(yaw), 0., math.sin(yaw)],
                    [0., 1., 0.],
                    [-math.sin(yaw), 0., math.cos(yaw)]])

    R_z = np.array([[math.cos(roll), -math.sin(roll), 0.],
                    [math.sin(roll), math.cos(roll), 0.],
                    [0., 0., 1.]])

    return R_z.dot(R_y).dot(R_x)

def get_random_pose():
    pitch = random.uniform(-PITCH_RANGE, PITCH_RANGE)
    yaw = random.uniform(-YAW_RANGE, YAW_RANGE)
    roll = random.uniform(-ROLL_RANGE, ROLL_RANGE)
    z = random.uniform(Z_CENTER-Z_RANGE/2, Z_CENTER+Z_RANGE/2)

    Pr = np.eye(4)
    Pr[:3,:3] = euler_to_rotation_matrix(pitch, yaw, roll)

    Pt = np.eye(4)
    Pt[2,3] = -z

    return Pr.dot(Pt)

def to_gl(pose):
    return pose.dot(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))

# Define geometry and properties
mesh = pyrender.Mesh.from_trimesh(get_colored_cube(), smooth=False)

# Define light
lights = [pyrender.PointLight(intensity=500.0) for _ in range(VIEWS)]
light_poses = [to_gl(get_random_pose()) for _ in range(VIEWS)]

# Define camera
intrinsics = get_intrinsics()

images = []
masks = []
poses = []
for view in range(VIEWS):

    # Define scene
    scene = pyrender.Scene()
    scene.add(mesh)
    for l, lp in zip(lights, light_poses): scene.add(l, pose=lp)
    pose = get_random_pose()
    scene.add(intrinsics, pose=to_gl(pose))
    poses.append(pose)

    # Render
    renderer = pyrender.OffscreenRenderer(IMAGE_SIZE, IMAGE_SIZE)
    image, depth = renderer.render(scene)
    images.append(image)
    masks.append(depth != 0.)

# Export
data_dummy_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dummy', 'scan0')

# Export cameras
cameras = {}
k = np.eye(3)
k[:2,:2] *= FOCAL
k[:2,2] = IMAGE_SIZE/2
for idx, p in enumerate(poses):
    p = np.linalg.inv(p)

    wm = np.eye(4)
    wm[:3,:3] = k@p[:3,:3]
    wm[:3,3] = k@p[:3,3]

    cameras['scale_mat_%d' % idx] = np.eye(4)
    cameras['world_mat_%d' % idx] = wm
np.savez(os.path.join(data_dummy_dir,'cameras.npz'), **cameras)

# Export images and masks
data_dummy_image_dir = os.path.join(data_dummy_dir, 'image')
data_dummy_mask_dir = os.path.join(data_dummy_dir, 'mask')
if not os.path.exists(data_dummy_dir): os.mkdir(data_dummy_dir)
if not os.path.exists(data_dummy_image_dir): os.mkdir(data_dummy_image_dir)
if not os.path.exists(data_dummy_mask_dir): os.mkdir(data_dummy_mask_dir)
for idx, (i, m) in enumerate(zip(images, masks)):
    PIL.Image.fromarray(i.astype(np.uint8)).save(os.path.join(data_dummy_image_dir, 'img_{0:04}.png'.format(idx)))
    PIL.Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(data_dummy_mask_dir, 'mask_{0:04}.png'.format(idx)))
