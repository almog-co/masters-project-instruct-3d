import trimesh
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image
from pyglet import gl

def showSnapshot(filename, degrees=90):
    mesh = trimesh.load_mesh(filename)
    scene = trimesh.Scene([mesh])
    rotation_matrix = trimesh.transformations.rotation_matrix(
        np.radians(degrees),
        [0, 1, 0]
    )
    window_conf = gl.Config(double_buffer=True, depth_size=24)
    scene.camera_transform = np.dot(rotation_matrix, scene.camera_transform)
    bytes = scene.save_image(visible=True, window_conf=window_conf)
    img = Image.open(io.BytesIO(bytes))
    img_array = np.array(img)
    plt.imshow(img_array)
    plt.axis('off')
    plt.show()
