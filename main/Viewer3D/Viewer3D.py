import trimesh
import vedo
import numpy as np
import io
from PIL import Image

def generateImagesTriMesh(filename):
    mesh = trimesh.load_mesh(filename)
    scene = mesh.scene()

    rotate_z = trimesh.transformations.rotation_matrix(
        angle=np.radians(72.0), direction=[0, 1, 1], point=scene.centroid
    )

    tilt_down = trimesh.transformations.rotation_matrix(
        angle=np.radians(-45.0), direction=[1, 0, 0], point=scene.centroid
    )

    mesh.apply_transform(tilt_down)
    imgs = []
    for i in range(5):
        camera_old, _geometry = scene.graph[scene.camera.name]
        camera_new = np.dot(rotate_z, camera_old)
        scene.graph[scene.camera.name] = camera_new
        try:
            
            png = scene.save_image(resolution=[512, 512], visible=True)
            image = Image.open(io.BytesIO(png))
            imgs.append(image)

        except BaseException as E:
            return []
            
    return imgs

if __name__ == "__main__":
    FILE = r'C:\Users\Almog\Dev\shap-e\project\models\color_blue_shoe_test_output.obj'
    imgs = generateImagesTriMesh(FILE)
    imgs[0].show()