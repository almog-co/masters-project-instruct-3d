import trimesh
import vedo
import numpy as np
import io
from PIL import Image

def generateImage(file: str, theta: int):
    mesh = vedo.Mesh(file)
    vp = vedo.Plotter(offscreen=False)
    vp += mesh

    cam = dict(
        position=(-170.906, 79.1394, 15.9800),
        focal_point=(-2.02597, 15.9931, 0.509021),
        viewup=(0.357535, 0.925400, 0.125714),
        distance=180.962,
        clipping_range=(120.677, 257.285),
    )

    vp.show(camera=cam, viewup='z')

def create_pan_cameras(size):
    origins = []
    xs = []
    ys = []
    zs = []
    for theta in np.linspace(0, 2 * np.pi, num=20):
        z = np.array([np.sin(theta), np.cos(theta), -0.5])
        z /= np.sqrt(np.sum(z**2))
        origin = -z * 4
        x = np.array([np.cos(theta), -np.sin(theta), 0.0])
        y = np.cross(z, x)
        origins.append(origin)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return origins, xs, ys, zs

# Read .obj file
def read_obj(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                # Extract XYZ coordinates and RGB values
                x, y, z, r, g, b = map(float, parts[1:])
                vertices.append(((x, y, z), (r, g, b)))
    return vertices

def normalize_coordinates(vertices, image_size):
    # Separate coordinates and colors
    coords, colors = zip(*vertices)

    # Find min and max values for x and y
    xs, ys, zs = zip(*coords)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Normalize and scale coordinates
    normalized_vertices = []
    for (x, y, z), color in vertices:
        nx = (x - min_x) / (max_x - min_x) * (image_size[0] - 1)
        ny = (y - min_y) / (max_y - min_y) * (image_size[1] - 1)
        normalized_vertices.append(((nx, image_size[1] - 1 - ny, z), color))
    
    return normalized_vertices

# Simple planar mapping for texture
def create_texture(vertices, image_size):
    img = Image.new('RGB', image_size, 'black')
    pixels = img.load()

    for (x, y, z), (r, g, b) in vertices:
        # Set pixel color (scaling RGB values to 0-255 range if necessary)
        pixels[int(x), int(y)] = (int(r * 255), int(g * 255), int(b * 255))

    return img


if __name__ == "__main__":
    FILE = r'C:\Users\Almog\Dev\shap-e\project\models\color_blue_shoe_test_output.obj'
    FILE_TEXTURE = r'C:\Users\Almog\Dev\shap-e\project\models\Test.png'
    
    origins, xs, ys, zs = create_pan_cameras(128)
    # mesh = vedo.Mesh(FILE)
    mesh = trimesh.load_mesh(FILE)
    scene = mesh.scene()
    rotate_z = trimesh.transformations.rotation_matrix(
        angle=np.radians(72.0), direction=[0, 1, 1], point=scene.centroid
    )

    tilt_down = trimesh.transformations.rotation_matrix(
        angle=np.radians(-45.0), direction=[1, 0, 0], point=scene.centroid
    )

    # Combine the rotations
    combined_rotation = np.dot(rotate_z, tilt_down)

    # Apply the combined rotation to the mesh
    mesh.apply_transform(tilt_down)

    for i in range(5):
        trimesh.constants.log.info(f"Saving image {i}")

        # rotate the camera view transform
        camera_old, _geometry = scene.graph[scene.camera.name]
        camera_new = np.dot(rotate_z, camera_old)

        # apply the new transform
        scene.graph[scene.camera.name] = camera_new

        # saving an image requires an opengl context, so if -nw
        # is passed don't save the image
        try:
            # increment the file name
            file_name = "render_" + str(i) + ".png"
            # save a render of the object as a png
            png = scene.save_image(resolution=[512, 512], visible=True)
            with open(file_name, "wb") as f:
                f.write(png)
                f.close()

        except BaseException as E:
            print("unable to save image", str(E))

    # mesh.texture(FILE_TEXTURE)

    # for i in range(0, 20):
    #     origin = origins[i]
    #     x = xs[i]
    #     y = ys[i]
    #     z = zs[i]

    #     focal_point = origin + z * 10 

    #     cam = dict(
    #         position = origin.tolist(),
    #         focal_point = focal_point.tolist(), 
    #         viewup = y.tolist(),
    #     )
        
    #     plt = vedo.show(mesh, camera=cam, axes=0, interactive=False)
    #     arr = plt.screenshot(asarray=True)
    #     plt.close()
    #     image = Image.fromarray(arr.astype('uint8'), 'RGB').rotate(180)
    #     image.save(f"test/{i}.png")

    # data = open(FILE).read().split('\n')
    # rgb_data = []
    # for line in data:
    #     if line[0] != 'v':
    #         continue
    #     val = line.split(' ')
    #     rgb_data.append([float(c) for c in val[-3:]])
    # length = len(rgb_data)
    # X = int(np.ceil(length**0.5))  # Size of one dimension
    # total_size = X ** 2
    # padding = [(0, 0, 0)] * (total_size - length)  # Replace with your preferred color
    # rgb_data.extend(padding)

    # array_data = np.array(rgb_data).reshape(X, X, 3)
    # image_data = (array_data * 255).astype(np.uint8)
    # image = Image.fromarray(image_data, 'RGB')
    # image.save(FILE_TEXTURE)
    
    # vertices = read_obj(FILE)
    # normalized_vertices = normalize_coordinates(vertices, (1024, 1024))
    # texture_img = create_texture(normalized_vertices, (1024, 1024))
    # texture_img.show()



