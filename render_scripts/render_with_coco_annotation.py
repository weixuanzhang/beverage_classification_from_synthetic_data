import blenderproc as bproc
import bpy
import argparse

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

C = bpy.context
# C.scene.render.engine = 'BLENDER_EEVEE'
scn = C.scene

parser = argparse.ArgumentParser()
parser.add_argument('camera', nargs='?', default="examples/resources/camera_positions", help="Path to the camera file")
args = parser.parse_args()

bproc.init()

blend_path = '/home/weixuan/Documents/Code/blenderproc/assets/scene/base_scene_water_removed.blend'
objs = bproc.loader.load_blend(blend_path)

# Set some category ids for loaded objects
for j, obj in enumerate(objs):
    print(obj.get_name())
    if 'can' in obj.get_name() or 'Can' in obj.get_name():
        obj.set_cp("category_id", 1)
    elif 'bottle' in obj.get_name() or 'Bottle' in obj.get_name():
        obj.set_cp("category_id", 2)
        # obj.set_cp("name","bottle")

    else:
        obj.set_cp("category_id", 3)
        # obj.set_cp("name","N")

#load the environment
hdri_path = "/home/weixuan/Documents/Code/blenderproc/assets/world/hdris/photo_studio_01_4k.exr"

node_tree = scn.world.node_tree
tree_nodes = node_tree.nodes
tree_nodes.clear()
node_background = tree_nodes.new(type='ShaderNodeBackground')
node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
node_environment.image = bpy.data.images.load(hdri_path)
node_environment.location = -300, 0
# Add Output node
node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
node_output.location = 200, 0

# Link all nodes
links = node_tree.links
link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

bproc.camera.set_resolution(640, 480)
#load sampled camera position and orientations.
# position = [1.4588913917541504, 9.5, 5.39830923080444]
# euler_rotation = [1.6608428955078125, 0.01745329052209854, -3.2883405685424805]
# matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
# bproc.camera.add_camera_pose(matrix_world)

# read the camera positions file and convert into homogeneous camera-world transformation
with open(args.camera, "r") as f:
    for line in f.readlines():
        line = [float(x) for x in line.split()]
        position, euler_rotation = line[:3], line[3:6]
        matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
        bproc.camera.add_camera_pose(matrix_world)


#Create a point light
light = bproc.types.Light()
light.set_location([2.7062454223632812, 7.695454120635986, 16.55386161804199])
light.set_energy(500)

# activate normal rendering
bproc.renderer.enable_normals_output()
# bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])


# render the whole pipeline
data = bproc.renderer.render()
#TODO: need to set up the render engine

# # write the data to a .hdf5 container
# output_dir = "/home/weixuan/Documents/Code/blenderproc/data/rendering/output/"
# bproc.writer.write_hdf5(output_dir, data)


# Write data to coco file
coco_data_dir = "/home/weixuan/Documents/Code/blenderproc/data/annotation/"
bproc.writer.write_coco_annotations(coco_data_dir,
                                    instance_segmaps=data["instance_segmaps"],
                                    instance_attribute_maps=data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    color_file_format="JPEG")