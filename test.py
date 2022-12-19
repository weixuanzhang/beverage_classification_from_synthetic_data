import blenderproc as bproc
import bpy
import argparse
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)


parser = argparse.ArgumentParser()
parser.add_argument('camera', help="Path to the camera file, should be examples/resources/camera_positions")
parser.add_argument('output_dir', help="Path to where the final files, will be saved, could be examples/basics/basic/output")

args = parser.parse_args()

bproc.init()

C = bpy.context
C.scene.render.engine = 'CYCLES'
scn = C.scene



# obj0_path = '/home/weixuan/Documents/Code/blenderproc/assets/bottle/evian/waterbottle_6.obj'
obj0_path = '/home/weixuan/Documents/Code/blenderproc/assets/bottle/plastic_water_bottle/plastic_water_bottle.obj'
objs = bproc.loader.load_obj(obj0_path)
objs[0].set_location([-10, -50, -3])
objs[0].set_scale([10,10,10])
mat = objs[0].get_materials()
print(mat)
#load hdri images
node_tree = scn.world.node_tree
tree_nodes = node_tree.nodes
tree_nodes.clear()
node_background = tree_nodes.new(type='ShaderNodeBackground')
node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
hdri_path = "/home/weixuan/Documents/Code/3d-dl/assets/world/photo_studio_01_4k.exr"
node_environment.image = bpy.data.images.load(hdri_path)
node_environment.location = -300, 0

# Add Output node
node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
node_output.location = 200,0

# Link all nodes
links = node_tree.links
link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])


# bpy.data.objects["Camera"].location = (-6, -30, 2)
# bpy.data.objects["Camera"].rotation_mode = 'XYZ'
# bpy.data.objects["Camera"].rotation_euler = (1.4306915998458862, -0.06857321411371231, -3.331981420516967)

# define the camera resolution
bproc.camera.set_resolution(640, 480)

# read the camera positions file and convert into homogeneous camera-world transformation
with open(args.camera, "r") as f:
    for line in f.readlines():
        line = [float(x) for x in line.split()]
        position, euler_rotation = line[:3], line[3:6]
        matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
        bproc.camera.add_camera_pose(matrix_world)


# render the whole pipeline
data = bproc.renderer.render()

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)
