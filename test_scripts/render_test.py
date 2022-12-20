import blenderproc as bproc
import bpy

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

C = bpy.context
# C.scene.render.engine = 'BLENDER_EEVEE'
scn = C.scene

bproc.init()

blend_path = '/home/weixuan/Documents/Code/blenderproc/assets/scene/base_scene.blend'
objs = bproc.loader.load_blend(blend_path)

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
position = [1.4588913917541504, 7.224208831787109, 5.39830923080444]
euler_rotation = [1.6608428955078125, 0.01745329052209854, -3.2883405685424805]
matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
bproc.camera.add_camera_pose(matrix_world)

#Create a point light

light = bproc.types.Light()
light.set_location([2.7062454223632812, 7.695454120635986, 16.55386161804199])
light.set_energy(500)

# render the whole pipeline
data = bproc.renderer.render()
#TODO: need to set up the render engine

# write the data to a .hdf5 container
output_dir = "/home/weixuan/Documents/Code/blenderproc/data/rendering/output/"
bproc.writer.write_hdf5(output_dir, data)
# bproc.loader.get_random_world_background_hdr_img_path_from_haven(hdri_path)