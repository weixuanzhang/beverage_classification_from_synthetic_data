import blenderproc as bproc
import bpy
import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

C = bpy.context
C.scene.render.engine = 'BLENDER_EEVEE'
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

# bproc.loader.get_random_world_background_hdr_img_path_from_haven(hdri_path)