import blenderproc as bproc
import bpy
import argparse
import numpy as np

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

C = bpy.context
# C.scene.render.engine = 'BLENDER_EEVEE'
scn = C.scene

parser = argparse.ArgumentParser()
# parser.add_argument('camera', nargs='?', default="examples/resources/camera_positions", help="Path to the camera file")
# args = parser.parse_args()

bproc.init()

blend_path = '/home/weixuan/Documents/Code/blenderproc/assets/scene/base_scene_water_occlusion.blend'
objs = bproc.loader.load_blend(blend_path)
beer_diameter = 0.75
# obj_path = '/home/weixuan/Documents/Code/blenderproc/assets/bottle/koicha/koicha.obj'
# obj_koicha = bproc.loader.load_obj(obj_path)

# Set some category ids for loaded objects
poi_objs = []
for j, obj in enumerate(objs):
    print(obj.get_name())
    # print(obj.get_bound_box())
    if 'can' in obj.get_name() or 'Can' in obj.get_name():
        obj.set_cp("category_id", 1)
        #only add the object without a child  into the poi_objs
        if obj.get_children() == []:
            poi_objs.append(obj)
    elif 'bottle' in obj.get_name() or 'Bottle' in obj.get_name():
        obj.set_cp("category_id", 2)
        # obj.set_cp("name","bottle")
        # only add the object without a child  into the poi_objs
        if obj.get_children() == []:
            poi_objs.append(obj)

    else:
        obj.set_cp("category_id", 3)
        # obj.set_cp("name","N")

# obj_koicha
# loc = objs[-2].get_location() - np.array([-0.2, -0.5, 0])
# obj_koicha[0].set_location(loc)
# obj_koicha[0].set_cp("category_id", 2)
#load the environment
hdri_path = "/home/weixuan/Documents/Code/blenderproc/assets/world/hdris/photo_studio_01_4k.exr"

# materials = bproc.material.collect_all()
# can_material = bproc.filter.one_by_attr(materials, "name", "Metal part.001")
beer_can = bproc.filter.one_by_attr(objs, "name", "beer_golden_can")
beer_location = beer_can.get_location()
for i in range(5):
    beer_duplicate = beer_can.duplicate()
    beer_new_loc = beer_location + np.array([0.0, -beer_diameter, 0])
    beer_duplicate.set_location(beer_new_loc)
    beer_location = beer_new_loc
# beer_can.set_material(0,can_material)

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
# with open(args.camera, "r") as f:
#     for line in f.readlines():
#         line = [float(x) for x in line.split()]
#         position, euler_rotation = line[:3], line[3:6]
#         matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
#         bproc.camera.add_camera_pose(matrix_world)

#camera sampling
# Find point of interest, all cam poses should look towards it
poi = bproc.object.compute_poi(poi_objs)
# Sample five camera poses
for i in range(5):
    # Sample random camera location above objects
    location = np.random.uniform([-4, 9.5, 5.4], [-1.0, 9.4, 9.4])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.1, 0.1))
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


#Create a point light
light = bproc.types.Light()
light.set_location([2.7062454223632812, 7.695454120635986, 16.55386161804199])
light.set_energy(500)

# activate normal rendering
bproc.renderer.enable_normals_output()
# bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance"])

# Enable motion blur
# bproc.renderer.enable_motion_blur(motion_blur_length=0.01)
bproc.renderer.set_noise_threshold(0.1)
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