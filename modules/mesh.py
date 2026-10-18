import torch
from types import SimpleNamespace
try:
    import kaolin as kal
except ImportError:
    # No kaolin on the Mac (2026-10-18): OBJ loading falls back to _import_obj_plain below;
    # rendering still needs kaolin + CUDA and runs on a GPU machine only.
    kal = None
import modules.utils as utils
class _LazyOpen3d:
    """open3d is imported on first use (2026-10-07): importing it costs ~10 s per process from the
    network volume, and almost no script that imports this module ever calls it."""
    def __getattr__(self, name):
        import open3d
        return getattr(open3d, name)
o3d = _LazyOpen3d()
import copy
import numpy as np
import PIL

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def _import_obj_plain(path, with_normals=False):
    """kaolin-free OBJ reader: same vertices / faces as kal.io.obj.import_mesh(path) without materials
    (all 'v' lines in order; first index of each 'f' corner, 0-based; uniform face size required)."""
    if with_normals:
        raise NotImplementedError("with_normals=True needs kaolin")
    vertices, faces = [], []
    with open(path) as f:
        for line in f:
            data = line.split()
            if not data:
                continue
            if data[0] == 'v':
                vertices.append(data[1:])
            elif data[0] == 'f':
                faces.append([int(d.split('/')[0]) for d in data[1:]])
    if len({len(f) for f in faces}) > 1:
        raise ValueError(f"Mesh is non-homogeneous and cannot be imported from {path}.")
    vertices = torch.FloatTensor([float(el) for v in vertices for el in v]).view(-1, 3)
    return SimpleNamespace(vertices=vertices, faces=torch.LongTensor(faces) - 1,
                           vertex_normals=None, face_normals=None)


class Mesh():
    def __init__(self,obj_path, color=torch.tensor([0.0,0.0,1.0]), with_normals=False):
        if ".obj" in obj_path:
            # obj loader
            if kal is not None:
                mesh = kal.io.obj.import_mesh(obj_path, with_normals=with_normals)
            else:
                mesh = _import_obj_plain(obj_path, with_normals=with_normals)
        elif ".off" in obj_path:
            mesh = kal.io.off.import_mesh(obj_path)
        else:
            raise ValueError(f"{obj_path} extension not implemented in mesh reader.")
        
        self.vertices = mesh.vertices.to(device)
        self.faces = mesh.faces.to(device)
        self.vertex_normals = None
        self.face_normals = None
        self.texture_map = None
        self.face_uvs = None
        
        self.first_ring=None
        if ".obj" in obj_path:
            if mesh.vertex_normals is not None:
                self.vertex_normals = mesh.vertex_normals.to(device).float()

                # Normalize
                self.vertex_normals = torch.nn.functional.normalize(self.vertex_normals)

            if mesh.face_normals is not None:
                self.face_normals = mesh.face_normals.to(device).float()

                # Normalize
                self.face_normals = torch.nn.functional.normalize(self.face_normals)

        self.set_mesh_color(color)

    def standardize_mesh(self,inplace=False):
        mesh = self if inplace else copy.deepcopy(self)
        return utils.standardize_mesh(mesh)

    def normalize_mesh(self,inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        return utils.normalize_mesh(mesh)

    def update_vertex(self,verts,inplace=False):

        mesh = self if inplace else copy.deepcopy(self)
        mesh.vertices = verts
        return mesh

    def set_mesh_color(self,color):
        self.texture_map = utils.get_texture_map_from_color(self,color)
        self.face_attributes = utils.get_face_attributes_from_color(self,color)

    def set_image_texture(self,texture_map,inplace=True):

        mesh = self if inplace else copy.deepcopy(self)

        if isinstance(texture_map,str):
            texture_map = PIL.Image.open(texture_map)
            texture_map = np.array(texture_map,dtype=np.float) / 255.0
            texture_map = torch.tensor(texture_map,dtype=torch.float).to(device).permute(2,0,1).unsqueeze(0)
        
        mesh.texture_map = texture_map
        return mesh

    def divide(self,inplace=True):
        mesh = self if inplace else copy.deepcopy(self)
        new_vertices, new_faces, new_vertex_normals, new_face_uvs, new_face_normals = utils.add_vertices(mesh)
        mesh.vertices = new_vertices
        mesh.faces = new_faces
        mesh.vertex_normals = new_vertex_normals
        mesh.face_uvs = new_face_uvs
        mesh.face_normals = new_face_normals
        return mesh

    def divide_with_labels(self, face_label_map, inplace=True):
        mesh = self if inplace else copy.deepcopy(self)
        new_vertices, new_faces, new_vertex_normals, new_face_label_map = utils.add_vertices_with_labels(mesh, face_label_map)
        mesh.vertices = new_vertices
        mesh.faces = new_faces
        mesh.vertex_normals = new_vertex_normals
        return mesh, new_face_label_map

    def export(self, file, extension="obj", color=None):
        with open(file, "w+") as f:
            if extension == "obj":
                for vi, v in enumerate(self.vertices):
                    if color is None:
                        f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
                    else:
                        f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
                    if self.vertex_normals is not None:
                        f.write("vn %f %f %f\n" % (self.vertex_normals[vi, 0], self.vertex_normals[vi, 1], self.vertex_normals[vi, 2]))
                for face in self.faces:
                    f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))
            elif extension == "ply":
                f.write("ply\nformat ascii 1.0\n")
                f.write("element vertex {}\n".format(self.vertices.shape[0]))
                f.write("property float32 x\nproperty float32 y\nproperty float32 z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write("element face {}\n".format(self.faces.shape[0]))
                f.write("property list uint8 int32 vertex_index\n")
                f.write("end_header\n")
                for vi, v in enumerate(self.vertices):
                    f.write("%f %f %f %d %d %d\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
                for face in self.faces:
                    f.write("3 %d %d %d\n" % (face[0], face[1], face[2]))
            else:
                print("error -- mesh.py: incorrect extension provdied in mesh.export()")

    def export_to_open3d(self):
        mesh = o3d.geometry.TriangleMesh()

        mesh.vertices = o3d.utility.Vector3dVector(self.vertices.detach().cpu().numpy().astype(np.float32))
        mesh.triangles = o3d.utility.Vector3iVector(self.faces.detach().cpu().numpy().astype(np.int32))

        return mesh
