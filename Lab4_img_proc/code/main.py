import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


EPS = 1e-6
INF = 1e18


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value):
        if isinstance(value, Vec3):
            return Vec3(self.x * value.x, self.y * value.y, self.z * value.z)
        return Vec3(self.x * value, self.y * value, self.z * value)

    def __rmul__(self, value):
        return self.__mul__(value)

    def __truediv__(self, value: float):
        return Vec3(self.x / value, self.y / value, self.z / value)

    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self):
        l = self.length()
        if l < EPS:
            return Vec3(0.0, 0.0, 0.0)
        return self / l

    def max_component(self) -> float:
        return max(self.x, self.y, self.z)

    def clamp(self, lo=0.0, hi=1.0):
        return Vec3(
            max(lo, min(hi, self.x)),
            max(lo, min(hi, self.y)),
            max(lo, min(hi, self.z)),
        )

    def tuple(self):
        return self.x, self.y, self.z


def reflect(v: Vec3, n: Vec3) -> Vec3:
    return v - n * (2.0 * v.dot(n))


def build_orthonormal_basis(n: Vec3) -> Tuple[Vec3, Vec3]:
    if abs(n.x) > 0.1:
        tangent = Vec3(0.0, 1.0, 0.0).cross(n).normalized()
    else:
        tangent = Vec3(1.0, 0.0, 0.0).cross(n).normalized()
    bitangent = n.cross(tangent).normalized()
    return tangent, bitangent


def cosine_sample_hemisphere(normal: Vec3) -> Tuple[Vec3, float]:
    r1 = random.random()
    r2 = random.random()
    phi = 2.0 * math.pi * r1
    r = math.sqrt(r2)

    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - r2))

    t, b = build_orthonormal_basis(normal)
    direction = (t * x + b * y + normal * z).normalized()
    pdf = max(EPS, direction.dot(normal)) / math.pi
    return direction, pdf


@dataclass
class Ray:
    origin: Vec3
    direction: Vec3


@dataclass
class Material:
    diffuse: Vec3
    specular: Vec3
    emission: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    name: str = "material"

    def validate_physical(self):
        for d, s, channel in zip(
            self.diffuse.tuple(),
            self.specular.tuple(),
            ("R", "G", "B")
        ):
            if d < 0 or s < 0:
                raise ValueError(f"Материал {self.name}: отрицательные коэффициенты в канале {channel}.")
            if d + s > 1.0 + 1e-9:
                raise ValueError(
                    f"Материал {self.name}: нарушена физичность в канале {channel}: "
                    f"diffuse({d:.3f}) + specular({s:.3f}) > 1."
                )

        for e, channel in zip(self.emission.tuple(), ("R", "G", "B")):
            if e < 0:
                raise ValueError(f"Материал {self.name}: отрицательное излучение в канале {channel}.")

    def is_light(self) -> bool:
        return self.emission.max_component() > 0.0

    def choose_event(self) -> Tuple[str, Vec3, float]:
        pd = self.diffuse.max_component()
        ps = self.specular.max_component()
        s = pd + ps

        if s < EPS:
            return "absorb", Vec3(0.0, 0.0, 0.0), 1.0

        pd /= s
        ps /= s

        r = random.random()
        if r < pd:
            return "diffuse", self.diffuse, pd
        return "specular", self.specular, ps


@dataclass
class Hit:
    t: float
    position: Vec3
    normal: Vec3
    material: Material


@dataclass
class Triangle:
    a: Vec3
    b: Vec3
    c: Vec3
    material: Material

    def normal(self) -> Vec3:
        return (self.b - self.a).cross(self.c - self.a).normalized()

    def area(self) -> float:
        return 0.5 * (self.b - self.a).cross(self.c - self.a).length()

    def intersect(self, ray: Ray) -> Optional[Hit]:
        edge1 = self.b - self.a
        edge2 = self.c - self.a
        h = ray.direction.cross(edge2)
        det = edge1.dot(h)

        if abs(det) < EPS:
            return None

        inv_det = 1.0 / det
        s = ray.origin - self.a
        u = inv_det * s.dot(h)
        if u < 0.0 or u > 1.0:
            return None

        q = s.cross(edge1)
        v = inv_det * ray.direction.dot(q)
        if v < 0.0 or u + v > 1.0:
            return None

        t = inv_det * edge2.dot(q)
        if t < EPS:
            return None

        pos = ray.origin + ray.direction * t
        n = self.normal()
        if n.dot(ray.direction) > 0.0:
            n = n * -1.0

        return Hit(t=t, position=pos, normal=n, material=self.material)

    def sample_point(self) -> Tuple[Vec3, Vec3, float]:
        r1 = math.sqrt(random.random())
        r2 = random.random()
        u = 1.0 - r1
        v = r1 * (1.0 - r2)
        w = r1 * r2
        p = self.a * u + self.b * v + self.c * w
        n = self.normal()
        pdf_area = 1.0 / max(EPS, self.area())
        return p, n, pdf_area


@dataclass
class Camera:
    position: Vec3
    target: Vec3
    up: Vec3
    fov_deg: float

    def generate_ray(self, x: float, y: float, width: int, height: int) -> Ray:
        forward = (self.target - self.position).normalized()
        right = forward.cross(self.up).normalized()
        true_up = right.cross(forward).normalized()

        aspect = width / height
        scale = math.tan(math.radians(self.fov_deg) * 0.5)

        px = (2.0 * ((x + 0.5) / width) - 1.0) * aspect * scale
        py = (1.0 - 2.0 * ((y + 0.5) / height)) * scale

        direction = (forward + right * px + true_up * py).normalized()
        return Ray(self.position, direction)


class Scene:
    def __init__(self):
        self.triangles: List[Triangle] = []
        self.lights: List[Triangle] = []

    def add_triangle(self, tri: Triangle):
        tri.material.validate_physical()
        self.triangles.append(tri)
        if tri.material.is_light():
            self.lights.append(tri)

    def intersect(self, ray: Ray) -> Optional[Hit]:
        closest_t = INF
        closest_hit = None
        for tri in self.triangles:
            hit = tri.intersect(ray)
            if hit and hit.t < closest_t:
                closest_t = hit.t
                closest_hit = hit
        return closest_hit

    def occluded(self, ray: Ray, max_dist: float) -> bool:
        for tri in self.triangles:
            hit = tri.intersect(ray)
            if hit and hit.t < max_dist - EPS:
                return True
        return False

    def choose_light(self) -> Optional[Tuple[Triangle, float]]:
        if not self.lights:
            return None

        powers = []
        total = 0.0
        for tri in self.lights:
            power = tri.area() * tri.material.emission.max_component()
            powers.append(power)
            total += power

        if total < EPS:
            idx = random.randrange(len(self.lights))
            return self.lights[idx], 1.0 / len(self.lights)

        r = random.random() * total
        acc = 0.0
        for tri, p in zip(self.lights, powers):
            acc += p
            if r <= acc:
                return tri, p / total

        return self.lights[-1], powers[-1] / total


def load_obj_as_triangles(
    filepath: str,
    material: Material,
    scale: float = 1.0,
    offset: Vec3 = Vec3(0.0, 0.0, 0.0),
) -> List[Triangle]:
    vertices: List[Vec3] = []
    triangles: List[Triangle] = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                vertices.append(Vec3(x * scale, y * scale, z * scale) + offset)

            elif line.startswith("f "):
                parts = line.split()[1:]
                indices = []
                for part in parts:
                    idx = part.split("/")[0]
                    indices.append(int(idx) - 1)

                if len(indices) < 3:
                    continue

                for i in range(1, len(indices) - 1):
                    a = vertices[indices[0]]
                    b = vertices[indices[i]]
                    c = vertices[indices[i + 1]]
                    triangles.append(Triangle(a, b, c, material))

    return triangles


class PathTracer:
    def __init__(self, scene: Scene, camera: Camera):
        self.scene = scene
        self.camera = camera
        self.max_depth = 5
        self.background = Vec3(0.0, 0.0, 0.0)
        self.stop_flag = False

    def estimate_direct_light(self, hit: Hit) -> Vec3:
        picked = self.scene.choose_light()
        if picked is None:
            return Vec3(0.0, 0.0, 0.0)

        light_tri, p_light_select = picked
        lp, ln, pdf_area = light_tri.sample_point()

        to_light = lp - hit.position
        dist2 = to_light.dot(to_light)
        dist = math.sqrt(dist2)
        wi = to_light / max(EPS, dist)

        cos_surface = max(0.0, hit.normal.dot(wi))
        cos_light = max(0.0, ln.dot(wi * -1.0))

        if cos_surface <= 0.0 or cos_light <= 0.0:
            return Vec3(0.0, 0.0, 0.0)

        shadow_ray = Ray(hit.position + hit.normal * 1e-4, wi)
        if self.scene.occluded(shadow_ray, dist):
            return Vec3(0.0, 0.0, 0.0)

        Le = light_tri.material.emission
        brdf = hit.material.diffuse / math.pi
        pdf = max(EPS, p_light_select * pdf_area)

        return brdf * Le * (cos_surface * cos_light / max(EPS, dist2)) / pdf

    def trace(self, ray: Ray, depth: int = 0) -> Vec3:
        if depth >= self.max_depth:
            return Vec3(0.0, 0.0, 0.0)

        hit = self.scene.intersect(ray)
        if hit is None:
            return self.background

        emitted = hit.material.emission
        event, color, p_event = hit.material.choose_event()

        if event == "absorb":
            return emitted

        result = emitted

        if event == "diffuse":
            direct = self.estimate_direct_light(hit)

            new_dir, pdf_dir = cosine_sample_hemisphere(hit.normal)
            brdf = hit.material.diffuse / math.pi
            cos_theta = max(0.0, hit.normal.dot(new_dir))

            new_ray = Ray(hit.position + hit.normal * 1e-4, new_dir)
            incoming = self.trace(new_ray, depth + 1)

            indirect = brdf * incoming * (cos_theta / max(EPS, pdf_dir))
            result += (direct + indirect) / max(EPS, p_event)

        elif event == "specular":
            new_dir = reflect(ray.direction, hit.normal).normalized()
            new_ray = Ray(hit.position + hit.normal * 1e-4, new_dir)
            incoming = self.trace(new_ray, depth + 1)
            result += (color * incoming) / max(EPS, p_event)

        return result

    def render_progressive(
        self,
        width: int,
        height: int,
        target_spp: int,
        time_limit_sec: float,
        progress_callback=None,
        preview_callback=None
    ) -> List[List[Vec3]]:
        accum = [[Vec3(0.0, 0.0, 0.0) for _ in range(width)] for _ in range(height)]
        passes_done = 0
        start_time = time.time()

        while not self.stop_flag:
            if 0 < target_spp <= passes_done:
                break
            if 0 < time_limit_sec <= (time.time() - start_time):
                break

            for y in range(height):
                if self.stop_flag:
                    break

                for x in range(width):
                    jx = random.random() - 0.5
                    jy = random.random() - 0.5
                    ray = self.camera.generate_ray(x + jx, y + jy, width, height)
                    sample = self.trace(ray, 0)
                    accum[y][x] = accum[y][x] + sample

                if progress_callback:
                    progress_callback(y + 1, height, passes_done + 1)

            passes_done += 1

            if preview_callback:
                current_avg = [[accum[y][x] / passes_done for x in range(width)] for y in range(height)]
                preview_callback(current_avg, passes_done, time.time() - start_time)

        if passes_done == 0:
            return accum

        return [[accum[y][x] / passes_done for x in range(width)] for y in range(height)]


def tonemap_and_gamma(framebuffer: List[List[Vec3]], gamma: float = 2.2) -> List[List[Tuple[int, int, int]]]:
    max_lum = 0.0
    for row in framebuffer:
        for c in row:
            max_lum = max(max_lum, c.max_component())

    scale = 1.0 / max(EPS, max_lum)

    def to_byte(value: float) -> int:
        return int(round(max(0.0, min(1.0, value)) * 255.0))

    img = []
    inv_gamma = 1.0 / gamma
    for row in framebuffer:
        out_row = []
        for c in row:
            mapped = (c * scale).clamp(0.0, 1.0)
            mapped = Vec3(
                mapped.x ** inv_gamma,
                mapped.y ** inv_gamma,
                mapped.z ** inv_gamma,
            )
            out_row.append((
                to_byte(mapped.x),
                to_byte(mapped.y),
                to_byte(mapped.z),
            ))
        img.append(out_row)

    return img


def save_ppm(filename: str, image_rgb: List[List[Tuple[int, int, int]]]):
    height = len(image_rgb)
    width = len(image_rgb[0]) if height > 0 else 0
    with open(filename, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        for row in image_rgb:
            for r, g, b in row:
                f.write(bytes((r, g, b)))


def make_default_scene(
    include_obj_path: Optional[str],
    light_intensity: float,
    light_color: Vec3,
    object_material_mode: str
) -> Tuple[Scene, Camera]:
    scene = Scene()

    white = Material(diffuse=Vec3(0.75, 0.75, 0.75), specular=Vec3(0.0, 0.0, 0.0), name="white")
    red = Material(diffuse=Vec3(0.75, 0.2, 0.2), specular=Vec3(0.0, 0.0, 0.0), name="red")
    green = Material(diffuse=Vec3(0.2, 0.75, 0.2), specular=Vec3(0.0, 0.0, 0.0), name="green")

    if object_material_mode == "mirror":
        object_material = Material(
            diffuse=Vec3(0.05, 0.05, 0.05),
            specular=Vec3(0.90, 0.90, 0.90),
            name="object_mirror"
        )
    elif object_material_mode == "diffuse_blue":
        object_material = Material(
            diffuse=Vec3(0.20, 0.35, 0.85),
            specular=Vec3(0.05, 0.05, 0.05),
            name="object_diffuse_blue"
        )
    elif object_material_mode == "mixed":
        object_material = Material(
            diffuse=Vec3(0.45, 0.45, 0.45),
            specular=Vec3(0.35, 0.35, 0.35),
            name="object_mixed"
        )
    else:
        object_material = Material(
            diffuse=Vec3(0.65, 0.65, 0.65),
            specular=Vec3(0.0, 0.0, 0.0),
            name="object_diffuse_gray"
        )

    light_emission = Vec3(
        light_color.x * light_intensity,
        light_color.y * light_intensity,
        light_color.z * light_intensity,
    )
    light = Material(
        diffuse=Vec3(0.0, 0.0, 0.0),
        specular=Vec3(0.0, 0.0, 0.0),
        emission=light_emission,
        name="area_light"
    )

    scene.add_triangle(Triangle(Vec3(-2, -1, -2), Vec3(2, -1, -2), Vec3(2, -1, 2), white))
    scene.add_triangle(Triangle(Vec3(-2, -1, -2), Vec3(2, -1, 2), Vec3(-2, -1, 2), white))

    scene.add_triangle(Triangle(Vec3(-2, 3, -2), Vec3(2, 3, 2), Vec3(2, 3, -2), white))
    scene.add_triangle(Triangle(Vec3(-2, 3, -2), Vec3(-2, 3, 2), Vec3(2, 3, 2), white))

    scene.add_triangle(Triangle(Vec3(-2, -1, 2), Vec3(2, -1, 2), Vec3(2, 3, 2), white))
    scene.add_triangle(Triangle(Vec3(-2, -1, 2), Vec3(2, 3, 2), Vec3(-2, 3, 2), white))

    scene.add_triangle(Triangle(Vec3(-2, -1, -2), Vec3(-2, -1, 2), Vec3(-2, 3, 2), red))
    scene.add_triangle(Triangle(Vec3(-2, -1, -2), Vec3(-2, 3, 2), Vec3(-2, 3, -2), red))

    scene.add_triangle(Triangle(Vec3(2, -1, -2), Vec3(2, 3, 2), Vec3(2, -1, 2), green))
    scene.add_triangle(Triangle(Vec3(2, -1, -2), Vec3(2, 3, -2), Vec3(2, 3, 2), green))

    p0 = Vec3(-0.7, -1.0, 0.5)
    p1 = Vec3(0.0, -1.0, -0.2)
    p2 = Vec3(0.5, -1.0, 0.8)
    p3 = Vec3(0.0, 0.3, 0.3)
    scene.add_triangle(Triangle(p0, p1, p3, object_material))
    scene.add_triangle(Triangle(p1, p2, p3, object_material))
    scene.add_triangle(Triangle(p2, p0, p3, object_material))
    scene.add_triangle(Triangle(p0, p2, p1, object_material))

    scene.add_triangle(Triangle(Vec3(-0.6, 2.95, -0.6), Vec3(0.6, 2.95, -0.6), Vec3(0.6, 2.95, 0.6), light))
    scene.add_triangle(Triangle(Vec3(-0.6, 2.95, -0.6), Vec3(0.6, 2.95, 0.6), Vec3(-0.6, 2.95, 0.6), light))

    if include_obj_path:
        obj_material = object_material
        tris = load_obj_as_triangles(
            include_obj_path,
            material=obj_material,
            scale=0.6,
            offset=Vec3(0.0, -1.0, 0.4),
        )
        for tri in tris:
            scene.add_triangle(tri)

    camera = Camera(
        position=Vec3(0.0, 1.0, -4.5),
        target=Vec3(0.0, 0.8, 0.5),
        up=Vec3(0.0, 1.0, 0.0),
        fov_deg=45.0
    )
    return scene, camera


def _add_labeled_entry(parent, label, variable):
    ttk.Label(parent, text=label).pack(anchor="w", pady=(4, 0))
    ttk.Entry(parent, textvariable=variable, width=30).pack(fill="x", pady=2)


def _parse_vec3_rgb(r_var, g_var, b_var, name="RGB"):
    r = float(r_var.get())
    g = float(g_var.get())
    b = float(b_var.get())

    if r < 0 or g < 0 or b < 0:
        raise ValueError(f"{name}: компоненты не должны быть отрицательными.")

    return Vec3(r, g, b)


class PathTracerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ЛР4 — Формирование изображения методом трассировки путей")
        self.root.geometry("1280x860")

        self.obj_path = tk.StringVar()
        self.width_var = tk.StringVar(value="500")
        self.height_var = tk.StringVar(value="500")
        self.spp_var = tk.StringVar(value="16")
        self.depth_var = tk.StringVar(value="5")
        self.gamma_var = tk.StringVar(value="2.2")
        self.seed_var = tk.StringVar(value="42")
        self.output_var = tk.StringVar(value="render.ppm")

        self.light_intensity_var = tk.StringVar(value="12.0")
        self.light_r_var = tk.StringVar(value="1.0")
        self.light_g_var = tk.StringVar(value="0.92")
        self.light_b_var = tk.StringVar(value="0.85")

        self.cam_x_var = tk.StringVar(value="0.0")
        self.cam_y_var = tk.StringVar(value="1.0")
        self.cam_z_var = tk.StringVar(value="-4.5")

        self.target_x_var = tk.StringVar(value="0.0")
        self.target_y_var = tk.StringVar(value="0.8")
        self.target_z_var = tk.StringVar(value="0.5")

        self.material_mode_var = tk.StringVar(value="mirror")

        self.render_mode_var = tk.StringVar(value="spp")
        self.time_limit_var = tk.StringVar(value="10")

        self.status_var = tk.StringVar(value="Готово.")
        self.progress_var = tk.DoubleVar(value=0.0)

        self.rendered_ppm_temp = "_preview.ppm"
        self.current_image = None
        self.renderer = None

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        left_container = ttk.Frame(main)
        left_container.pack(side="left", fill="y", padx=(0, 10))

        left_canvas = tk.Canvas(left_container, width=420, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_scrollbar.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="y", expand=False)

        left = ttk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        def on_left_configure(_event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def on_canvas_configure(event):
            left_canvas.itemconfigure(left_window, width=event.width)

        left.bind("<Configure>", on_left_configure)
        left_canvas.bind("<Configure>", on_canvas_configure)

        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-event.delta / 120), "units")

        def _bind_mousewheel(_event):
            left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(_event):
            left_canvas.unbind_all("<MouseWheel>")

        left_canvas.bind("<Enter>", _bind_mousewheel)
        left_canvas.bind("<Leave>", _unbind_mousewheel)

        ttk.Label(left, text="Параметры рендера", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 8))

        _add_labeled_entry(left, "Ширина (не меньше 500):", self.width_var)
        _add_labeled_entry(left, "Высота (не меньше 500):", self.height_var)
        _add_labeled_entry(left, "SPP:", self.spp_var)
        _add_labeled_entry(left, "Макс. глубина:", self.depth_var)
        _add_labeled_entry(left, "Гамма:", self.gamma_var)
        _add_labeled_entry(left, "Seed:", self.seed_var)
        _add_labeled_entry(left, "Выходной PPM:", self.output_var)

        ttk.Label(left, text="Режим остановки:").pack(anchor="w", pady=(8, 0))
        mode_frame = ttk.Frame(left)
        mode_frame.pack(fill="x", pady=2)
        ttk.Radiobutton(mode_frame, text="По SPP", variable=self.render_mode_var, value="spp").pack(side="left")
        ttk.Radiobutton(mode_frame, text="По времени", variable=self.render_mode_var, value="time").pack(side="left",
                                                                                                         padx=10)

        _add_labeled_entry(left, "Лимит времени, сек:", self.time_limit_var)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(left, text="Источник света", font=("Arial", 11, "bold")).pack(anchor="w")
        _add_labeled_entry(left, "Интенсивность:", self.light_intensity_var)
        _add_labeled_entry(left, "Цвет R:", self.light_r_var)
        _add_labeled_entry(left, "Цвет G:", self.light_g_var)
        _add_labeled_entry(left, "Цвет B:", self.light_b_var)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(left, text="Камера", font=("Arial", 11, "bold")).pack(anchor="w")
        _add_labeled_entry(left, "Cam X:", self.cam_x_var)
        _add_labeled_entry(left, "Cam Y:", self.cam_y_var)
        _add_labeled_entry(left, "Cam Z:", self.cam_z_var)

        _add_labeled_entry(left, "Target X:", self.target_x_var)
        _add_labeled_entry(left, "Target Y:", self.target_y_var)
        _add_labeled_entry(left, "Target Z:", self.target_z_var)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(left, text="Материал объекта", font=("Arial", 11, "bold")).pack(anchor="w")
        material_box = ttk.Combobox(
            left,
            textvariable=self.material_mode_var,
            state="readonly",
            values=["mirror", "diffuse_gray", "diffuse_blue", "mixed"]
        )
        material_box.pack(fill="x", pady=4)

        ttk.Label(left, text="OBJ файл (необязательно):").pack(anchor="w", pady=(10, 0))
        obj_row = ttk.Frame(left)
        obj_row.pack(fill="x", pady=3)
        ttk.Entry(obj_row, textvariable=self.obj_path, width=28).pack(side="left", fill="x", expand=True)
        ttk.Button(obj_row, text="...", width=4, command=self.choose_obj).pack(side="left", padx=(5, 0))

        ttk.Button(left, text="Запустить рендер", command=self.start_render).pack(fill="x", pady=(16, 6))
        ttk.Button(left, text="Остановить", command=self.stop_render).pack(fill="x", pady=4)
        ttk.Button(left, text="Сохранить превью как...", command=self.save_preview_as).pack(fill="x", pady=4)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=12)
        ttk.Label(left, textvariable=self.status_var, wraplength=320).pack(anchor="w", pady=(0, 8))
        ttk.Progressbar(left, variable=self.progress_var, maximum=100).pack(fill="x")

        note = (
            "Примечание:\n"
            "- Свет: протяжённый ламбертовский источник.\n"
            "- Антиалиасинг: случайная выборка внутри пикселя.\n"
            "- Выбор события: по значимости + русская рулетка.\n"
            "- Рендер: прогрессивный."
        )
        ttk.Label(left, text=note, wraplength=320, justify="left").pack(anchor="w", pady=(12, 12))

        ttk.Label(right, text="Превью", font=("Arial", 12, "bold")).pack(anchor="w")
        self.preview_label = ttk.Label(right)
        self.preview_label.pack(fill="both", expand=True, pady=(8, 0))

    def choose_obj(self):
        filename = filedialog.askopenfilename(
            title="Выберите OBJ файл",
            filetypes=[("OBJ files", "*.obj"), ("All files", "*.*")]
        )
        if filename:
            self.obj_path.set(filename)

    def start_render(self):
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            spp = int(self.spp_var.get())
            depth = int(self.depth_var.get())
            gamma = float(self.gamma_var.get())
            seed = int(self.seed_var.get())
            time_limit = float(self.time_limit_var.get())

            if width < 500 or height < 500:
                raise ValueError("Разрешение должно быть не меньше 500x500.")
            if spp <= 0:
                raise ValueError("SPP должно быть положительным.")
            if depth <= 0:
                raise ValueError("Макс. глубина должна быть положительной.")
            if gamma <= 0:
                raise ValueError("Гамма должна быть положительной.")
            if time_limit < 0:
                raise ValueError("Лимит времени не может быть отрицательным.")

            light_intensity = float(self.light_intensity_var.get())
            if light_intensity < 0:
                raise ValueError("Интенсивность света не может быть отрицательной.")

            light_color = _parse_vec3_rgb(
                self.light_r_var,
                self.light_g_var,
                self.light_b_var,
                name="Цвет света"
            )

            cam_pos = Vec3(
                float(self.cam_x_var.get()),
                float(self.cam_y_var.get()),
                float(self.cam_z_var.get())
            )
            cam_target = Vec3(
                float(self.target_x_var.get()),
                float(self.target_y_var.get()),
                float(self.target_z_var.get())
            )

            if (cam_target - cam_pos).length() < EPS:
                raise ValueError("Положение камеры и точка взгляда не должны совпадать.")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Некорректные параметры: {e}")
            return

        random.seed(seed)
        self.progress_var.set(0)
        self.status_var.set("Подготовка сцены...")

        try:
            scene, camera = make_default_scene(
                include_obj_path=self.obj_path.get().strip() or None,
                light_intensity=light_intensity,
                light_color=light_color,
                object_material_mode=self.material_mode_var.get()
            )
        except Exception as e:
            messagebox.showerror("Ошибка сцены", str(e))
            return

        camera.position = cam_pos
        camera.target = cam_target

        self.renderer = PathTracer(scene, camera)
        self.renderer.max_depth = depth
        self.renderer.stop_flag = False

        render_mode = self.render_mode_var.get()
        target_spp = spp if render_mode == "spp" else 10**9
        effective_time_limit = time_limit if render_mode == "time" else 0.0

        def worker():
            try:
                start = time.time()
                self.status_var.set("Рендер выполняется...")

                def on_progress(done_rows, total_rows, pass_id):
                    percent = 100.0 * done_rows / total_rows
                    self.root.after(0, lambda: self.progress_var.set(percent))
                    self.root.after(
                        0,
                        lambda: self.status_var.set(
                            f"Проход {pass_id}: строка {done_rows}/{total_rows}"
                        )
                    )

                def on_preview(current_avg, passes_done, elapsed):
                    image = tonemap_and_gamma(current_avg, gamma=gamma)
                    save_ppm(self.rendered_ppm_temp, image)
                    self.root.after(0, self.update_preview)
                    self.root.after(
                        0,
                        lambda: self.status_var.set(
                            f"Прогрессивный рендер: проходов {passes_done}, время {elapsed:.2f} сек."
                        )
                    )

                framebuffer = self.renderer.render_progressive(
                    width=width,
                    height=height,
                    target_spp=target_spp,
                    time_limit_sec=effective_time_limit,
                    progress_callback=on_progress,
                    preview_callback=on_preview
                )

                image = tonemap_and_gamma(framebuffer, gamma=gamma)
                out_name = self.output_var.get().strip() or "render.ppm"
                save_ppm(out_name, image)
                save_ppm(self.rendered_ppm_temp, image)

                elapsed = time.time() - start
                self.root.after(0, self.update_preview)
                self.root.after(0, lambda: self.progress_var.set(100.0))
                self.root.after(
                    0,
                    lambda: self.status_var.set(
                        f"Готово. Сохранено в {out_name}. Время: {elapsed:.2f} сек."
                    )
                )

            except Exception as exception:
                self.root.after(0, lambda: messagebox.showerror("Ошибка рендера", str(exception)))
                self.root.after(0, lambda: self.status_var.set("Ошибка рендера."))

        threading.Thread(target=worker, daemon=True).start()

    def stop_render(self):
        if self.renderer is not None:
            self.renderer.stop_flag = True
            self.status_var.set("Остановка запрошена...")

    def update_preview(self):
        try:
            self.current_image = tk.PhotoImage(file=self.rendered_ppm_temp)
            self.preview_label.configure(image=self.current_image)
        except Exception as e:
            messagebox.showwarning("Предупреждение", f"Не удалось показать превью: {e}")

    def save_preview_as(self):
        src = self.rendered_ppm_temp
        dst = filedialog.asksaveasfilename(
            title="Сохранить изображение",
            defaultextension=".ppm",
            filetypes=[("PPM image", "*.ppm"), ("All files", "*.*")]
        )
        if not dst:
            return

        try:
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
            messagebox.showinfo("Успех", f"Файл сохранён: {dst}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()

    available_themes = style.theme_names()
    if "clam" in available_themes:
        style.theme_use("clam")

    app = PathTracerApp(root)
    root.mainloop()