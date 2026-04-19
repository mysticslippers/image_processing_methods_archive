from dataclasses import dataclass
from math import sqrt, pi
from typing import List, Tuple


EPS = 1e-9


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value: float) -> "Vec3":
        return Vec3(self.x * value, self.y * value, self.z * value)

    def __rmul__(self, value: float) -> "Vec3":
        return self.__mul__(value)

    def __truediv__(self, value: float) -> "Vec3":
        if abs(value) < EPS:
            raise ValueError("Деление на ноль при работе с вектором")
        return Vec3(self.x / value, self.y / value, self.z / value)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def norm(self) -> float:
        return sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        n = self.norm()
        if n < EPS:
            raise ValueError("Нельзя нормализовать нулевой вектор")
        return self / n

    def clamp01(self) -> "Vec3":
        return Vec3(
            max(0.0, min(1.0, self.x)),
            max(0.0, min(1.0, self.y)),
            max(0.0, min(1.0, self.z))
        )

    def hadamard(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)

    def __repr__(self) -> str:
        return f"({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"


@dataclass
class Light:
    position: Vec3
    axis: Vec3
    intensity0: Vec3


@dataclass
class Surface:
    color: Vec3
    kd: float
    ks: float
    ke: float


@dataclass
class Triangle:
    p0: Vec3
    p1: Vec3
    p2: Vec3

    def edge1(self) -> Vec3:
        return self.p1 - self.p0

    def edge2(self) -> Vec3:
        return self.p2 - self.p0

    def normal(self) -> Vec3:
        n = self.edge1().cross(self.edge2())
        return n.normalized()

    def point_from_local(self, x: float, y: float) -> Vec3:
        e1 = self.edge1().normalized()
        e2 = self.edge2().normalized()
        return self.p0 + e1 * x + e2 * y


def safe_pow(base: float, exponent: float) -> float:
    if base <= 0.0:
        return 0.0
    return base ** exponent


def light_intensity_for_direction(light: Light, source_to_point_hat: Vec3) -> Vec3:
    axis_hat = light.axis.normalized()
    cos_theta = max(0.0, source_to_point_hat.dot(axis_hat))
    return light.intensity0 * cos_theta


def compute_illuminance(light: Light, point: Vec3, normal: Vec3) -> Vec3:
    source_to_point = point - light.position
    r2 = source_to_point.dot(source_to_point)
    if r2 < EPS:
        raise ValueError("Точка совпала с положением источника света")

    source_to_point_hat = source_to_point.normalized()
    point_to_light_hat = (light.position - point).normalized()

    cos_alpha = max(0.0, normal.dot(point_to_light_hat))
    if cos_alpha <= 0.0:
        return Vec3(0.0, 0.0, 0.0)

    intensity = light_intensity_for_direction(light, source_to_point_hat)
    return intensity * (cos_alpha / r2)


def compute_brdf(surface: Surface, normal: Vec3, view_dir: Vec3, point_to_light_hat: Vec3) -> Vec3:
    v_hat = view_dir.normalized()
    l_hat = point_to_light_hat.normalized()

    n_dot_v = normal.dot(v_hat)
    n_dot_l = normal.dot(l_hat)

    if n_dot_v <= 0.0 or n_dot_l <= 0.0:
        return Vec3(0.0, 0.0, 0.0)

    h_raw = v_hat + l_hat
    if h_raw.norm() < EPS:
        specular_term = 0.0
    else:
        h_hat = h_raw.normalized()
        specular_term = surface.ks * safe_pow(max(0.0, h_hat.dot(normal)), surface.ke)

    scalar = surface.kd + specular_term
    return surface.color * scalar


def compute_brightness(
    lights: List[Light],
    surface: Surface,
    triangle: Triangle,
    point: Vec3,
    view_dir: Vec3
) -> Tuple[List[Vec3], Vec3]:
    n = triangle.normal()
    illuminances = []
    total = Vec3(0.0, 0.0, 0.0)

    v_hat = view_dir.normalized()
    if n.dot(v_hat) <= 0.0:
        zero_list = [Vec3(0.0, 0.0, 0.0) for _ in lights]
        return zero_list, Vec3(0.0, 0.0, 0.0)

    for light in lights:
        point_to_light_hat = (light.position - point).normalized()

        e = compute_illuminance(light, point, n)
        f = compute_brdf(surface, n, view_dir, point_to_light_hat)

        l_i = e.hadamard(f)
        illuminances.append(e)
        total = total + l_i

    total = total * (1.0 / pi)
    return illuminances, total


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_input_data(
    lights: List[Light],
    triangle: Triangle,
    local_points: List[Tuple[float, float]],
    view_dir: Vec3,
    surface: Surface
) -> None:
    print_header("ВХОДНЫЕ ДАННЫЕ")

    for i, light in enumerate(lights, start=1):
        print(f"I0{i}(RGB) = {light.intensity0}")
    for i, light in enumerate(lights, start=1):
        print(f"O{i} = {light.axis.normalized()}")
    for i, light in enumerate(lights, start=1):
        print(f"P_L{i} = {light.position}")

    print(f"P0 = {triangle.p0}")
    print(f"P1 = {triangle.p1}")
    print(f"P2 = {triangle.p2}")

    for i, (x, y) in enumerate(local_points, start=1):
        print(f"x{i} = {x:.4f}, y{i} = {y:.4f}")

    print(f"V = {view_dir.normalized()}")
    print(f"K(RGB) = {surface.color}")
    print(f"kd = {surface.kd:.4f}")
    print(f"ks = {surface.ks:.4f}")
    print(f"ke = {surface.ke:.4f}")


def print_table(
    title: str,
    local_points: List[Tuple[float, float]],
    values: List[Vec3]
) -> None:
    print_header(title)
    print(f"{'x':>10} {'y':>10} {'RGB':>35}")
    print("-" * 60)
    for (x, y), value in zip(local_points, values):
        print(f"{x:>10.4f} {y:>10.4f} {str(value):>35}")


def main() -> None:
    lights = [
        Light(
            position=Vec3(-2.0, 3.0, 5.0),
            axis=Vec3(0.6, -0.3, -0.7),
            intensity0=Vec3(180.0, 120.0, 90.0)
        ),
        Light(
            position=Vec3(4.0, -1.0, 6.0),
            axis=Vec3(-0.5, 0.2, -0.8),
            intensity0=Vec3(100.0, 160.0, 210.0)
        )
    ]

    triangle = Triangle(
        p0=Vec3(0.0, 0.0, 0.0),
        p1=Vec3(4.0, 0.0, 1.0),
        p2=Vec3(0.0, 3.0, 2.0)
    )

    surface = Surface(
        color=Vec3(0.80, 0.55, 0.35),
        kd=0.65,
        ks=0.30,
        ke=20.0
    )

    view_dir = Vec3(-1.0, -1.0, 2.0)

    local_points = [
        (0.2, 0.2),
        (0.6, 0.4),
        (1.0, 0.8),
        (1.4, 1.0),
        (1.8, 1.2),
    ]

    print_input_data(lights, triangle, local_points, view_dir, surface)

    print_header("НОРМАЛЬ К ПЛОСКОСТИ")
    n = triangle.normal()
    print(f"N = {n}")

    global_points = []
    e1_values = []
    e2_values = []
    l_values = []

    for x, y in local_points:
        point = triangle.point_from_local(x, y)
        global_points.append(point)

        illuminances, brightness = compute_brightness(
            lights=lights,
            surface=surface,
            triangle=triangle,
            point=point,
            view_dir=view_dir
        )

        e1_values.append(illuminances[0])
        e2_values.append(illuminances[1])
        l_values.append(brightness)

    print_header("ГЛОБАЛЬНЫЕ КООРДИНАТЫ ТОЧЕК")
    print(f"{'x(local)':>10} {'y(local)':>10} {'P_T(global)':>35}")
    print("-" * 60)
    for (x, y), pt in zip(local_points, global_points):
        print(f"{x:>10.4f} {y:>10.4f} {str(pt):>35}")

    print_table("ОСВЕЩЕННОСТЬ E1(RGB, P_T)", local_points, e1_values)
    print_table("ОСВЕЩЕННОСТЬ E2(RGB, P_T)", local_points, e2_values)
    print_table("ЯРКОСТЬ L(RGB, P_T, v)", local_points, l_values)


if __name__ == "__main__":
    main()
