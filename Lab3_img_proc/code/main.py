import math
import random
from dataclasses import dataclass
from typing import List, Tuple


SAMPLES = 100000
SEED = 42


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k, self.z * k)

    def __rmul__(self, k: float) -> "Vec3":
        return self.__mul__(k)

    def __truediv__(self, k: float) -> "Vec3":
        return Vec3(self.x / k, self.y / k, self.z / k)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        n = self.norm()
        if n == 0:
            raise ValueError("Нельзя нормализовать нулевой вектор")
        return self / n

    def __repr__(self) -> str:
        return f"({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_subheader(title: str) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def build_orthonormal_basis(n: Vec3) -> Tuple[Vec3, Vec3, Vec3]:
    n = n.normalized()

    if abs(n.x) < 0.9:
        helper = Vec3(1.0, 0.0, 0.0)
    else:
        helper = Vec3(0.0, 1.0, 0.0)

    u = n.cross(helper).normalized()
    v = n.cross(u).normalized()
    return u, v, n


def print_counts(title: str, counts: List[int]) -> None:
    total = sum(counts)
    expected = total / len(counts)
    print_subheader(title)
    print(f"{'Область':>10} | {'Число':>10} | {'Доля':>12} | {'Откл. от ожидаемого':>22}")
    print("-" * 65)
    for i, c in enumerate(counts, start=1):
        frac = c / total
        dev = c - expected
        print(f"{i:>10} | {c:>10} | {frac:>12.6f} | {dev:>22.2f}")
    print(f"\nОжидаемое число в каждой области: {expected:.2f}")


def sample_point_in_triangle(v1: Vec3, v2: Vec3, v3: Vec3, rng: random.Random) -> Tuple[Vec3, Tuple[float, float, float]]:
    r1 = rng.random()
    r2 = rng.random()

    s = math.sqrt(r1)
    l1 = 1.0 - s
    l2 = s * (1.0 - r2)
    l3 = s * r2

    p = v1 * l1 + v2 * l2 + v3 * l3
    return p, (l1, l2, l3)


def triangle_experiment(rng: random.Random, samples: int) -> None:
    print_header("1. РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ ТОЧЕК ВНУТРИ ТРЕУГОЛЬНИКА")

    v1 = Vec3(0.0, 0.0, 0.0)
    v2 = Vec3(4.0, 1.0, 0.0)
    v3 = Vec3(1.0, 5.0, 2.0)

    print(f"V1 = {v1}")
    print(f"V2 = {v2}")
    print(f"V3 = {v3}")
    print(f"Число выборок = {samples}")

    centroid_theoretical = (v1 + v2 + v3) / 3.0

    sum_x = sum_y = sum_z = 0.0
    inside_count = 0

    bary1 = []
    bary2 = []
    bary3 = []

    counts = [0, 0, 0, 0]

    for _ in range(samples):
        p, (l1, l2, l3) = sample_point_in_triangle(v1, v2, v3, rng)

        sum_x += p.x
        sum_y += p.y
        sum_z += p.z

        bary1.append(l1)
        bary2.append(l2)
        bary3.append(l3)

        if l1 >= -1e-12 and l2 >= -1e-12 and l3 >= -1e-12 and abs((l1 + l2 + l3) - 1.0) < 1e-9:
            inside_count += 1

        if l1 >= 0.5:
            counts[0] += 1
        elif l2 >= 0.5:
            counts[1] += 1
        elif l3 >= 0.5:
            counts[2] += 1
        else:
            counts[3] += 1

    centroid_empirical = Vec3(sum_x / samples, sum_y / samples, sum_z / samples)

    print_subheader("Проверка принадлежности")
    print(f"Точек внутри треугольника: {inside_count} из {samples}")

    print_subheader("Проверка равномерности")
    print("Теоретически для равномерного распределения:")
    print("- центр масс выборки должен быть близок к центроиду треугольника")
    print("- средние барицентрические координаты должны быть близки к 1/3")
    print("- в 4 равновеликие части должно попадать примерно по 25% точек")

    print(f"\nТеоретический центроид: {centroid_theoretical}")
    print(f"Эмпирический центроид : {centroid_empirical}")

    print(f"\nСреднее lambda1 = {mean(bary1):.6f} (ожидается ~ 0.333333)")
    print(f"Среднее lambda2 = {mean(bary2):.6f} (ожидается ~ 0.333333)")
    print(f"Среднее lambda3 = {mean(bary3):.6f} (ожидается ~ 0.333333)")

    print_counts("Попадания в 4 равновеликие части треугольника", counts)


def sample_point_in_circle(center: Vec3, normal: Vec3, radius: float, rng: random.Random) -> Tuple[Vec3, float, float]:
    u1 = rng.random()
    u2 = rng.random()

    r = radius * math.sqrt(u1)
    phi = 2.0 * math.pi * u2

    u, v, n = build_orthonormal_basis(normal)

    p = center + u * (r * math.cos(phi)) + v * (r * math.sin(phi))
    return p, r, phi


def circle_experiment(rng: random.Random, samples: int) -> None:
    print_header("2. РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ ТОЧЕК ВНУТРИ КРУГА")

    center = Vec3(1.0, 2.0, -1.0)
    normal = Vec3(1.0, 2.0, 3.0).normalized()
    radius = 3.0

    print(f"C = {center}")
    print(f"N = {normal}")
    print(f"R = {radius}")
    print(f"Число выборок = {samples}")

    u_axis, v_axis, n_axis = build_orthonormal_basis(normal)

    inside_count = 0
    radial_sq_values = []

    counts = [0] * 8

    split_r = radius / math.sqrt(2.0)

    for _ in range(samples):
        p, r, phi = sample_point_in_circle(center, normal, radius, rng)

        d = p - center
        proj_u = d.dot(u_axis)
        proj_v = d.dot(v_axis)
        proj_n = d.dot(n_axis)
        rr = math.sqrt(proj_u * proj_u + proj_v * proj_v)

        if abs(proj_n) < 1e-9 and rr <= radius + 1e-9:
            inside_count += 1

        radial_sq_values.append((rr / radius) ** 2)

        angle = math.atan2(proj_v, proj_u)
        if angle < 0:
            angle += 2.0 * math.pi
        sector = int(angle / (math.pi / 2.0))
        ring = 0 if rr < split_r else 1
        idx = ring * 4 + sector
        counts[idx] += 1

    print_subheader("Проверка принадлежности")
    print(f"Точек внутри круга: {inside_count} из {samples}")

    print_subheader("Проверка равномерности")
    print("Теоретически для равномерного распределения в круге:")
    print("- величина (r/R)^2 должна быть равномерной на [0,1]")
    print("- среднее значение (r/R)^2 должно быть близко к 0.5")
    print("- в 8 равновеликих областей должно попадать примерно одинаковое число точек")

    print(f"\nСреднее (r/R)^2 = {mean(radial_sq_values):.6f} (ожидается ~ 0.500000)")
    print_counts("Попадания в 8 равновеликих областей круга", counts)


def sample_uniform_sphere_direction(rng: random.Random) -> Vec3:
    z = 2.0 * rng.random() - 1.0
    phi = 2.0 * math.pi * rng.random()
    r_xy = math.sqrt(max(0.0, 1.0 - z * z))
    x = r_xy * math.cos(phi)
    y = r_xy * math.sin(phi)
    return Vec3(x, y, z)


def sphere_experiment(rng: random.Random, samples: int) -> None:
    print_header("3. РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ НАПРАВЛЕНИЙ НА ЕДИНИЧНОЙ СФЕРЕ")

    lengths = []
    z_values = []

    counts = [0] * 8
    all_generated = 0

    for _ in range(samples):
        d = sample_uniform_sphere_direction(rng)
        all_generated += 1

        length = d.norm()
        lengths.append(length)
        z_values.append(d.z)

        phi = math.atan2(d.y, d.x)
        if phi < 0:
            phi += 2.0 * math.pi
        sector = int(phi / (math.pi / 2.0))
        hemi = 0 if d.z < 0 else 1
        idx = hemi * 4 + sector
        counts[idx] += 1

    print_subheader("Проверка формирования направлений")
    print(f"Сформировано направлений: {all_generated} из {samples}")

    print_subheader("Проверка корректности")
    print("Теоретически для равномерного распределения на сфере:")
    print("- длина каждого вектора должна быть 1")
    print("- координата z должна быть равномерной на [-1,1]")
    print("- в 8 равновеликих областей сферы должно попадать примерно одинаковое число направлений")

    print(f"\nСредняя длина вектора = {mean(lengths):.6f} (ожидается ~ 1.000000)")
    print(f"Среднее z            = {mean(z_values):.6f} (ожидается ~ 0.000000)")
    print_counts("Попадания в 8 равновеликих областей сферы", counts)


def sample_cosine_weighted_direction(normal: Vec3, rng: random.Random) -> Vec3:
    u1 = rng.random()
    u2 = rng.random()

    r = math.sqrt(u1)
    phi = 2.0 * math.pi * u2

    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - u1))

    u_axis, v_axis, n_axis = build_orthonormal_basis(normal)
    world = u_axis * x + v_axis * y + n_axis * z
    return world.normalized()


def cosine_experiment(rng: random.Random, samples: int) -> None:
    print_header("4. КОСИНУСНОЕ РАСПРЕДЕЛЕНИЕ НАПРАВЛЕНИЙ ОТНОСИТЕЛЬНО N")

    normal = Vec3(0.0, 0.0, 1.0).normalized()
    print(f"N = {normal}")
    print(f"Число выборок = {samples}")

    mu_values = []
    lengths = []
    positive_count = 0

    counts = [0] * 8

    u_axis, v_axis, n_axis = build_orthonormal_basis(normal)

    for _ in range(samples):
        d = sample_cosine_weighted_direction(normal, rng)
        lengths.append(d.norm())

        mu = d.dot(n_axis)
        mu_values.append(mu)

        if mu >= -1e-12:
            positive_count += 1

        x = d.dot(u_axis)
        y = d.dot(v_axis)
        phi = math.atan2(y, x)
        if phi < 0:
            phi += 2.0 * math.pi

        sector = int(phi / (math.pi / 2.0))
        t = mu * mu
        layer = 0 if t < 0.5 else 1
        idx = layer * 4 + sector
        counts[idx] += 1

    print_subheader("Проверка формирования направлений")
    print(f"Сформировано направлений: {samples} из {samples}")
    print(f"Направлений в полусфере N: {positive_count} из {samples}")

    print_subheader("Проверка косинусного распределения")
    print("Теоретически для косинусного распределения:")
    print("- все направления должны лежать в полусфере относительно N")
    print("- длина каждого вектора должна быть 1")
    print("- mu = cos(theta) должно иметь плотность 2*mu на [0,1]")
    print("- значит среднее mu должно быть близко к 2/3")
    print("- величина mu^2 должна быть равномерной на [0,1]")
    print("- в 8 равновероятных областей должно попадать примерно одинаковое число направлений")

    mu_sq_values = [m * m for m in mu_values]

    print(f"\nСредняя длина вектора = {mean(lengths):.6f} (ожидается ~ 1.000000)")
    print(f"Среднее mu=cos(theta) = {mean(mu_values):.6f} (ожидается ~ 0.666667)")
    print(f"Среднее mu^2          = {mean(mu_sq_values):.6f} (ожидается ~ 0.500000)")

    print_counts("Попадания в 8 равновероятных областей косинусного распределения", counts)


def main() -> None:
    rng = random.Random(SEED)

    print(f"Число выборок для каждого эксперимента: {SAMPLES}")
    print(f"Seed = {SEED}")

    triangle_experiment(rng, SAMPLES)
    circle_experiment(rng, SAMPLES)
    sphere_experiment(rng, SAMPLES)
    cosine_experiment(rng, SAMPLES)


if __name__ == "__main__":
    main()