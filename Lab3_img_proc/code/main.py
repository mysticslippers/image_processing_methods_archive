import math
import random
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import List, Tuple

import matplotlib.pyplot as plt


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
            self.x * other.y - self.y * other.x,
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


def sample_point_in_triangle(
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
    rng: random.Random,
) -> Tuple[Vec3, Tuple[float, float, float]]:
    r1 = rng.random()
    r2 = rng.random()

    s = math.sqrt(r1)

    l1 = 1.0 - s
    l2 = s * (1.0 - r2)
    l3 = s * r2

    p = v1 * l1 + v2 * l2 + v3 * l3

    return p, (l1, l2, l3)


def sample_point_in_circle(
    center: Vec3,
    normal: Vec3,
    radius: float,
    rng: random.Random,
) -> Tuple[Vec3, float, float]:
    u1 = rng.random()
    u2 = rng.random()

    r = radius * math.sqrt(u1)
    phi = 2.0 * math.pi * u2

    u, v, _ = build_orthonormal_basis(normal)

    p = center + u * (r * math.cos(phi)) + v * (r * math.sin(phi))

    return p, r, phi


def sample_uniform_sphere_direction(rng: random.Random) -> Vec3:
    z = 2.0 * rng.random() - 1.0
    phi = 2.0 * math.pi * rng.random()

    r_xy = math.sqrt(max(0.0, 1.0 - z * z))

    x = r_xy * math.cos(phi)
    y = r_xy * math.sin(phi)

    return Vec3(x, y, z)


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


def format_counts(title: str, counts: List[int]) -> str:
    total = sum(counts)
    expected = total / len(counts)

    lines = []
    lines.append(title)
    lines.append("-" * 70)
    lines.append(f"{'Область':>10} | {'Число':>10} | {'Доля':>12} | {'Отклонение':>14}")
    lines.append("-" * 70)

    for i, c in enumerate(counts, start=1):
        frac = c / total
        dev = c - expected
        lines.append(f"{i:>10} | {c:>10} | {frac:>12.6f} | {dev:>14.2f}")

    lines.append(f"\nОжидаемое число в каждой области: {expected:.2f}")
    lines.append("")

    return "\n".join(lines)


def triangle_experiment(samples: int, rng: random.Random) -> Tuple[str, List[Vec3], List[int]]:
    v1 = Vec3(0.0, 0.0, 0.0)
    v2 = Vec3(4.0, 1.0, 0.0)
    v3 = Vec3(1.0, 5.0, 2.0)

    points = []
    bary1 = []
    bary2 = []
    bary3 = []

    counts = [0, 0, 0, 0]
    inside_count = 0

    sum_x = sum_y = sum_z = 0.0

    for _ in range(samples):
        p, (l1, l2, l3) = sample_point_in_triangle(v1, v2, v3, rng)

        points.append(p)

        sum_x += p.x
        sum_y += p.y
        sum_z += p.z

        bary1.append(l1)
        bary2.append(l2)
        bary3.append(l3)

        if (
            l1 >= -1e-12
            and l2 >= -1e-12
            and l3 >= -1e-12
            and abs((l1 + l2 + l3) - 1.0) < 1e-9
        ):
            inside_count += 1

        if l1 >= 0.5:
            counts[0] += 1
        elif l2 >= 0.5:
            counts[1] += 1
        elif l3 >= 0.5:
            counts[2] += 1
        else:
            counts[3] += 1

    centroid_theoretical = (v1 + v2 + v3) / 3.0
    centroid_empirical = Vec3(sum_x / samples, sum_y / samples, sum_z / samples)

    text = []
    text.append("1. РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ ТОЧЕК ВНУТРИ ТРЕУГОЛЬНИКА")
    text.append("=" * 80)
    text.append(f"V1 = {v1}")
    text.append(f"V2 = {v2}")
    text.append(f"V3 = {v3}")
    text.append(f"Число выборок = {samples}")
    text.append("")
    text.append(f"Точек внутри треугольника: {inside_count} из {samples}")
    text.append(f"Теоретический центроид: {centroid_theoretical}")
    text.append(f"Эмпирический центроид : {centroid_empirical}")
    text.append("")
    text.append(f"Среднее lambda1 = {mean(bary1):.6f} / ожидается 0.333333")
    text.append(f"Среднее lambda2 = {mean(bary2):.6f} / ожидается 0.333333")
    text.append(f"Среднее lambda3 = {mean(bary3):.6f} / ожидается 0.333333")
    text.append("")
    text.append(format_counts("Попадания в 4 равновеликие части треугольника", counts))

    return "\n".join(text), points, counts


def circle_experiment(samples: int, rng: random.Random) -> Tuple[str, List[Vec3], List[int]]:
    center = Vec3(1.0, 2.0, -1.0)
    normal = Vec3(1.0, 2.0, 3.0).normalized()
    radius = 3.0

    u_axis, v_axis, n_axis = build_orthonormal_basis(normal)

    points = []
    radial_sq_values = []
    counts = [0] * 8

    inside_count = 0
    split_r = radius / math.sqrt(2.0)

    for _ in range(samples):
        p, _, _ = sample_point_in_circle(center, normal, radius, rng)
        points.append(p)

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

    text = []
    text.append("2. РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ ТОЧЕК ВНУТРИ КРУГА")
    text.append("=" * 80)
    text.append(f"C = {center}")
    text.append(f"N = {normal}")
    text.append(f"R = {radius}")
    text.append(f"Число выборок = {samples}")
    text.append("")
    text.append(f"Точек внутри круга: {inside_count} из {samples}")
    text.append(f"Среднее (r/R)^2 = {mean(radial_sq_values):.6f} / ожидается 0.500000")
    text.append("")
    text.append(format_counts("Попадания в 8 равновеликих областей круга", counts))

    return "\n".join(text), points, counts


def sphere_experiment(samples: int, rng: random.Random) -> Tuple[str, List[Vec3], List[int]]:
    points = []
    lengths = []
    z_values = []
    counts = [0] * 8

    for _ in range(samples):
        d = sample_uniform_sphere_direction(rng)

        points.append(d)
        lengths.append(d.norm())
        z_values.append(d.z)

        phi = math.atan2(d.y, d.x)
        if phi < 0:
            phi += 2.0 * math.pi

        sector = int(phi / (math.pi / 2.0))
        hemi = 0 if d.z < 0 else 1
        idx = hemi * 4 + sector

        counts[idx] += 1

    text = []
    text.append("3. РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ НАПРАВЛЕНИЙ НА ЕДИНИЧНОЙ СФЕРЕ")
    text.append("=" * 80)
    text.append(f"Сформировано направлений: {samples} из {samples}")
    text.append(f"Средняя длина вектора = {mean(lengths):.6f} / ожидается 1.000000")
    text.append(f"Среднее z = {mean(z_values):.6f} / ожидается 0.000000")
    text.append("")
    text.append(format_counts("Попадания в 8 равновеликих областей сферы", counts))

    return "\n".join(text), points, counts


def cosine_experiment(samples: int, rng: random.Random) -> Tuple[str, List[Vec3], List[int]]:
    normal = Vec3(0.0, 0.0, 1.0).normalized()

    u_axis, v_axis, n_axis = build_orthonormal_basis(normal)

    points = []
    lengths = []
    mu_values = []
    counts = [0] * 8
    positive_count = 0

    for _ in range(samples):
        d = sample_cosine_weighted_direction(normal, rng)

        points.append(d)
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

    mu_sq_values = [m * m for m in mu_values]

    text = []
    text.append("4. КОСИНУСНОЕ РАСПРЕДЕЛЕНИЕ НАПРАВЛЕНИЙ ОТНОСИТЕЛЬНО N")
    text.append("=" * 80)
    text.append(f"N = {normal}")
    text.append(f"Сформировано направлений: {samples} из {samples}")
    text.append(f"Направлений в полусфере N: {positive_count} из {samples}")
    text.append(f"Средняя длина вектора = {mean(lengths):.6f} / ожидается 1.000000")
    text.append(f"Среднее mu = cos(theta) = {mean(mu_values):.6f} / ожидается 0.666667")
    text.append(f"Среднее mu^2 = {mean(mu_sq_values):.6f} / ожидается 0.500000")
    text.append("")
    text.append(format_counts("Попадания в 8 равновероятных областей", counts))

    return "\n".join(text), points, counts


def plot_points_2d(points: List[Vec3], title: str, x_label: str = "X", y_label: str = "Y") -> None:
    xs = [p.x for p in points]
    ys = [p.y for p in points]

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=1)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis("equal")
    plt.grid(True)


def plot_points_3d(points: List[Vec3], title: str) -> None:
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(xs, ys, zs, s=1)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1, 1, 1])


def plot_counts(counts: List[int], title: str) -> None:
    labels = [f"{i + 1}" for i in range(len(counts))]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, counts)
    plt.title(title)
    plt.xlabel("Область")
    plt.ylabel("Количество точек / направлений")
    plt.grid(axis="y")


class ExperimentApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Лабораторная работа №3 — визуализация распределений")
        self.geometry("980x720")
        self.minsize(760, 520)

        self.samples_var = tk.StringVar(value=str(SAMPLES))
        self.seed_var = tk.StringVar(value=str(SEED))
        self.experiment_var = tk.StringVar(value="all")

        self._build_widgets()

    def _build_widgets(self) -> None:
        main_frame = ttk.Frame(self, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        settings_frame = ttk.LabelFrame(main_frame, text="Параметры", padding=10)
        settings_frame.pack(fill=tk.X)

        ttk.Label(settings_frame, text="Число выборок:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(settings_frame, textvariable=self.samples_var, width=15).grid(row=0, column=1, padx=5)

        ttk.Label(settings_frame, text="Seed:").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Entry(settings_frame, textvariable=self.seed_var, width=15).grid(row=0, column=3, padx=5)

        ttk.Label(settings_frame, text="Эксперимент:").grid(row=0, column=4, sticky=tk.W, padx=5)

        experiment_box = ttk.Combobox(
            settings_frame,
            textvariable=self.experiment_var,
            values=("all", "triangle", "circle", "sphere", "cosine"),
            state="readonly",
            width=15,
        )
        experiment_box.grid(row=0, column=5, padx=5)

        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            buttons_frame,
            text="Запустить и построить графики",
            command=self.run_experiment,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            buttons_frame,
            text="Очистить вывод",
            command=self.clear_output,
        ).pack(side=tk.LEFT, padx=5)

        output_frame = ttk.LabelFrame(main_frame, text="Результаты вычислений", padding=8)
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(output_frame, wrap=tk.NONE, font=("Consolas", 10))

        y_scroll = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        x_scroll = ttk.Scrollbar(output_frame, orient=tk.HORIZONTAL, command=self.output_text.xview)

        self.output_text.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
        )

        self.output_text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.output_text.insert(
            tk.END,
            "Выберите эксперимент и нажмите кнопку запуска.\n"
            "Программа выведет численные проверки и построит графики через Matplotlib.\n",
        )

    def read_parameters(self) -> Tuple[int, int, str]:
        try:
            samples = int(self.samples_var.get())
            seed = int(self.seed_var.get())
        except ValueError as exc:
            raise ValueError("Число выборок и seed должны быть целыми числами") from exc

        if samples <= 0:
            raise ValueError("Число выборок должно быть положительным")

        experiment = self.experiment_var.get()

        return samples, seed, experiment

    def run_experiment(self) -> None:
        try:
            samples, seed, experiment = self.read_parameters()
        except ValueError as exc:
            messagebox.showerror("Ошибка параметров", str(exc))
            return

        rng = random.Random(seed)

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"Число выборок: {samples}\n")
        self.output_text.insert(tk.END, f"Seed: {seed}\n\n")

        if experiment in ("all", "triangle"):
            text, points, counts = triangle_experiment(samples, rng)
            self.output_text.insert(tk.END, text + "\n\n")

            plot_points_2d(points, "Треугольник: равномерное распределение")
            plot_counts(counts, "Треугольник: попадания в 4 равновеликие области")

        if experiment in ("all", "circle"):
            text, points, counts = circle_experiment(samples, rng)
            self.output_text.insert(tk.END, text + "\n\n")

            plot_points_2d(points, "Круг: равномерное распределение")
            plot_counts(counts, "Круг: попадания в 8 равновеликих областей")

        if experiment in ("all", "sphere"):
            text, points, counts = sphere_experiment(samples, rng)
            self.output_text.insert(tk.END, text + "\n\n")

            plot_points_3d(points, "Сфера: равномерное распределение направлений")
            plot_counts(counts, "Сфера: попадания в 8 равновеликих областей")

        if experiment in ("all", "cosine"):
            text, points, counts = cosine_experiment(samples, rng)
            self.output_text.insert(tk.END, text + "\n\n")

            plot_points_3d(points, "Косинусное распределение направлений относительно N")
            plot_counts(counts, "Косинусное распределение: 8 равновероятных областей")

        plt.show()

    def clear_output(self) -> None:
        self.output_text.delete("1.0", tk.END)


def main() -> None:
    app = ExperimentApp()
    app.mainloop()


if __name__ == "__main__":
    main()