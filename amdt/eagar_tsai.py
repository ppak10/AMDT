import configparser
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy import integrate, interpolate, optimize
from skimage import measure

dir_path = Path(__file__).parent


def load_config_file(config_dir, config_file, config_override):
    config = configparser.ConfigParser()
    config.read(os.path.join(dir_path, config_dir, config_file))
    output = {}

    for section in config.sections():
        for key, value in config[section].items():
            if section == "float":
                output[key] = float(value)
            else:
                # Defaults to string
                output[key] = value

    for key, value in config_override.items():
        output[key] = value

    return output


class EagarTsai:
    """
    Class for producing an analytical Eagar-Tsai solution.
    """

    def __init__(
        self,
        mesh_config_file="scale_millimeter.ini",
        mesh={},
        build_config_file="nominal.ini",
        build={},
        material_config_file="SS316L.ini",
        material={},
    ):
        #########
        # Build #
        #########
        self.build = load_config_file("build", build_config_file, build)
        print(self.build)

        ############
        # Material #
        ############
        self.material = load_config_file("material", material_config_file, material)

        # Thermal Diffusivity
        self.D = self.material["k"] / (self.material["rho"] * self.material["c_p"])

        ########
        # Mesh #
        ########
        self.mesh = load_config_file("mesh", mesh_config_file, mesh)
        x_start = self.mesh["x_min"] - self.mesh["x_start_pad"]
        x_end = self.mesh["x_max"] + self.mesh["x_end_pad"]
        y_start = self.mesh["y_min"] - self.mesh["y_start_pad"]
        y_end = self.mesh["y_max"] + self.mesh["y_end_pad"]
        z_start = self.mesh["z_min"] - self.mesh["z_start_pad"]
        z_end = self.mesh["z_max"] + self.mesh["z_end_pad"]

        self.xs = np.arange(x_start, x_end, step=self.mesh["x_step"])
        self.ys = np.arange(y_start, y_end, step=self.mesh["y_step"])
        self.zs = np.arange(z_start, z_end, step=self.mesh["z_step"])

        self.step = 0
        self.time = 0

        self.depths = np.zeros((len(self.xs), len(self.ys)))
        self.depths_pcl = np.zeros((0, 3))
        self.location = [self.mesh["x_location"], self.mesh["y_location"]]
        self.location_idx = [
            np.argmin(np.abs(self.xs - self.location[0])),
            np.argmin(np.abs(self.ys - self.location[1])),
        ]
        self.times = []

        self.theta = (
            np.ones((len(self.xs), len(self.ys), len(self.zs))) * self.build["t_0"]
        )

        self.oldellipse = np.zeros((len(self.xs), len(self.ys)))
        self.store_idx = {}
        self.store = []
        self.visitedx = []
        self.visitedy = []

    def forward(self, dt, phi, power=None, velocity=None):

        if power is None:
            power = self.build["power"]

        theta = self.solve(dt, phi, power)
        self.diffuse(dt)
        self.graft(dt, phi, theta)

        self.time += dt

    def freefunc(self, x, coeff, xs, ys, phi):
        # print(f"self.D {self.D}")
        x_coord = xs[:, None, None, None]
        y_coord = ys[None, :, None, None]
        z_coord = self.zs[None, None, :, None]

        xp = -self.build["velocity"] * x * np.cos(phi)
        yp = -self.build["velocity"] * x * np.sin(phi)

        sigma = self.build["beam_diameter"] / 4  # 13.75e-6
        lmbda = np.sqrt(4 * self.D * x)
        gamma = np.sqrt(2 * sigma**2 + lmbda**2)
        start = (4 * self.D * x) ** (-3 / 2)
        # print(x)
        # print(f"start {start}")

        termy = sigma * lmbda * np.sqrt(2 * np.pi) / (gamma)
        yexp1 = np.exp(-1 * ((y_coord - yp) ** 2) / gamma**2)
        termx = termy
        xexp1 = np.exp(-1 * ((x_coord - xp) ** 2) / gamma**2)
        yintegral = termy * (yexp1)
        xintegral = termx * xexp1

        zintegral = 2 * np.exp(-(z_coord**2) / (4 * self.D * x))
        value = coeff * start * xintegral * yintegral * zintegral
        return value

    def solve(self, dt, phi, P):
        """
        Adapted from `Solutions.solve()` and `_altsolve()` helper function.
        """
        sigma = self.build["beam_diameter"] / 4  # 13.75e-6
        coeff = (
            P
            * self.material["alpha"]
            / (
                2
                * np.pi
                * self.material["rho"]
                * self.material["c_p"]
                * sigma**2
                * np.pi ** (3 / 2)
            )
        )
        xs = self.xs - self.xs[len(self.xs) // 2]
        ys = self.ys - self.ys[len(self.ys) // 2]

        theta = np.ones((len(xs), len(ys), len(self.zs))) * 300

        integral_result = integrate.fixed_quad(
            self.freefunc, dt / 50000, dt, args=(coeff, xs, ys, phi), n=75
        )
        theta += integral_result[0]
        return theta

    def graft(self, dt, phi, theta):
        l = self.build["velocity"] * dt
        l_new_x = int(
            np.rint(self.build["velocity"] * dt * np.cos(phi) / self.mesh["x_step"])
        )
        l_new_y = int(
            np.rint(self.build["velocity"] * dt * np.sin(phi) / self.mesh["y_step"])
        )
        y = len(self.ys) // 2

        y_offset = len(self.ys) // 2
        x_offset = len(self.xs) // 2

        x_roll = -(x_offset) + self.location_idx[0] + l_new_x
        y_roll = -(y_offset) + self.location_idx[1] + l_new_y

        self.theta += np.roll(theta, (x_roll, y_roll, 0), axis=(0, 1, 2)) - 300

        if self.theta.shape == (0, 0, 0):
            breakpoint()

        self.location[0] += l * (np.cos(phi))
        self.location[1] += l * (np.sin(phi))
        self.location_idx[0] += int(np.rint(l * np.cos(phi) / self.mesh["x_step"]))
        self.location_idx[1] += int(np.rint(l * np.sin(phi) / self.mesh["y_step"]))
        self.visitedx.append(self.location_idx[0])
        self.visitedy.append(self.location_idx[1])

    def diffuse(self, dt):
        diffuse_sigma = np.sqrt(2 * self.D * dt)

        if dt < 0:
            breakpoint()

        # not sure which `self.mesh["step"]` should fit here
        padsize = int((4 * diffuse_sigma) // (self.mesh["z_step"] * 2))

        if padsize == 0:
            padsize = 1

        theta_pad = (
            np.pad(
                self.theta,
                ((padsize, padsize), (padsize, padsize), (padsize, padsize)),
                mode="reflect",
            )
            - 300
        )

        theta_pad_flip = np.copy(theta_pad)

        if self.mesh["b_c"] == "temp":
            theta_pad_flip[-padsize:, :, :] = -theta_pad[-padsize:, :, :]
            theta_pad_flip[:padsize, :, :] = -theta_pad[:padsize, :, :]
            theta_pad_flip[:, -padsize:, :] = -theta_pad[:, -padsize:, :]
            theta_pad_flip[:, :padsize, :] = -theta_pad[:, :padsize, :]

        if self.mesh["b_c"] == "flux":
            theta_pad_flip[-padsize:, :, :] = theta_pad[-padsize:, :, :]
            theta_pad_flip[:padsize, :, :] = theta_pad[:padsize, :, :]
            theta_pad_flip[:, -padsize:, :] = theta_pad[:, -padsize:, :]
            theta_pad_flip[:, :padsize, :] = theta_pad[:, :padsize, :]

        theta_pad_flip[:, :, :padsize] = -theta_pad[:, :, :padsize]
        theta_pad_flip[:, :, -padsize:] = theta_pad[:, :, -padsize:]

        # not sure which `self.mesh["step"]` should fit here
        theta_diffuse = (
            gaussian_filter(theta_pad_flip, sigma=diffuse_sigma / self.mesh["z_step"])[
                padsize:-padsize, padsize:-padsize, padsize:-padsize
            ]
            + 300
        )

        self.theta = theta_diffuse
        return theta_diffuse

    def get_coords(self):
        return self.xs, self.ys, self.zs

    # Plot cross sections of domain
    def plot(self):
        figures = []
        axes = []

        for i in range(3):
            fig = plt.figure()
            figures.append(fig)
            axes.append(fig.add_subplot(1, 1, 1))
        xcurrent = np.argmax(self.theta[:, len(self.ys) // 2, -1])

        pcm0 = axes[0].pcolormesh(
            self.xs, self.ys, self.theta[:, :, -1].T, cmap="jet", vmin=300, vmax=1923
        )
        axes[0].plot(self.location[0], self.location[1], "r.")
        axes[0].plot(self.xs[self.location_idx[0]], self.ys[self.location_idx[1]], "k.")
        pcm1 = axes[1].pcolormesh(
            self.xs,
            self.zs,
            self.theta[:, len(self.ys) // 2, :].T,
            shading="gouraud",
            cmap="jet",
            vmin=300,
            vmax=4000,
        )
        pcm2 = axes[2].pcolormesh(
            self.ys,
            self.zs,
            self.theta[xcurrent, :, :].T,
            shading="gouraud",
            cmap="jet",
            vmin=300,
            vmax=4000,
        )
        pcms = [pcm0, pcm1, pcm2]
        scale_x = 1e-6
        scale_y = 1e-6
        ticks_x = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale_x))
        ticks_y = ticker.FuncFormatter(lambda y, pos: "{0:g}".format(y / scale_y))
        iter = 0
        axes[0].set_xlabel(r"x [$\mu$m]")
        axes[0].set_ylabel(r"y [$\mu$m]")
        axes[1].set_xlabel(r"x [$\mu$m]")
        axes[1].set_ylabel(r"z [$\mu$m]")
        axes[2].set_xlabel(r"y [$\mu$m]")
        axes[2].set_ylabel(r"z [$\mu$m]")

        for axis, pcm, fig in zip(axes, pcms, figures):
            axis.set_aspect("equal")
            axis.xaxis.set_major_formatter(ticks_x)
            axis.yaxis.set_major_formatter(ticks_y)

            axis.title.set_text(
                str(round(self.time * 1e6))
                + r"[$\mu$s] "
                + " Power: "
                + str(np.around(self.build["power"], decimals=2))
                + "W"
                + " Velocity: "
                + str(np.around(self.build["velocity"], decimals=2))
                + r" [m/s]"
            )
            clb = fig.colorbar(pcm, ax=axis)
            clb.ax.set_title(r"T [$K$]")
            iter += 1
        return figures

    def meltpool(self, calc_length=False, calc_width=False, verbose=False):
        y_center = np.unravel_index(
            np.argmax(self.theta[:, :, -1]), self.theta[:, :, -1].shape
        )[1]
        #  breakpoint()
        if not np.array(self.theta[:, :, -1] > self.material["t_melt"]).any():
            print(
                f"Energy Density too low to melt material, melting temperature: {self.material['t_melt']} K, max temperature: {np.max(self.theta[:,:,-1])} K"
            )
            prop_l = 0
            prop_w = 0
            depth = 0
            if calc_length and calc_width:
                return prop_w, prop_l, depth
            elif calc_length:
                return prop_l, depth
            elif calc_width:
                return prop_w, depth
            else:
                return depth, depths
        else:
            if calc_length:
                f = interpolate.CubicSpline(
                    self.xs, self.theta[:, y_center, -1] - self.material["t_melt"]
                )
                try:
                    root = optimize.brentq(
                        f, self.xs[1], self.location[0] - self.mesh["x_step"]
                    )

                    root2 = optimize.brentq(
                        f, self.location[0] - self.mesh["x_step"], self.xs[-1]
                    )
                    if verbose:
                        print("Length: " + str((root2 - root) * 1e6))
                    prop = measure.regionprops(
                        np.array(
                            self.theta[:, :, -1] > self.material["t_melt"], dtype="int"
                        )
                    )
                    prop_l = prop[0].major_axis_length * self.mesh["x_step"]
                    print("Length: " + str(prop_l * 1e6))

                except:

                    prop = measure.regionprops(
                        np.array(
                            self.theta[:, :, -1] > self.material["t_melt"], dtype="int"
                        )
                    )
                    if not np.array(
                        self.theta[:, :, -1] > self.material["t_melt"]
                    ).any():
                        prop_l = 0
                    else:
                        prop_l = prop[0].major_axis_length * self.mesh["x_step"]
                    length = prop_l
                    if verbose:
                        print(
                            "Length: {:.04} ± {:.04}".format(
                                prop_l * 1e6, self.mesh["x_step"] * 1e6
                            )
                        )

            if calc_width:

                widths = []
                for i in range(len(self.xs)):
                    g = interpolate.CubicSpline(
                        self.ys, self.theta[i, :, -1] - self.material["t_melt"]
                    )
                    if self.theta[i, y_center, -1] > self.material["t_melt"]:
                        root = optimize.brentq(g, self.ys[1], 0)
                        root2 = optimize.brentq(g, 0, self.ys[-1])
                        widths.append(np.abs(root2 - root))
                prop = measure.regionprops(
                    np.array(
                        self.theta[:, :, -1] > self.material["t_melt"], dtype="int"
                    )
                )
                prop_w = prop[0].minor_axis_length * self.mesh["y_step"]
                if verbose:
                    print(
                        "Width: {:.04} ± {:.04}".format(
                            prop_w * 1e6, self.mesh["y_step"] * 1e6
                        )
                    )

            depths = []
            for j in range(len(self.ys)):
                for i in range(len(self.xs)):
                    if self.theta[i, j, -1] > self.material["t_melt"]:
                        g = interpolate.CubicSpline(
                            self.zs, self.theta[i, j, :] - self.material["t_melt"]
                        )
                        root = optimize.brentq(g, self.zs[0], self.zs[-1])
                        depths.append(root)
                        self.depths[i, j] = -1 * root
                        self.depths_pcl = np.vstack(
                            (self.depths_pcl, np.array([self.xs[i], self.ys[j], root]))
                        )
            if len(depths) == 0:
                depth = 0
            else:
                depth = np.min(depths)
            if calc_length and calc_width:
                return prop_w, prop_l, depth
            elif calc_length:
                return prop_l, depth
            elif calc_width:
                return prop_w, depth
            else:
                return depth, depths
