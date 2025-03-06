import numpy as np

from amdt.solver.utils import SolverUtils
from amdt.solver.types import SolverForwardParameters, SolverForwardState

from pprint import pprint
from scipy.ndimage import gaussian_filter


class SolverBase:
    """
    Base file for Solver class.
    """

    def __init__(
        self,
        build_config_file="default.ini",
        build={},
        material_config_file="SS316L.ini",
        material={},
        mesh_config_file="scale_millimeter.ini",
        mesh={},
        verbose=False,
        **kwargs,
    ):
        self.verbose = verbose

        #########
        # Build #
        #########
        self.build = SolverUtils.load_config_file("build", build_config_file, build)
        if verbose:
            print("\nBuild Configuration")
            pprint(self.build)

        ############
        # Material #
        ############
        self.material = SolverUtils.load_config_file(
            "material", material_config_file, material
        )
        if verbose:
            print("\nMaterial Configuration")
            pprint(self.material)

        ########
        # Mesh #
        ########
        self.mesh = SolverUtils.load_config_file("mesh", mesh_config_file, mesh)

        x_start = self.mesh["x_min"] - self.mesh["x_start_pad"]
        x_end = self.mesh["x_max"] + self.mesh["x_end_pad"]
        y_start = self.mesh["y_min"] - self.mesh["y_start_pad"]
        y_end = self.mesh["y_max"] + self.mesh["y_end_pad"]
        z_start = self.mesh["z_min"] - self.mesh["z_start_pad"]
        z_end = self.mesh["z_max"] + self.mesh["z_end_pad"]

        self.xs = np.arange(x_start, x_end, step=self.mesh["x_step"])
        self.ys = np.arange(y_start, y_end, step=self.mesh["y_step"])
        self.zs = np.arange(z_start, z_end, step=self.mesh["z_step"])

        #########
        # State #
        #########
        self.location = [
            self.mesh["x_location"],
            self.mesh["y_location"],
            self.mesh["z_location"],
        ]

        self.location_idx = [
            np.argmin(np.abs(self.xs - self.location[0])),
            np.argmin(np.abs(self.ys - self.location[1])),
            np.argmin(np.abs(self.zs - self.location[2])),
        ]

        self.theta = (
            np.ones((len(self.xs), len(self.ys), len(self.zs))) * self.build["t_0"]
        )

        if verbose:
            print("\nMesh Configuration")
            pprint(self.mesh)

        super().__init__(**kwargs)

    def forward(
        self,
        parameters: SolverForwardParameters = {},
        state: SolverForwardState = {},
        model="eagar-tsai",
    ):
        """
        Parameters and state can have values passed through for static method.
        """

        ##############
        # Parameters #
        ##############

        # TODO: Standardize keys here, the mix of greek letters
        # and actual parameter values is not consistent. It would
        # be better to have the written out parameter values as
        # its a bit more descriptive.
        parameter_args = {
            "absorptivity": self.material["absorptivity"],
            "beam_diameter": self.build["beam_diameter"],
            "c_p": self.material["c_p"],
            "k": self.material["k"],
            "power": self.build["power"],
            "rho": self.material["rho"],
            "t_0": self.build["t_0"],
            "t_liquidus": self.material["t_liquidus"],
            "t_solidus": self.material["t_solidus"],
            "velocity": self.build["velocity"],
            "xs": self.xs,
            "ys": self.ys,
            "zs": self.zs,
        }

        for key, value in parameters.items():
            parameter_args[key] = value

        #########
        # State #
        #########

        state_args = {
            "location": self.location,
            "location_idx": self.location_idx,
            "theta": self.theta,
        }

        for key, value in state.items():
            state[key] = value

        #########
        # Model #
        #########

        match model:
            case "eagar-tsai":
                theta = self.eagar_tsai(parameter_args)
            case "rosenthal":
                theta = self.rosenthal(parameter_args)
            case _:
                print(f"'{model}' model not found")

        self.diffuse(parameter_args["dt"])
        self.graft(parameter_args["dt"], parameter_args["phi"], theta)
        # print(self.theta)

    # TODO: Move to its own class
    def diffuse(self, dt):
        # Thermal Diffusivity (Wolfer et al. Equation 1)
        D = self.material["k"] / (self.material["rho"] * self.material["c_p"])

        diffuse_sigma = np.sqrt(2 * D * dt)

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

    # TODO: Move to its own class.
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
        # self.visitedx.append(self.location_idx[0])
        # self.visitedy.append(self.location_idx[1])
