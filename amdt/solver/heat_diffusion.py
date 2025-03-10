import numpy as np

from amdt.solver.types import SolverForwardParameters
from scipy.ndimage import gaussian_filter

class SolverHeatDiffusion:
    """
    Class for modeling heat diffusion from source.
    """

    # TODO: Add in check that dt is always positive and non-zero
    def diffuse(self, dt, theta = None):
        """
        Diffuses heat of `self.theta` of the previous timestep with time delta.
        """

        if theta is None:
            theta = self.theta

        # Thermal Diffusivity (Wolfer et al. Equation 1)
        D = self.material["k"] / (self.material["rho"] * self.material["c_p"])

        diffuse_sigma = np.sqrt(2 * D * dt)

        pad_x = max(int((4 * diffuse_sigma) // (self.mesh["x_step"] * 2)), 1)
        pad_y = max(int((4 * diffuse_sigma) // (self.mesh["y_step"] * 2)), 1)
        pad_z = max(int((4 * diffuse_sigma) // (self.mesh["z_step"] * 2)), 1)

        # Applies padding with mirror reflection along all boundaries.
        pad_reflect = np.pad(
            theta,
            ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)),
            mode="reflect",
        )

        # Subtracts preheat temperature to center around temperature delta.
        theta_padded = pad_reflect - self.build["t_0"]

        theta_padded_bounded = np.copy(theta_padded)

        # Temperature Boundary Condition (2019 Wolfer et al. Figure 3b)
        if self.mesh["b_c"] == "temp":
            # X and Y values are flipped alongside boundary condition
            theta_padded_bounded[-pad_x:, :, :] = -theta_padded[-pad_x:, :, :]
            theta_padded_bounded[:pad_x, :, :] = -theta_padded[:pad_x, :, :]
            theta_padded_bounded[:, -pad_y:, :] = -theta_padded[:, -pad_y:, :]
            theta_padded_bounded[:, :pad_y, :] = -theta_padded[:, :pad_y, :]

        # Flux Boundary Condition (2019 Wolfer et al. Figure 3a)
        if self.mesh["b_c"] == "flux":
            # X and Y values are mirrored alongside boundary condition
            theta_padded_bounded[-pad_x:, :, :] = theta_padded[-pad_x:, :, :]
            theta_padded_bounded[:pad_x, :, :] = theta_padded[:pad_x, :, :]
            theta_padded_bounded[:, -pad_y:, :] = theta_padded[:, -pad_y:, :]
            theta_padded_bounded[:, :pad_y, :] = theta_padded[:, :pad_y, :]

        # Z Padding is the same for either 'temp' or 'flux' boundary condition.
        theta_padded_bounded[:, :, :pad_z] = -theta_padded[:, :, :pad_z]
        theta_padded_bounded[:, :, -pad_z:] = theta_padded[:, :, -pad_z:]

        # TODO: See if there's a sepcific gaussian_filter function that takes in
        # varying sigma values based on axes.
        # not sure which `self.mesh["step"]` should fit here
        sigma = diffuse_sigma / self.mesh["z_step"]
        theta_filtered = gaussian_filter(theta_padded_bounded, sigma=sigma)

        # Crop out the padded areas.
        theta_cropped = theta_filtered[pad_x:-pad_x, pad_y:-pad_y, pad_z:-pad_z]

        # Re-add in the preheat temperature values
        theta = theta_cropped + self.build["t_0"]

        return theta
