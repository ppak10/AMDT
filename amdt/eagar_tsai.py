import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy import integrate, interpolate, optimize
from skimage import measure

class EagarTsai():
    """
    Class for producing an analytical Eagar-Tsai solution.
    """

    def __init__(
            self,
            dimstep, 
            V = 0.8, 
            bc = 'flux',
            absorp = 1, 
            cp = 455, 
            k = 8.9,
            beamD = 50e-6, 
            rho = 7910,
            P = 200, 
            melt_T = 1673, 
            bounds = {'x': [0, 10000e-6], 'y': [0, 10000e-6], 'z': [-800e-6, 0]},
            location = [0, 0],
            b = 0,
        ):

        self.P = P 
        self.V = V
        self.sigma = beamD / 4 #13.75e-6
        self.A = absorp
        self.rho =  rho
        self.cp = cp
        self.k = k
        self.bc = bc
        self.step = 0
        self.dimstep = dimstep
        self.time = 0
        self.melt_T = melt_T

        self.xs = np.arange(-b + bounds['x'][0], bounds['x'][1]+ b, step = self.dimstep)
        self.ys = np.arange(bounds['y'][0] - b, bounds['y'][1] + b, step = self.dimstep)
        self.zs = np.arange(bounds['z'][0], bounds['z'][1] + self.dimstep, step = self.dimstep)

        self.depths = np.zeros((len(self.xs), len(self.ys)))
        self.D = self.k/(self.rho*self.cp)
        self.depths_pcl = np.zeros((0, 3))
        self.location = location 
        self.location_idx = [np.argmin(np.abs(self.xs - self.location[0])), np.argmin(np.abs(self.ys - self.location[1]))]
        self.a = 4
        self.times = []
        self.T0 = 300

        self.theta = np.ones((len(self.xs), len(self.ys), len(self.zs)))*self.T0

        self.oldellipse = np.zeros((len(self.xs), len(self.ys)))
        self.store_idx = {}
        self.store = []
        self.visitedx = []
        self.visitedy = []
        
    def forward(self, dt, phi, V = None, P = None):

        if P is None:
            P = self.P

        theta = self.solve(dt, phi, P)
        self.diffuse(dt)
        self.graft(dt, phi, theta)

        self.time += dt
    
    def freefunc(self, x, coeff, xs, ys, phi):
        x_coord = xs[:, None, None, None]
        y_coord = ys[None, :, None, None]
        z_coord = self.zs[None, None, :, None]

        xp = -self.V*x*np.cos(phi)
        yp = -self.V*x*np.sin(phi)

        lmbda  = np.sqrt(4*self.D*x)
        gamma  = np.sqrt(2*self.sigma**2 + lmbda**2)
        start = (4*self.D*x)**(-3/2)


        termy = self.sigma*lmbda*np.sqrt(2*np.pi)/(gamma)
        yexp1 = np.exp(-1*((y_coord - yp)**2)/gamma**2)
        termx = termy
        xexp1 = np.exp(-1*((x_coord - xp)**2)/gamma**2)
        yintegral = termy*(yexp1)
        xintegral = termx*xexp1

        zintegral = 2*np.exp(-(z_coord**2)/(4*self.D*x))
        value = coeff*start*xintegral*yintegral*zintegral
        return value


    def solve(self, dt, phi, P):
        """
        Adapted from `Solutions.solve()` and `_altsolve()` helper function.
        """
        coeff = P * self.A / ( 2 * np.pi * self.rho * self.cp * (self.sigma**2) * np.pi**(3/2))
        xs = self.xs - self.xs[len(self.xs)//2]
        ys = self.ys - self.ys[len(self.ys)//2]

        theta = np.ones((len(xs), len(ys), len(self.zs)))*300
    
        integral_result = integrate.fixed_quad(
            self.freefunc,
            dt / 50000,
            dt,
            args=(coeff, xs, ys, phi),
            n = 75
        )
        theta += integral_result[0]
        return theta

    def graft(self, dt, phi, theta):
        l = self.V * dt
        l_new_x = int(np.rint(self.V * dt * np.cos(phi) / self.dimstep))
        l_new_y = int(np.rint(self.V * dt * np.sin(phi) / self.dimstep))
        y = len(self.ys)//2

        y_offset = len(self.ys) // 2
        x_offset = len(self.xs) // 2

        x_roll = -(x_offset) + self.location_idx[0] + l_new_x
        y_roll = -(y_offset) + self.location_idx[1] + l_new_y

        self.theta += np.roll(theta, (x_roll, y_roll, 0), axis = (0, 1, 2)) - 300

        if self.theta.shape == (0, 0, 0):
            breakpoint()

        self.location[0] += l * (np.cos(phi))
        self.location[1] += l * (np.sin(phi))
        self.location_idx[0] += int(np.rint(l * np.cos(phi)/self.dimstep))
        self.location_idx[1] += int(np.rint(l * np.sin(phi)/self.dimstep)) 
        self.visitedx.append(self.location_idx[0])
        self.visitedy.append(self.location_idx[1])

    def diffuse(self, dt):
        diffuse_sigma = np.sqrt(2 * self.D * dt)

        if dt < 0:
            breakpoint()

        padsize = int((4 * diffuse_sigma) // (self.dimstep * 2))

        if padsize == 0:
            padsize = 1
        
        theta_pad = np.pad(self.theta, ((padsize, padsize), (padsize, padsize), (padsize, padsize)), mode = 'reflect') - 300

        theta_pad_flip = np.copy(theta_pad)

        if self.bc == 'temp':
            theta_pad_flip[-padsize:, :, :]  = -theta_pad[-padsize:, :, :]
            theta_pad_flip[:padsize, :, :]  = -theta_pad[:padsize, :, :]
            theta_pad_flip[:, -padsize:, :] = -theta_pad[:, -padsize:, :]
            theta_pad_flip[:, :padsize, :] = -theta_pad[:, :padsize, :]

        if self.bc == 'flux':
            theta_pad_flip[-padsize:, :, :]  = theta_pad[-padsize:, :, :]
            theta_pad_flip[:padsize, :, :]  = theta_pad[:padsize, :, :]
            theta_pad_flip[:, -padsize:, :] = theta_pad[:, -padsize:, :]
            theta_pad_flip[:, :padsize, :] = theta_pad[:, :padsize, :]

        theta_pad_flip[:, :, :padsize] = -theta_pad[:, :, :padsize]
        theta_pad_flip[:, :, -padsize:] = theta_pad[:, :, -padsize:]

        theta_diffuse = gaussian_filter(theta_pad_flip, sigma = diffuse_sigma/self.dimstep)[padsize:-padsize, padsize:-padsize, padsize:-padsize]  + 300
        
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
        xcurrent = np.argmax(self.theta[:, len(self.ys)//2, -1])

        pcm0 = axes[0].pcolormesh(self.xs, self.ys, self.theta[:, :, -1].T, cmap = 'jet', vmin = 300, vmax = 1923)
        axes[0].plot(self.location[0], self.location[1] ,'r.')
        axes[0].plot(self.xs[self.location_idx[0]], self.ys[self.location_idx[1]], 'k.')
        pcm1 = axes[1].pcolormesh(self.xs, self.zs, self.theta[:, len(self.ys)//2, :].T, shading = 'gouraud', cmap = 'jet', vmin = 300, vmax = 4000)
        pcm2 = axes[2].pcolormesh(self.ys, self.zs, self.theta[xcurrent, :, :].T, shading = 'gouraud', cmap = 'jet', vmin = 300, vmax = 4000)
        pcms = [pcm0, pcm1, pcm2]
        scale_x = 1e-6
        scale_y = 1e-6
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
        ticks_y = ticker.FuncFormatter(lambda y, pos:'{0:g}'.format(y/scale_y) )
        iter = 0
        axes[0].set_xlabel(r"x [$\mu$m]")
        axes[0].set_ylabel(r"y [$\mu$m]")
        axes[1].set_xlabel(r"x [$\mu$m]")
        axes[1].set_ylabel(r"z [$\mu$m]")
        axes[2].set_xlabel(r"y [$\mu$m]")
        axes[2].set_ylabel(r"z [$\mu$m]")
        
        for axis, pcm, fig in zip(axes, pcms, figures):
            axis.set_aspect('equal')
            axis.xaxis.set_major_formatter(ticks_x)
            axis.yaxis.set_major_formatter(ticks_y)

            axis.title.set_text(str(round(self.time*1e6)) + r'[$\mu$s] ' + " Power: " + str(np.around(self.P, decimals = 2)) + "W" + " Velocity: " + str(np.around(self.V, decimals = 2)) + r" [m/s]")
            clb = fig.colorbar(pcm, ax = axis)
            clb.ax.set_title(r'T [$K$]')
            iter += 1
        return figures

    def meltpool(self, calc_length = False, calc_width = False, verbose = False):
        y_center = np.unravel_index(np.argmax(self.theta[:, :,-1 ]), self.theta[:, :, -1].shape)[1]
        #  breakpoint()
        if not np.array(self.theta[:,:,-1]>self.melt_T).any():
            print(f"Energy Density too low to melt material, melting temperature: {self.melt_T} K, max temperature: {np.max(self.theta[:,:,-1])} K")
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
                return  depth, depths
        else:
            if calc_length:
                f = interpolate.CubicSpline(self.xs, self.theta[:, y_center, -1] - self.melt_T)
                try:
                    root = optimize.brentq(f, self.xs[1], self.location[0] - self.dimstep)

                    root2 = optimize.brentq(f, self.location[0] - self.dimstep, self.xs[-1])
                    if verbose:
                        print("Length: " + str((root2 - root)*1e6))
                    prop = measure.regionprops(np.array(self.theta[:,:,-1]>self.melt_T, dtype = 'int'))
                    prop_l = prop[0].major_axis_length*self.dimstep
                    print("Length: " + str(prop_l*1e6))

                except:

                    prop = measure.regionprops(np.array(self.theta[:,:,-1]>self.melt_T, dtype = 'int'))
                    if not np.array(self.theta[:,:,-1]>self.melt_T).any():
                        prop_l = 0
                    else:
                        prop_l = prop[0].major_axis_length*self.dimstep
                    length =  prop_l
                    if verbose:
                        print("Length: {:.04} ± {:.04}".format(prop_l*1e6, self.dimstep*1e6))

                
            if calc_width:
                
                widths = []
                for i in range(len(self.xs)):
                    g = interpolate.CubicSpline(self.ys, self.theta[i, :, -1] - self.melt_T)
                    if self.theta[i,y_center,-1] > self.melt_T:
                        root = optimize.brentq(g, self.ys[1],0)
                        root2 = optimize.brentq(g,0, self.ys[-1])
                        widths.append(np.abs(root2-root))
                prop = measure.regionprops(np.array(self.theta[:,:,-1]>self.melt_T, dtype = 'int'))
                prop_w = prop[0].minor_axis_length*self.dimstep
                if verbose:
                    print("Width: {:.04} ± {:.04}".format(prop_w*1e6, self.dimstep*1e6))

            depths = []
            for j in range(len(self.ys)):
                for i in range(len(self.xs)):               
                    if self.theta[i, j, -1] > self.melt_T:
                        g = interpolate.CubicSpline(self.zs, self.theta[i, j, :] - self.melt_T)
                        root = optimize.brentq(g, self.zs[0],self.zs[-1])
                        depths.append(root)
                        self.depths[i, j] = -1*root
                        self.depths_pcl = np.vstack((self.depths_pcl, np.array([self.xs[i], self.ys[j], root])))
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
                return  depth, depths
