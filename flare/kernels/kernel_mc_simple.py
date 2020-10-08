import numpy as np
from ..env import AtomicEnvironment
from .two_body_mc_simple import TwoBodyKernel
from .three_body_mc_simple import ThreeBodyKernel


class Kernel:
    def __init__(self, kernel_list, hyps, cutoffs):
        """
        Args:
            kernel_list

        Example:
        >>> kernel_list = ["twobody", "threebody"]
        >>> hyps = [1.0, 0.5, 0.3, 0.2]
        >>> cutoffs = {"twobody": 7.0, "threebody": 3.5}
        """

        #if isinstance(kernel_list, str):
        #    if "+" in kernel_list:
        #        kernel_list = kernel_list.split("+")
        #    elif "plus" in kernel_list:
        #        kernel_list = kernel_list.split("plus")

        #    for k in range(len(kernel_list)):
        #        for kern_name in ["twobody", "threebody", "manybody"]:
        #            if kern_name in kernel_list[k]:
        #                kernel_list[k] = kern_name
        #                break
            
        assert len(hyps) == 2 * len(kernel_list) + 1
        assert len(cutoffs.keys()) >= len(kernel_list)

        kernels = []
        for k in range(len(kernel_list)):
            kern_name = kernel_list[k]
            if kern_name == "twobody":
                kern = TwoBodyKernel(hyps[2*k:2*k+2], cutoffs[kern_name])
            if kern_name == "threebody":
                kern = ThreeBodyKernel(hyps[2*k:2*k+2], cutoffs[kern_name])
            kernels.append(kern)

        self.kernels = kernels

    def energy_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        kern = 0.0
        for kernel in self.kernels:
            kern += kernel.energy_energy(env1, env2)
        return kern

    def force_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        kern = 0.0
        for kernel in self.kernels:
            kern += kernel.force_energy(env1, env2)
        return kern

    def stress_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        kern = 0.0
        for kernel in self.kernels:
            kern += kernel.stress_energy(env1, env2)
        return kern

    def force_force(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        kern = 0.0
        for kernel in self.kernels:
            kern += kernel.force_force(env1, env2)
        return kern

    def stress_force(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        kern = 0.0
        for kernel in self.kernels:
            kern += kernel.stress_force(env1, env2)
        return kern

    def stress_stress(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        kern = 0.0
        for kernel in self.kernels:
            kern += kernel.stress_stress(env1, env2)
        return kern

    def force_force_gradient(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        kern = 0.0
        kern_grad = np.zeros((len(self) * 2, 3, 3))
        n = 0
        for kernel in self.kernels:
            kern_and_grad = kernel.force_force_gradient(env1, env2)
            kern += kern_and_grad[0]
            kern_grad[2*n:2*n+2, :, :] += kern_and_grad[1]
            n += 1
        return kern, kern_grad

    def efs_energy(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        e_kern = 0.0
        f_kern = np.zeros(3)
        s_kern = np.zeros(6)
        for kernel in self.kernels:
            kern = kernel.efs_energy(env1, env2)
            e_kern += kern[0]
            f_kern += kern[1]
            s_kern += kern[2]
        return e_kern, f_kern, s_kern

    def efs_force(self, env1: AtomicEnvironment, env2: AtomicEnvironment):
        e_kern = np.zeros(3)
        f_kern = np.zeros((3, 3))
        s_kern = np.zeros((6, 3))
        for kernel in self.kernels:
            kern = kernel.efs_force(env1, env2)
            e_kern += kern[0]
            f_kern += kern[1]
            s_kern += kern[2]
        return e_kern, f_kern, s_kern

    def efs_self(self, env1: AtomicEnvironment):
        e_kern = 0.0
        f_kern = np.zeros(3)
        s_kern = np.zeros(6)
        for kernel in self.kernels:
            kern = kernel.efs_self(env1)
            e_kern += kern[0]
            f_kern += kern[1]
            s_kern += kern[2]
        return e_kern, f_kern, s_kern

    def __list__(self):
        kern_list = []
        for kernel in self.kernels:
            kern_list.append(kernel.__name__)
        return kern_list

    def __len__(self):
        return len(self.kernels)
