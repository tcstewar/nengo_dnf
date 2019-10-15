import numpy as np
import scipy.signal
import nengo

class AbsSigmoid(object):
    def __init__(self, beta=100):
        self.beta = beta
    def __call__(self, x):
        return 0.5 * (1 + self.beta * x) / (1 + self.beta * np.abs(x)) + 0.5


class Kernel(object):
    def __init__(self, c_exc, c_inh, sigma_exc, sigma_inh):
        self.c_exc = c_exc
        self.c_inh = c_inh
        self.sigma_exc = sigma_exc
        self.sigma_inh = sigma_inh
    def __call__(self, dx):
        dx2 = dx**2
        return (self.c_exc*np.exp(-dx2/(2*self.sigma_exc**2))-
                self.c_inh*np.exp(-dx2/(2*self.sigma_inh**2)))


class NeuralField(object):
    def __init__(self, sizes, h, tau, kernel,
                 c_glob=1, dx=1,
                 nonlinearity=AbsSigmoid(beta=100)):
        self.u = np.zeros(sizes)
        self.h = h
        self.tau = tau
        self.sizes = sizes
        self.c_glob = c_glob
        self.nonlinearity = nonlinearity
        self.kernel = kernel
        self.dx = dx

        self.init()

    def init(self):
        if len(self.sizes) == 1:
            self.init_recurrent_1d()
        elif len(self.sizes) == 2:
            self.init_recurrent_2d()
        else:
            raise Exception('Unhandled number of dimensions: %d' % len(sizes))

    def update(self, stim):
        a = self.nonlinearity(self.u)
        recurr = self.recurrent(a)
        self.u += (-self.u + self.h + self.c_glob*recurr + stim)/self.tau
        return self.u
        
    def init_recurrent_2d(self):
        x = np.arange(self.sizes[0])*self.dx[0]
        y = np.arange(self.sizes[1])*self.dx[1]
        grid_x, grid_y = np.meshgrid(x, y)
        cx = self.sizes[0]//2
        cy = self.sizes[1]//2
        dx = np.sqrt((grid_x-cx)**2 + (grid_y-cy)**2)
        self.kernel_matrix = self.kernel(dx)

        self.recurrent = lambda a: scipy.signal.convolve2d(a, 
                                       self.kernel_matrix,
                                       boundary='fill', mode='same')

    def init_recurrent_1d(self):
        x = np.arange(self.sizes[0])*self.dx[0]
        cx = self.sizes[0]//2
        dx = np.abs(x-cx)
        self.kernel_vector = self.kernel(dx)
        self.recurrent = lambda a: scipy.signal.convolve(a,
                                     self.kernel_vector,
                                     mode='same')
    
