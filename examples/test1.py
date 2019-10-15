import nengo_dnf
import importlib
importlib.reload(nengo_dnf.dnf)
importlib.reload(nengo_dnf)

import nengo
import numpy as np

class DNF(nengo.Network):
    def __init__(self, dimensions, domain=1, resolution=100,
                 h=0.1, tau=0.01, c_glob=2.0, dt=0.001,
                 
                 kernel=nengo_dnf.Kernel(c_exc=2.25, c_inh=0.7,
                                         sigma_exc=0.1, sigma_inh=1.0),
                 nonlinearity=nengo_dnf.AbsSigmoid(beta=100),
                 label=None, add_to_container=True
                ):
        super().__init__(label=label, add_to_container=add_to_container)
        sizes = np.zeros(dimensions, dtype=int)
        sizes[:] = resolution
        dx = np.zeros(dimensions, dtype=float)
        dx[:] = domain / resolution
        self.nf = nengo_dnf.NeuralField(sizes=sizes, h=h, tau=tau/dt,
                                         c_glob=c_glob,
                                         kernel=kernel, dx=dx,
                                         nonlinearity=nonlinearity)
        
        with self:
            N=np.product(self.nf.sizes)
            self.input = nengo.Node(None, size_in=N)
            self.output = nengo.Node(None, size_in=N)
            self.dnf = nengo.Node(self.update, size_in=N, size_out=N)
            nengo.Connection(self.input, self.dnf, synapse=None)
            nengo.Connection(self.dnf, self.output, synapse=None)

    def update(self, t, x):
        return self.nf.update(x.reshape(self.nf.sizes)).flatten()
        
    
    def make_display(self, range=10):
        svg_x = np.linspace(0, 100, self.nf.sizes[0])
        template = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
                <line x1=50 y1=0 x2=50 y2=100 stroke="#dddddd"/>
                <line x1=0 y1=50 x2=100 y2=50 stroke="#dddddd"/>
                %s
            </svg>'''

        palette = ["#1c73b3", "#039f74", "#d65e00",
                   "#cd79a7", "#f0e542", "#56b4ea"]
        def plot(t):
            v = 100-np.interp(self.nf.u, [-range, range], [0, 100])
            out = 100-np.interp(self.nf.nonlinearity(self.nf.u), [-range, range], [0, 100])
            data = [v, out]            

            paths = []
            for i, row in enumerate(data):
                path = []
                for j, d in enumerate(row):
                    path.append('%1.0f %1.0f' % (svg_x[j], d))
                paths.append('<path d="M%s" fill="none" stroke="%s"/>' %
                             ('L'.join(path),
                              palette[i % len(palette)]))

            plot._nengo_html_ = template % (''.join(paths))
        with self:
            self.display = nengo.Node(plot)
    
    
class StimDNF(nengo.Network):
    def __init__(self, dimensions, domain=1, resolution=100):
        super().__init__()
        with self:
            self.resolution = 100
            self.domain = 1
            self.x = np.linspace(-domain, domain, resolution)
            self.mean = nengo.Node(0)
            self.sigma = nengo.Node(0.1)
            self.amp = nengo.Node(8)
            self.output = nengo.Node(self.update, size_in=3)
            nengo.Connection(self.mean, self.output[0], synapse=None)
            nengo.Connection(self.sigma, self.output[1], synapse=None)
            nengo.Connection(self.amp, self.output[2], synapse=None)
            
    def update(self, t, x):
        mean, sigma, amp = x

        v = amp*np.exp(-(self.x-mean)**2/(2*sigma**2))
        return v
        

model = nengo.Network()
with model:
        
    stim1 = StimDNF(dimensions=1)
    stim2 = StimDNF(dimensions=1)
    
    dnf = DNF(dimensions=1, c_glob=10, h=-1)
    dnf.make_display()
    nengo.Connection(stim1.output, dnf.input)
    #nengo.Connection(stim2.output, dnf.input)    