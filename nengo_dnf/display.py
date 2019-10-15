import numpy as np
import nengo

class Display1D(object):
    def __init__(self, size, n_lines, range=(-1,1)):
        self.size = size
        self.n_lines = n_lines
        self.range = range

        self.svg_x = np.linspace(0, 100, size)
        self.template = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            %s
            <line x1=50 y1=0 x2=50 y2=100 stroke="#aaaaaa"/>
            <line x1=0 y1=50 x2=100 y2=50 stroke="#aaaaaa"/>
            </svg>'''

        self.palette = ["#1c73b3", "#039f74", "#d65e00",
                        "#cd79a7", "#f0e542", "#56b4ea"]

    def make_node(self):
        def plot(t, x):
            y = np.interp(x, self.range, (0,100))
            y = y.reshape((self.n_lines, self.size))
            paths = []
            for i, yy in enumerate(y):
                path = []
                for j, d in enumerate(yy):
                    path.append('%1.0f %1.0f' % (self.svg_x[j], d))
                paths.append('<path d="M%s" fill="none" stroke="%s"/>' %
                             ('L'.join(path), self.palette[i % len(self.palette)]))
            plot._nengo_html_ = self.template % (''.join(paths))

        return nengo.Node(plot, size_in=self.size*self.n_lines)
