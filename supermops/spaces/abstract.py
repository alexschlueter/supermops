import numpy as np

class ShapedSpace:
    def __init__(self, shape):
        self.shape = shape

    def total_dofs(self):
        return np.prod(self.shape)

    def reshape(self, array):
        return array.reshape(self.shape)

class RegularProductSpace(ShapedSpace):
    def __init__(self, num_components, component_shape):
        self.num_components = num_components
        super().__init__((num_components, *component_shape))

    def __iter__(self):
        for idx in range(self.num_components):
            yield self[idx]

class FiniteElementSpace:
    def __iter__(self):
        for idx in np.ndindex(*self.grid_sizes()):
            yield self[idx]

    def ravel_index(self, idx):
        return np.ravel_multi_index(idx, self.grid_sizes())

    def unravel_index(self, idx):
        return np.unravel_index(idx, self.grid_sizes())

    def mpl_draw(self, ax, *args, **kwargs):
        for cell in self:
            cell.mpl_draw(ax, *args, **kwargs)

    def get_center_grid(self):
        grid = []
        for cell in self:
            grid.append(cell.midpoint())
        return np.asarray(grid)