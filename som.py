import torch
import numpy as np
import scipy
import utils
import enum

WrappingMode = enum.Enum('WrappingMode', ['FLAT', 'TOROIDAL', 'SPHERICAL'])

class SOM(object):

    def __init__(self, shape, wrapping, gauss=10, decay=.99, use_onehot=True, clip_models=True, use_tanh=False):

        if not hasattr(shape, '__len__') or len(shape) != 3:
            raise TypeError(
                f"Expected a shape of (width, height, dimension) but got {shape}"
            )

        self.shape = shape
        self.width, self.height, self.dim = shape

        if use_onehot:
            # Select a random index to use as the hot element.
            idxs = torch.randint(low=0,
                                 high=self.dim,
                                 size=(self.width, self.height))
            # Convert to one hot of shape.
            self.models = torch.nn.functional.one_hot(
                idxs, num_classes=self.dim).float()
        else:
            self.models = torch.rand(size=(self.width, self.height, self.dim)).float()

        self.gauss = gauss
        self.decay = decay
        self.clip_models = clip_models

        self.models = utils.flatten(self.models)
        self.map_idx, _ = utils.get_idx_grid(self.width, self.height, 1)

        self.wrapping = wrapping
        self.inter_model_distances = self.__get_inter_model_distances()

        if self.clip_models:
            self.models = torch.nn.functional.normalize(self.models,dim=-1)

        self.use_tanh = use_tanh

    def do_decay(self):
        # From the author:
        # For large SOMs, the final value of σ may be on the order of five per cent of the shorter side of the grid.
        # For smaller ones, idk.
        min_gauss = min(self.width, self.height) * .05
        self.gauss *= self.decay
        self.gauss = max(self.gauss, min_gauss)

    def __get_activations_gauss(self,models,encoding):
        # Using a gaussian distribution so that dist of 0 has
        # activation of 1, and all activations are > 0.
        sqr_dists = (models - encoding).square().sum(dim=-1)
        return torch.exp(torch.neg(sqr_dists))

    def __get_activations_tanh(self,models,encoding):
        # Tanh function where x = 0 has y = 1 and approaches 0 asymptotically.
        dists = (models - encoding).square().sum(dim=-1).sqrt()
        # There will never be a negative distance.
        return 1 - torch.tanh(dists)

    def __get_activations(self,models,encoding):
        if self.use_tanh:
            return self.__get_activations_tanh(models,encoding)
        else:
            return self.__get_activations_gauss(models,encoding)

    def get_activations(self,encoding):
        utils.assert_tensor_shape(encoding, (self.dim, ), "encoding")
        return self.__get_activations(self.models,encoding)

    def __get_bmu(self,actvtn):
        # Especially at the beginning of training, there may be a larger amount
        # of models that are equidistant to the encoding.
        bmu_idxs = (actvtn == torch.max(actvtn)).nonzero()
        # In order to prevent embedding collapse, we select one candidate BMU randomly.
        selected = np.random.randint(low=0, high=len(bmu_idxs))
        return bmu_idxs[selected]

    def get_bmu(self, encoding):
        utils.assert_tensor_shape(encoding, (self.dim, ), "encoding")

        actvtn = self.get_activations(encoding)
        return self.__get_bmu(actvtn)

    def get_bmu_pos(self, encoding):
        utils.assert_tensor_shape(encoding, (self.dim, ), "encoding")

        return self.map_idx[self.get_bmu(encoding)]

    def __get_cyclic_grid(self):
        eye = [0, 0]
        flip_x = [self.width, 0]
        flip_y = [0, self.height]
        
        if self.wrapping == WrappingMode.SPHERICAL:
            flip_xy = [self.width, self.height]
            flip_yx = [-self.width, self.height]
        else:
            flip_xy = [0,0]
            flip_yx = [0,0]
        dist_all = []
        transform_idxs = []
        transform_models = []
        # We could probably vectorize this efficiently
        for f in [eye, flip_x, flip_y, flip_xy, flip_yx]:
            for sgn in [1, -1]:
                transform_idxs.append(self.map_idx + sgn * torch.tensor(f))
                transform_models.append(self.models)
        
        return torch.stack(transform_idxs), torch.stack(transform_models)

    def __get_wrapped_grid(self):
        if self.wrapping == WrappingMode.FLAT:
            return self.map_idx.unsqueeze(0), self.models.unsqueeze(0)
        else:
            return self.__get_cyclic_grid()

    def get_interpolated_activations(self, encoding, step, method="linear"):
        utils.assert_tensor_shape(encoding, (self.dim, ), "encoding")

        # For interpolation, we need to take into account the wrapped positions of the models
        wrapped_idx, wrapped_models = self.__get_wrapped_grid()
        wrapped_idx = utils.flatten(wrapped_idx)
        wrapped_models = utils.flatten(wrapped_models)
        fine_grid, fine_shape = utils.get_idx_grid(self.width, self.height, step)
        # This step is computationally expensive
        # and only needs to be done once for each time the models are updated
        # We could cache the interpolated models, by step size.
        fine_models = torch.FloatTensor(scipy.interpolate.griddata(wrapped_idx.numpy(),
                                              wrapped_models.numpy(),
                                              fine_grid.numpy(),
                                              method=method))
        
        assert torch.all(torch.isclose(fine_models[0], self.models[0]))
        assert torch.all(fine_grid[0] == self.map_idx[0])


        assert torch.all(torch.isclose(fine_models[-1], self.models[-1]))
        assert torch.all(fine_grid[-1] == self.map_idx[-1])

        fine_act = self.__get_activations(fine_models,encoding)

        fine_bmu_pos = fine_grid[self.__get_bmu(fine_act)].squeeze()

        return fine_grid, fine_act, fine_shape, fine_bmu_pos

    def __mean_encoding_by_bmu(self, batch_encodings, bmus):
        utils.assert_tensor_shape(batch_encodings,
                                  (batch_encodings.shape[0], self.dim),
                                  "batch_encodings")

        # https://stackoverflow.com/questions/56154604/groupby-aggregate-mean-in-pytorch/56155805#56155805
        M = torch.zeros(len(self.models), len(batch_encodings))
        M[bmus, torch.arange(len(batch_encodings))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        return torch.mm(M, batch_encodings)

    def __get_inter_model_distances(self):
        wrapped_idx, wrapped_models = self.__get_wrapped_grid()
        xy_dist = self.map_idx.unsqueeze(0) - wrapped_idx.unsqueeze(2)
        dists = torch.sqrt(torch.sum(torch.square(xy_dist), dim=-1))
        return dists.amin(dim=0)

    def update_factor(self):
        return torch.exp(
            torch.neg(
                torch.div(self.inter_model_distances.square(),
                          2 * self.gauss**2)))

    def batch_update_step(self, batch_encodings, bmus):
        # Although this is referred to as h_ji in the paper
        # it is symmetric (so self.height[j][i] == self.height[i][j])
        h_ij = self.update_factor()
        x_mj = self.__mean_encoding_by_bmu(batch_encodings, bmus)

        bmu_count_by_idx = torch.bincount(bmus, minlength=len(self.map_idx))
        # Unsqueeze the first dimension of the counts so that the update factor
        # for i to j is weighted based on occurences of j.
        weighted_h_ji = bmu_count_by_idx.unsqueeze(0) * h_ij

        return torch.mm(weighted_h_ji, x_mj) / weighted_h_ji.sum(dim=-1,
                                                                 keepdim=True)

    def update_batch(self, batch_encodings):
        utils.assert_tensor_shape(batch_encodings,
                                  (batch_encodings.shape[0], self.dim),
                                  "batch_encodings")

        # This step is not vectorized, but we could do a random partitioning or something above.
        bmus = torch.cat([self.get_bmu(e) for e in batch_encodings])
        updated_models = self.batch_update_step(batch_encodings, bmus)
        self.models = updated_models
        if self.clip_models:
            self.models = torch.nn.functional.normalize(self.models,dim=-1)

def test():
    mm = SOM((3, 2, 10), True)
    batch_encodings = torch.randint(high=10, size=(60, 10)).float()
    mm.update_batch(batch_encodings)
    act = mm.get_activations(batch_encodings[0])
    igrid,iact,ishape,fbpos = mm.get_interpolated_activations(batch_encodings[0],.01)
    print(igrid.shape,iact.shape,ishape,fbpos)

if __name__ == "__main__":
    test()
