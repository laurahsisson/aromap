import torch
import numpy as np
import scipy
import utils
import enum

WrappingMode = enum.Enum('WrappingMode', ['FLAT', 'TOROIDAL', 'SPHERICAL'])

class SOM(object):

    def __init__(self, shape, wrapping, gauss=10, decay=.99, use_onehot=True):

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
            self.models = torch.nn.functional.normalize(
                torch.rand(size=(self.width, self.height, self.dim)).float(),
                dim=-1)

        self.gauss = gauss
        self.decay = decay

        self.models = utils.flatten(self.models)
        self.map_idx = utils.get_idx_grid(self.width, self.height, 1)

        if wrapping == WrappingMode.FLAT:
            self.inter_model_distances = self.__get_flat_distances()
        else:
            self.inter_model_distances = self.__get_cyclic_distances(wrapping)

    def do_decay(self):
        # From the author:
        # For large SOMs, the final value of σ may be on the order of five per cent of the shorter side of the grid.
        # For smaller ones, idk.
        min_gauss = min(self.width, self.height) * .05
        self.gauss *= self.decay
        self.gauss = max(self.gauss, min_gauss)

    def get_activations(self, encoding):
        utils.assert_tensor_shape(encoding, (self.dim, ), "encoding")

        # Activation is 1 / Euclidian(models, encoding).
        # The closer a vector is to the encoding, the higher the activation.
        return 1 / (self.models - encoding).square().sum(dim=-1).sqrt()

    def get_enc_dists(self, encoding):
        utils.assert_tensor_shape(encoding, (self.dim, ), "encoding")

        # Activation is 1 / Euclidian(models, encoding).
        # The closer a vector is to the encoding, the higher the activation.
        return (self.models - encoding).square().sum(dim=-1).sqrt()

    def get_bmu(self, encoding):
        utils.assert_tensor_shape(encoding, (self.dim, ), "encoding")

        actvtn = self.get_activations(encoding)
        # Especially at the beginning of training, there may be a larger amount
        # of models that are equidistant to the encoding.
        bmu_idxs = (actvtn == torch.max(actvtn)).nonzero()
        # In order to prevent embedding collapse, we select one randomly as the bmu.
        selected = np.random.randint(low=0, high=len(bmu_idxs))
        # print(bmu_idxs,bmu_idxs[selected])
        return bmu_idxs[selected]

    def get_bmu_pos(self, encoding):
        utils.assert_tensor_shape(encoding, (self.dim, ), "encoding")

        return self.map_idx[self.get_bmu(encoding)]

    def mean_encoding_by_bmu(self, batch_encodings, bmus):
        utils.assert_tensor_shape(batch_encodings,
                                  (batch_encodings.shape[0], self.dim),
                                  "batch_encodings")

        # https://stackoverflow.com/questions/56154604/groupby-aggregate-mean-in-pytorch/56155805#56155805
        M = torch.zeros(len(self.models), len(batch_encodings))
        M[bmus, torch.arange(len(batch_encodings))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        return torch.mm(M, batch_encodings)

    def __get_flat_distances(self):
        # Distance from each node to every other node
        xy_dist = self.map_idx.unsqueeze(0) - self.map_idx.unsqueeze(1)
        return torch.sqrt(torch.sum(torch.square(xy_dist), dim=-1))

    def __get_cyclic_distances(self,wrapping):
        # This is only computed once, so we could cache it if we wanted.
        # Distance from each node to every other node
        eye = [0, 0]
        flip_x = [self.width, 0]
        flip_y = [0, self.height]
        
        if wrapping == WrappingMode.SPHERICAL:
            flip_xy = [self.width, self.height]
            flip_yx = [-self.width, self.height]
        else:
            flip_xy = [0,0]
            flip_yx = [0,0]
        dist_all = []
        for f in [eye, flip_x, flip_y, flip_xy, flip_yx]:
            for sgn in [1, -1]:
                transform_idx = self.map_idx + sgn * torch.tensor(f)
                xy_dist = self.map_idx.unsqueeze(0) - transform_idx.unsqueeze(
                    1)
                dist_all.append(
                    torch.sqrt(torch.sum(torch.square(xy_dist), dim=-1)))

        return torch.stack(dist_all).amin(0)

    def update_factor(self):
        return torch.exp(
            torch.neg(
                torch.div(self.inter_model_distances.square(),
                          2 * self.gauss**2)))

    def batch_update_step(self, batch_encodings, bmus):
        # Although this is referred to as h_ji in the paper
        # it is symmetric (so self.height[j][i] == self.height[i][j])
        h_ij = self.update_factor()
        x_mj = self.mean_encoding_by_bmu(batch_encodings, bmus)

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
        self.models = torch.nn.functional.normalize(updated_models, dim=-1)

    def interpolate_activation(self, step, encoding, method="linear"):
        utils.assert_tensor_shape(encoding, (self.dim, ), "encoding")

        activations = self.get_activations(encoding)
        fine_grid = utils.get_idx_grid(self.width, self.height, step)
        assert len(activations.shape) == 1
        fine_act = scipy.interpolate.griddata(self.map_idx.numpy(),
                                              activations.numpy(),
                                              fine_grid.numpy(),
                                              method=method)
        assert fine_act[0] == activations[0]
        assert torch.all(fine_grid[0] == self.map_idx[0])


        assert fine_act[-1] == activations[-1]
        assert torch.all(fine_grid[-1] == self.map_idx[-1])

        return fine_grid, torch.FloatTensor(fine_act)


def test():
    mm = SOM((20, 16, 10), True)
    batch_encodings = torch.randint(high=10, size=(60, 10)).float()
    mm.update_batch(batch_encodings)
    mm.interpolate_activation(.01, batch_encodings[0])


if __name__ == "__main__":
    test()
