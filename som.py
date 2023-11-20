import torch
import numpy as np


def flatten(mtrx):
    return mtrx.reshape((mtrx.shape[0] * mtrx.shape[1], -1)).squeeze()


def get_idx_grid(width, height, step):
    x_idx, y_idx = torch.meshgrid(torch.arange(start=0,
                                               end=width - 1 + step,
                                               step=step),
                                  torch.arange(start=0,
                                               end=height - 1 + step,
                                               step=step),
                                  indexing='ij')
    grid_idx = torch.stack([x_idx, y_idx], dim=-1)
    return flatten(grid_idx)


class SOM(object):

    def __init__(self,
                 width,
                 height,
                 dim,
                 is_cyclic,
                 gauss=10,
                 decay=.99,
                 use_onehot=True):
        if use_onehot:
            # Select a random index to use as the hot element.
            idxs = torch.randint(low=0, high=dim, size=(width, height))
            # Convert to one hot of shape.
            self.models = torch.nn.functional.one_hot(idxs,
                                                      num_classes=dim).float()
        else:
            self.models = torch.nn.functional.normalize(
                torch.rand(size=(width, height, dim)).float(), dim=-1)

        self.gauss = gauss
        self.decay = decay
        self.width = width
        self.height = height

        self.models = flatten(self.models)
        self.map_idx = get_idx_grid(width, height, 1)

        if is_cyclic:
            self.inter_model_distances = self.__get_cyclic_distances(
                width, height)
        else:
            self.inter_model_distances = self.__get_flat_distances()

    def do_decay(self):
        # From the author:
        # For large SOMs, the final value of σ may be on the order of five per cent of the shorter side of the grid.
        # For smaller ones, idk.
        min_gauss = min(self.width, self.height) * .05
        self.gauss *= self.decay
        self.gauss = max(self.gauss, min_gauss)

    def get_activations(self, encoding):
        # Activation is 1 / Euclidian(models, encoding).
        # The closer a vector is to the encoding, the higher the activation.
        return 1 / (self.models - encoding).square().sum(dim=-1).sqrt()

    def get_enc_dists(self, encoding):
        # Activation is 1 / Euclidian(models, encoding).
        # The closer a vector is to the encoding, the higher the activation.
        return (self.models - encoding).square().sum(dim=-1).sqrt()

    def get_bmu(self, encoding):
        actvtn = self.get_activations(encoding)
        # Especially at the beginning of training, there may be a larger amount
        # of models that are equidistant to the encoding.
        bmu_idxs = (actvtn == torch.max(actvtn)).nonzero()
        # In order to prevent embedding collapse, we select one randomly as the bmu.
        selected = np.random.randint(low=0, high=len(bmu_idxs))
        # print(bmu_idxs,bmu_idxs[selected])
        return bmu_idxs[selected]

    def get_bmu_pos(self, encoding):
        return self.map_idx[self.get_bmu(encoding)]

    def mean_encoding_by_bmu(self, encodings, bmus):
        # https://stackoverflow.com/questions/56154604/groupby-aggregate-mean-in-pytorch/56155805#56155805
        M = torch.zeros(len(self.models), len(encodings))
        M[bmus, torch.arange(len(encodings))] = 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        return torch.mm(M, encodings)

    def __get_flat_distances(self):
        # Distance from each node to every other node
        xy_dist = self.map_idx.unsqueeze(0) - self.map_idx.unsqueeze(1)
        return torch.sqrt(torch.sum(torch.square(xy_dist), dim=-1))

    def __get_cyclic_distances(self, w, h):
        # This is only computed once, so we could cache it if we wanted.
        # Distance from each node to every other node
        eye = [0, 0]
        flip_x = [w, 0]
        flip_y = [0, h]
        # Results look better when map is on a torus, if we wanted a sphere this should be [w,h]
        flip_xy = eye
        dist_all = []
        for f in [eye, flip_x, flip_y, flip_xy]:
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

    def batch_sum_encodings(self, bmus, encodings):
        # Although this is referred to as h_ji in the paper
        # it is symmetric (so h[j][i] == h[i][j])
        h_ij = self.update_factor()
        x_mj = self.mean_encoding_by_bmu(encodings, bmus)

        bmu_count_by_idx = torch.bincount(bmus, minlength=len(self.map_idx))
        # Unsqueeze the first dimension of the counts so that the update factor
        # for i to j is weighted based on occurences of j.
        weighted_h_ji = bmu_count_by_idx.unsqueeze(0) * h_ij

        return torch.mm(weighted_h_ji, x_mj) / weighted_h_ji.sum(dim=-1,
                                                                 keepdim=True)

    def update_batch(self, encodings):
        # This step is not vectorized, but we could do a random partitioning or something above.
        bmus = torch.cat([self.get_bmu(e) for e in encodings])
        self.models = torch.nn.functional.normalize(self.batch_sum_encodings(
            bmus, encodings),
                                                    dim=-1)


def test():
    mm = SOM(3, 2, 10, True)
    encodings = torch.randint(high=10, size=(60, 10)).float()
    mm.update_batch(encodings)


if __name__ == "__main__":
    test()
