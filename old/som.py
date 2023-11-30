import torch
import numpy as np
import scipy
import utils
import enum

WrappingMode = enum.Enum('WrappingMode', ['FLAT', 'TOROIDAL', 'SPHERICAL'])
ActivationMode = enum.Enum('ActivationMode', ['INVERSE_DISTANCE', 'SIMILARITY'])

class SOM(object):

    def __init__(self, shape, wrapping, activation, gauss=10, decay=.99):

        if not hasattr(shape, '__len__') or len(shape) != 3:
            raise TypeError(
                f"Expected a shape of (width, height, dimension) but got {shape}"
            )

        self.shape = shape
        self.width, self.height, self.dim = shape

        self.activation = activation
        self.models = self.__init_models()

        self.gauss = gauss
        self.decay = decay

        self.models = utils.flatten(self.models)
        self.map_idx, _ = utils.get_idx_grid(self.width, self.height, 1)

        self.wrapping = wrapping
        self.inter_model_distances = self.__get_inter_model_distances()


    def do_decay(self):
        # From the author:
        # For large SOMs, the final value of σ may be on the order of five per cent of the shorter side of the grid.
        # For smaller ones, idk.
        min_gauss = min(self.width, self.height) * .05
        self.gauss *= self.decay
        self.gauss = max(self.gauss, min_gauss)

    def __init_models(self):
        if self.activation == ActivationMode.INVERSE_DISTANCE:
            return torch.rand(size=(self.width, self.height, self.dim)).float()
        else:
            # Select a random index to use as the one-hot element.
            idxs = torch.randint(low=0,
                                 high=self.dim,
                                 size=(self.width, self.height))
            # Convert to one hot of shape.
            return torch.nn.functional.one_hot(
                idxs, num_classes=self.dim).float()

    def __cosine_similarity_matrix(self,m1, m2):
        # Normalize the rows of m1 and m2
        m1_normalized = m1 / torch.norm(m1, dim=1, keepdim=True)
        m2_normalized = m2 / torch.norm(m2, dim=1, keepdim=True)

        # Compute the dot product between each pair of normalized rows
        similarity_matrix = torch.matmul(m1_normalized, m2_normalized.t())

        return similarity_matrix

    def __get_batch_activations(self,models,batch_encodings):
        if self.activation == ActivationMode.INVERSE_DISTANCE:
            return 1 / torch.cdist(batch_encodings,models)
        else: 
            return self.__cosine_similarity_matrix(batch_encodings,models)

    def __get_activations(self,models,encoding):
        utils.assert_tensor_shape(encoding, (self.dim, ), "encoding")
        return self.__get_batch_activations(models,encoding.unsqueeze(0)).squeeze()

    def get_activations(self,encoding):
        return self.__get_activations(self.models,encoding)

    def __get_kmu_batch(self,batch_activations,k=1):
        # Because early in traing, there may be multiple best matching units
        # and also different activations may match to a single bmu.
        # To resolve this, we permute the indices of the activations (and thus matching units)
        # row by row. This ensures that the activations do not all use the same bmu.
        # Assuming batch_acivations.shape is consistent, we can do this once ahead of time.
        permuted_indices = torch.stack([torch.randperm(batch_activations.shape[1]) for _ in range(batch_activations.shape[0])])
        # Permute the activations using the new indices
        permuted_activations = torch.gather(batch_activations,1,permuted_indices)
        # Find the topk activations for each row
        _, top_1tok_indices = torch.topk(permuted_activations,k)
        # We want to find exactly the kth (assuming 1 based indexing) element.
        k_indices = top_1tok_indices[:,k-1]
        # Return the original indices from the permuted indices.
        return permuted_indices[torch.arange(permuted_indices.size(0)), k_indices]

    def get_bmu(self, encoding):
        utils.assert_tensor_shape(encoding, (self.dim, ), "encoding")

        batch_activations = self.__get_batch_activations(self.models,encoding.unsqueeze(0))
        return self.__get_kmu_batch(batch_activations,k=1).squeeze()

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
        # So the expensive grid wrapping operation needs to be recalculated.
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

        fine_bmu_pos = fine_grid[self.__get_kmu_batch(fine_act.unsqueeze(0),k=1)].squeeze()

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

        bmu_models = self.models[bmus]
        return torch.sum(1-self.__get_batch_activations(bmu_models, batch_encodings))

    def __quantization_loss(self,batch_encodings,bmus):
        bmu_models = self.models[bmus]
        # No helper function that can calculate an nx1 activation
        if self.activation == ActivationMode.INVERSE_DISTANCE:
            batch_bmu_activations = torch.sqrt(torch.sum(torch.square(bmu_models - batch_encodings), dim=-1))
        else:
            batch_bmu_activations = 1-torch.nn.functional.cosine_similarity(bmu_models,batch_encodings)
        # Quantization loss is the inverse of the activation for each encoding
        # and its best matching unit.
        utils.assert_tensor_shape(batch_bmu_activations, bmus.shape, "quantization loss")
        return torch.sum(batch_bmu_activations)

    def __topographic_loss(self,batch_activations,bmus):
        # Get the index of second best matching unit
        next_matching_units = self.__get_kmu_batch(batch_activations,k=2)
        # Index into our precalculated model distance matrix
        bmu_nmu_distance = self.inter_model_distances[bmus,next_matching_units]
        # Topographic loss is sum of all bmu to nmu distances
        utils.assert_tensor_shape(bmu_nmu_distance, bmus.shape, "topographic loss")
        return torch.sum(bmu_nmu_distance)

    def loss(self,batch_encodings):
        batch_activations = self.__get_batch_activations(self.models,batch_encodings)
        bmus = self.__get_kmu_batch(batch_activations,k=1)
        ql = self.__quantization_loss(batch_encodings,bmus)
        tl = self.__topographic_loss(batch_activations,bmus)
        return {"loss":ql+tl,"quantization":ql,"topographic":tl}

    def update_batch(self, batch_encodings):
        batch_activations = self.__get_batch_activations(self.models,batch_encodings)
        bmus = self.__get_kmu_batch(batch_activations,k=1)
        updated_models = self.batch_update_step(batch_encodings, bmus)
        self.models = updated_models

def test():
    mm = SOM((3, 2, 10), WrappingMode.FLAT,ActivationMode.SIMILARITY)
    batch_encodings = torch.randint(high=10, size=(2, 10)).float()
    mm.update_batch(batch_encodings)
    act = mm.get_activations(batch_encodings[0])
    igrid,iact,ishape,fbpos = mm.get_interpolated_activations(batch_encodings[0],.01)
    print(f"Interpolated Grid Shape: {igrid.shape} \nInterpolated Activation Shape: {iact.shape} \nInterpolated Matrix Shape: {ishape} \nInterpolated BMU Position: {fbpos}")
    print(mm.loss(batch_encodings))

if __name__ == "__main__":
    test()
