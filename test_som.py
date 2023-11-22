import torch
import som
import unittest
import itertools

# Testing by using an iterative implementation.
class IterativeSOM(object):

    def __init__(self, iterative, wrapping):
        self.iterative = iterative

        if wrapping == som.WrappingMode.FLAT:
            self.distance_fn = self.__flat_distance
        if wrapping == som.WrappingMode.TOROIDAL:
            self.use_sphere = False
            self.distance_fn = self.__spherical_distance
        if wrapping == som.WrappingMode.SPHERICAL:
            self.use_sphere = True
            self.distance_fn = self.__spherical_distance

        self.inter_model_distances = self.get_inter_model_distances()

    def get_activations_gauss(self, models, encoding):
        # The closer a vector is to the encoding, the higher the activation.
        activations = torch.zeros(len(models))
        for i,model in enumerate(models):
            sqr_dist = (model - encoding).square().sum(dim=-1)
            activations[i] = torch.exp(torch.neg(sqr_dist))
        return activations

    def get_activations_tanh(self,models, encoding):
        activations = torch.zeros(len(models))
        for i,model in enumerate(models):
            dist = (model - encoding).square().sum(dim=-1).sqrt()
            activations[i] = 1 - tanh(dist)
        return activations

    def __get_activations(self,models,encoding):
        if self.iterative.use_tanh:
            return get_activations_tanh(models,encoding)
        else:
            return get_activations_gauss(models,encoding)

    def get_activations(self,encoding):
        utils.assert_tensor_shape(encoding, (self.iterative.dim, ), "encoding")
        return self.__get_activations(self.iterative.models,encoding)

    def get_bmu(self, encoding):
        actvtn = self.get_activations(encoding)
        # Especially at the beginning of training, there may be a larger amount
        # of models that are equidistant to the encoding.
        bmu_idxs = (actvtn == torch.max(actvtn)).nonzero()
        # In order to prevent embedding collapse, we select one randomly as the bmu.
        selected = np.random.randint(low=0, high=len(bmu_idxs))
        return bmu_idxs[selected]

    def __spherical_distance(self, p1, p2):
        eye = [0, 0]
        flip_x = [self.iterative.width, 0]
        flip_y = [0, self.iterative.height]
        
        if self.use_sphere:
            flip_xy = [self.iterative.width, self.iterative.height]
            flip_yx = [-self.iterative.width, self.iterative.height]
        else:
            flip_xy = [0,0]
            flip_yx = [0,0]

        dist_all = []
        for f in [eye, flip_x, flip_y, flip_xy, flip_yx]:
            for sgn in [1, -1]:
                transform_p2 = p2 + sgn * torch.tensor(f)
                xy_dist = p1 - transform_p2
                d1 = torch.sqrt(torch.sum(torch.square(xy_dist), dim=-1))
                dist_all.append(d1)
                
        return min(dist_all)

    def __flat_distance(self, p1, p2):
        xy_dist = p1 - p2
        return torch.sqrt(torch.sum(torch.square(xy_dist), dim=-1))


    def get_inter_model_distances(self):
        distances = torch.empty((len(self.iterative.map_idx), len(self.iterative.map_idx)))
        for i, p1 in enumerate(self.iterative.map_idx):
            for j, p2 in enumerate(self.iterative.map_idx):
                distances[i][j] = self.distance_fn(p1, p2)
        return distances


    def mean_encoding_by_bmu(self, batch_encodings, bmus):
        sum_mj = torch.zeros(self.iterative.models.shape)
        count_mj = torch.zeros(self.iterative.models.shape[0])
        for i, v_idx in enumerate(bmus):
            count_mj[v_idx] += 1
            sum_mj[v_idx] += batch_encodings[i]

        x_mj = torch.zeros(self.iterative.models.shape)
        for i, sm in enumerate(sum_mj):
            if count_mj[i] > 0:
                x_mj[i] = sm / count_mj[i]
            else:
                x_mj[i] = torch.zeros(sm.shape)

        return x_mj

    def update_factor(self):
        uf = torch.empty((len(self.iterative.map_idx), len(self.iterative.map_idx)))
        for i, p1 in enumerate(self.iterative.map_idx):
            for j, p2 in enumerate(self.iterative.map_idx):
                d = self.distance_fn(p1, p2)
                uf[i][j] = torch.exp(
                    torch.neg(torch.div(d.square(), 2 * self.iterative.gauss**2)))
        return uf

    def batch_update_step(self, batch_encodings, bmus):
        h_ij = self.update_factor()
        x_mj = self.mean_encoding_by_bmu(batch_encodings, bmus)

        bmu_count_by_idx = torch.zeros(self.iterative.models.shape[0])
        for i, v_idx in enumerate(bmus):
            bmu_count_by_idx[v_idx] += 1

        bse = torch.zeros(self.iterative.models.shape)
        for i in range(len(h_ij)):
            denom = torch.zeros(bmu_count_by_idx.shape)
            for j in range(len(h_ij)):
                bse[i] += x_mj[j] * h_ij[i][j] * bmu_count_by_idx[j]
                denom[i] += h_ij[i][j] * bmu_count_by_idx[j]

            if denom[i] > 0:
                bse[i] = bse[i] / denom[i]
            else:
                bse[i] = torch.zeros(bse[i].shape)

        return bse


class TestBatchSOM(unittest.TestCase):

    def __make_test(self, trial):
        batch_som = som.SOM((3, 3, 10), trial["wrapping"], gauss=1, use_tanh=trial["use_tanh"])
        iterative = IterativeSOM(batch_som, trial["wrapping"])
        batch_encodings = torch.randint(high=10, size=(60, 10)).float()
        # bmus are selected randomly, so we inject them here
        bmus = torch.cat([batch_som.get_bmu(e) for e in batch_encodings])
        return batch_som, iterative, batch_encodings, bmus

    def get_all_trials(self):
        trial_values = {"wrapping":list(som.WrappingMode),"use_tanh":[True,False]}
        keys, values = zip(*trial_values.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def test_inter_model_distances(self):
        for trial in self.get_all_trials():
            with self.subTest(trial=trial):
                batch_som, iterative, batch_encodings, bmus = self.__make_test(
                    trial)

                imd = batch_som.inter_model_distances
                imd_expected = iterative.inter_model_distances
                self.assertTrue(torch.isclose(imd, imd_expected).all())

    def test_update_factor(self):
        for trial in self.get_all_trials():
            with self.subTest(trial=trial):
                batch_som, iterative, batch_encodings, bmus = self.__make_test(
                    trial)

                h_ij = batch_som.update_factor()
                h_ij_expected = iterative.update_factor()
                self.assertTrue(torch.isclose(h_ij, h_ij_expected).all())

    def test_update_factor_symmetric(self):
        for trial in self.get_all_trials():
            with self.subTest(trial=trial):
                batch_som, iterative, batch_encodings, bmus = self.__make_test(
                    trial)

                h_ij = batch_som.update_factor()
                h_ij_expected = iterative.update_factor()
                for i in range(len(h_ij_expected)):
                    for j in range(len(h_ij_expected)):
                        self.assertEqual(h_ij_expected[i][j],
                                         h_ij_expected[j][i])

    def test_batch_update_step(self):
        for trial in self.get_all_trials():
            with self.subTest(trial=trial):
                batch_som, iterative, batch_encodings, bmus = self.__make_test(
                    trial)

                bse = batch_som.batch_update_step(batch_encodings, bmus)
                bse_expected = iterative.batch_update_step(
                    batch_encodings, bmus)
                self.assertTrue(torch.isclose(bse, bse_expected).all())


if __name__ == '__main__':
    unittest.main()
