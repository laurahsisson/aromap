import torch
import som
import unittest


# Testing by using an iterative implementation.
class IterativeSOM(object):

    def __init__(self, iterative, wrapping):
        self.models = iterative.models
        self.map_idx = iterative.map_idx
        self.width = iterative.width
        self.height = iterative.height
        self.gauss = iterative.gauss

        if wrapping == som.WrappingMode.FLAT:
            self.distance_fn = self.__flat_distance
        if wrapping == som.WrappingMode.TOROIDAL:
            self.use_sphere = False
            self.distance_fn = self.__spherical_distance
        if wrapping == som.WrappingMode.SPHERICAL:
            self.use_sphere = True
            self.distance_fn = self.__spherical_distance

        self.inter_model_distances = self.get_inter_model_distances()

    def get_activations(self, encoding):
        # Activation is 1 / Euclidian(models, encoding).
        # The closer a vector is to the encoding, the higher the activation.
        return 1 / (self.models - encoding).square().sum(dim=-1).sqrt()

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
        flip_x = [self.width, 0]
        flip_y = [0, self.height]
        
        if self.use_sphere:
            flip_xy = [self.width, self.height]
            flip_yx = [-self.width, self.height]
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
        distances = torch.empty((len(self.map_idx), len(self.map_idx)))
        for i, p1 in enumerate(self.map_idx):
            for j, p2 in enumerate(self.map_idx):
                distances[i][j] = self.distance_fn(p1, p2)
        return distances


    def mean_encoding_by_bmu(self, batch_encodings, bmus):
        sum_mj = torch.zeros(self.models.shape)
        count_mj = torch.zeros(self.models.shape[0])
        for i, v_idx in enumerate(bmus):
            count_mj[v_idx] += 1
            sum_mj[v_idx] += batch_encodings[i]

        x_mj = torch.zeros(self.models.shape)
        for i, sm in enumerate(sum_mj):
            if count_mj[i] > 0:
                x_mj[i] = sm / count_mj[i]
            else:
                x_mj[i] = torch.zeros(sm.shape)

        return x_mj

    def update_factor(self):
        uf = torch.empty((len(self.map_idx), len(self.map_idx)))
        for i, p1 in enumerate(self.map_idx):
            for j, p2 in enumerate(self.map_idx):
                d = self.distance_fn(p1, p2)
                uf[i][j] = torch.exp(
                    torch.neg(torch.div(d.square(), 2 * self.gauss**2)))
        return uf

    def batch_update_step(self, batch_encodings, bmus):
        h_ij = self.update_factor()
        x_mj = self.mean_encoding_by_bmu(batch_encodings, bmus)

        bmu_count_by_idx = torch.zeros(self.models.shape[0])
        for i, v_idx in enumerate(bmus):
            bmu_count_by_idx[v_idx] += 1

        bse = torch.zeros(self.models.shape)
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

    def __make_test(self, wrapping):
        batch_som = som.SOM((3, 3, 10), wrapping,gauss=1)
        iterative = IterativeSOM(batch_som, wrapping)
        batch_encodings = torch.randint(high=10, size=(60, 10)).float()
        # bmus are selected randomly, so we inject them here
        bmus = torch.cat([batch_som.get_bmu(e) for e in batch_encodings])
        return batch_som, iterative, batch_encodings, bmus

    def test_inter_model_distances(self):
        for wrapping in list(som.WrappingMode):
            with self.subTest(wrapping=wrapping):
                batch_som, iterative, batch_encodings, bmus = self.__make_test(
                    wrapping)

                imd = batch_som.inter_model_distances
                imd_expected = iterative.inter_model_distances
                self.assertTrue(torch.isclose(imd, imd_expected).all())

    def test_update_factor(self):
        for wrapping in list(som.WrappingMode):
            with self.subTest(wrapping=wrapping):
                batch_som, iterative, batch_encodings, bmus = self.__make_test(
                    wrapping)

                h_ij = batch_som.update_factor()
                h_ij_expected = iterative.update_factor()
                self.assertTrue(torch.isclose(h_ij, h_ij_expected).all())

    def test_update_factor_symmetric(self):
        for wrapping in list(som.WrappingMode):
            with self.subTest(wrapping=wrapping):
                batch_som, iterative, batch_encodings, bmus = self.__make_test(
                    wrapping)

                h_ij = batch_som.update_factor()
                h_ij_expected = iterative.update_factor()
                for i in range(len(h_ij_expected)):
                    for j in range(len(h_ij_expected)):
                        self.assertEqual(h_ij_expected[i][j],
                                         h_ij_expected[j][i])

    def test_mean_encoding_by_bmu(self):
        for wrapping in list(som.WrappingMode):
            with self.subTest(wrapping=wrapping):
                batch_som, iterative, batch_encodings, bmus = self.__make_test(
                    wrapping)

                x_mj = batch_som.mean_encoding_by_bmu(batch_encodings, bmus)
                x_mj_expected = iterative.mean_encoding_by_bmu(
                    batch_encodings, bmus)
                self.assertTrue(torch.isclose(x_mj, x_mj_expected).all())

    def test_batch_update_step(self):
        for wrapping in list(som.WrappingMode):
            with self.subTest(wrapping=wrapping):
                batch_som, iterative, batch_encodings, bmus = self.__make_test(
                    wrapping)

                bse = batch_som.batch_update_step(batch_encodings, bmus)
                bse_expected = iterative.batch_update_step(
                    batch_encodings, bmus)
                self.assertTrue(torch.isclose(bse, bse_expected).all())


if __name__ == '__main__':
    unittest.main()
