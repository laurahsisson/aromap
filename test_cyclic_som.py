import torch
import som
import unittest


# Testing by using an iterative implementation.
class SOM__(object):

    def __init__(self, som, is_cyclic):
        self.models = som.models
        self.map_idx = som.map_idx
        self.width = som.width
        self.height = som.height
        self.gauss = som.gauss

        if is_cyclic:
            self.distance_fn = self.__cyclic_distance
        else:
            self.distance_fn = self.__flat_distance




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

    def mean_encoding_by_bmu__(self, encodings, bmus):
        sum_mj = torch.zeros(self.models.shape)
        count_mj = torch.zeros(self.models.shape[0])
        for i, v_idx in enumerate(bmus):
            count_mj[v_idx] += 1
            sum_mj[v_idx] += encodings[i]

        x_mj = torch.zeros(self.models.shape)
        for i, sm in enumerate(sum_mj):
            if count_mj[i] > 0:
                x_mj[i] = sm / count_mj[i]
            else:
                x_mj[i] = torch.zeros(sm.shape)

        return x_mj

    def __cyclic_distance(self,p1,p2):
        eye = [0, 0]
        flip_x = [self.width, 0]
        flip_y = [0, self.height]
        # To change this to a toroidal SOM, flip_xy should be removed.
        flip_xy = [self.width,self.height]
        dist_all = []
        for f in [eye, flip_x, flip_y, flip_xy]:
            for sgn in [1, -1]:
                transform_p2 = p2 + sgn * torch.tensor(f)
                xy_dist = p1 - transform_p2
                d = torch.sqrt(torch.sum(torch.square(xy_dist), dim=-1))
                dist_all.append(d)
        return min(dist_all)

    def __flat_distance(self,p1,p2):
        xy_dist = p1 - p2
        return torch.sqrt(torch.sum(torch.square(xy_dist), dim=-1))

    def update_factor__(self):
        uf = torch.empty((len(self.map_idx), len(self.map_idx)))
        for i, p1 in enumerate(self.map_idx):
            for j, p2 in enumerate(self.map_idx):
                d = self.distance_fn(p1,p2)
                uf[i][j] = torch.exp(
                    torch.neg(torch.div(d.square(), 2 * self.gauss**2)))
        return uf

    def batch_update_step__(self, bmus, encodings):
        h_ij = self.update_factor__()
        x_mj = self.mean_encoding_by_bmu__(encodings, bmus)

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

    def setUp(self):
        self.mm = som.SOM((2, 1, 10), True)
        self.mm__ = SOM__(self.mm,True)
        self.encodings = torch.randint(high=10, size=(60, 10)).float()
        # self.bmus are selected randomly, so we inject them here
        self.bmus = torch.cat([self.mm.get_bmu(e) for e in self.encodings])

        self.assertTrue(torch.all(self.mm.models == self.mm__.models))

    def test_update_factor(self):
        h_ij = self.mm.update_factor()
        h_ij__ = self.mm__.update_factor__()
        self.assertTrue(torch.isclose(h_ij, h_ij__).all())

    def test_update_factor_symmetric(self):
        h_ij__ = self.mm__.update_factor__()
        for i in range(len(h_ij__)):
            for j in range(len(h_ij__)):
                self.assertEqual(h_ij__[i][j], h_ij__[j][i])

    def test_mean_encoding_by_bmu(self):
        x_mj = self.mm.mean_encoding_by_bmu(self.encodings, self.bmus)
        x_mj__ = self.mm__.mean_encoding_by_bmu__(self.encodings, self.bmus)
        self.assertTrue(torch.isclose(x_mj, x_mj__).all())

    def test_batch_update_step(self):
        bse = self.mm.batch_update_step(self.bmus, self.encodings)
        bse__ = self.mm__.batch_update_step__(self.bmus, self.encodings)
        self.assertTrue(torch.isclose(bse, bse__).all())


if __name__ == '__main__':
    unittest.main()
