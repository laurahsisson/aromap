import torch
import numpy as np
import scipy
import utils
import enum

WrappingMode = enum.Enum('WrappingMode', ['FLAT', 'TOROIDAL', 'SPHERICAL'])

class OnehotModels(object):
    def init_models(self,shape):
        width, height, dim = shape
        # Select a random index to use as the hot element.
        idxs = torch.randint(low=0,
                             high=dim,
                             size=(width, height))
        # Convert to one hot of shape.
        return torch.nn.functional.one_hot(idxs, num_classes=dim).float()

    def get_activations(self,models,encoding):
        raise torch.nn.functional.cosine_similarity(models,encoding,dim=-1)

    # Not vectorized but relatively quick
    def mean_encoding_by_bmu(self, encodings, bmus):
        # Get unique BMU values
        unique_bmus = torch.unique(bmus)

        # Create a mask for each unique BMU value
        bmu_masks = [(bmus == bmu_value) for bmu_value in unique_bmus]

        # Initialize a tensor to store mean encodings for each BMU
        mean_encodings = torch.zeros_like(encodings)

        # Calculate cosine similarity between each pair of encodings for each BMU
        for bmu_mask in bmu_masks:
            bmu_encodings = encodings[bmu_mask, :]
            print(bmu_encodings)
            similarity_matrix = torch.nn.functional.cosine_similarity(bmu_encodings, bmu_encodings, dim=1)
            max_row_indices = torch.argmax(similarity_matrix.sum(dim=1))

            # Use advanced indexing to assign the mean encoding for each BMU
            mean_encodings[bmu_mask, :] = bmu_encodings[max_row_indices, :]

        return mean_encodings

def test():
    mm = OnehotModels()
    encoding = torch.rand(5,3)
    bmus = torch.randint(low=0,high=2,size=(5,))
    print(mm.mean_encoding_by_bmu(encoding,bmus))
#     enc = torch.rand(3)
#     print(models)
#     print(enc)
#     print(.shape)


if __name__ == "__main__":
    test()
