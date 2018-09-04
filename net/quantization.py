import torch
import numpy as np
from sklearn.cluster import KMeans

def apply_weight_sharing(model):
    """
    Applies weight sharing to the given model
    """
    for module in model.children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        original_shape = weight.shape
        min_ = min(x for x in np.nditer(weight))
        max_ = max(x for x in np.nditer(weight))
        space = np.linspace(min_, max_, num=32)
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(weight.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(original_shape)
        module.weight.data = torch.from_numpy(new_weight).to(dev)


