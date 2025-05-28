import os

import numpy as np
import torch

from annealed_bg.systems.gmm.system import create_FAB_gmm_target

if __name__ == "__main__":
    target = create_FAB_gmm_target()

    torch.cuda.manual_seed(12345)

    test_set = target.test_set.cpu().numpy()

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "gmm"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "data.npy")

    with open(output_path, "wb") as f:
        np.save(f, test_set)

    print("Saved GMM test set with shape", test_set.shape, "to", output_path)

    print(test_set)
    print(np.max(test_set), np.min(test_set))
