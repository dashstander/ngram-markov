import numpy as np
import torch
from tqdm import tqdm
import os

import numpy as np
from tqdm import tqdm


def efron_stein_decomposition(tensor):
    """
    Perform Efron-Stein decomposition on a 3D tensor representing a single output logit.
    
    :param tensor: numpy array of shape (512, 512, 512)
    :return: tuple of (zeroth_order, first_order, second_order, third_order)
    """
    # Zeroth-order effect (mean)
    zeroth_order = np.mean(tensor)
    
    # First-order effects
    first_order = {
        0: np.mean(tensor, axis=(1, 2)) - zeroth_order,
        1: np.mean(tensor, axis=(0, 2)) - zeroth_order,
        2: np.mean(tensor, axis=(0, 1)) - zeroth_order
    }
    
    # Second-order effects
    second_order = {
        (0, 1): np.mean(tensor, axis=2) - first_order[0][:, np.newaxis] - first_order[1][np.newaxis, :] - zeroth_order,
        (0, 2): np.mean(tensor, axis=1) - first_order[0][:, np.newaxis] - first_order[2][np.newaxis, :] - zeroth_order,
        (1, 2): np.mean(tensor, axis=0) - first_order[1][:, np.newaxis] - first_order[2][np.newaxis, :] - zeroth_order
    }
    
    # Third-order effect
    third_order = (tensor - 
                   zeroth_order - 
                   first_order[0][:, np.newaxis, np.newaxis] -
                   first_order[1][np.newaxis, :, np.newaxis] -
                   first_order[2][np.newaxis, np.newaxis, :] -
                   second_order[(0, 1)][:, :, np.newaxis] -
                   second_order[(0, 2)][:, np.newaxis, :] -
                   second_order[(1, 2)][np.newaxis, :, :])
    
    return zeroth_order, first_order, second_order, third_order


def check_reconstruction(tensor, zeroth_order, first_order, second_order, third_order):
    reconstructed = (zeroth_order + 
                     first_order[0][:, np.newaxis, np.newaxis] +
                     first_order[1][np.newaxis, :, np.newaxis] +
                     first_order[2][np.newaxis, np.newaxis, :] +
                     second_order[(0,1)][:, :, np.newaxis] +
                     second_order[(0,2)][:, np.newaxis, :] +
                     second_order[(1,2)][np.newaxis, :, :] +
                     third_order)
    max_error = np.max(np.abs(tensor - reconstructed))
    print(f"Maximum reconstruction error: {max_error}")
    assert np.allclose(tensor, reconstructed, atol=1.0e-5), "Reconstruction failed"



def check_variances(tensor, first_order, second_order, third_order):
    total_var = np.var(tensor)
    component_vars = (np.var(first_order[0]) + np.var(first_order[1]) + np.var(first_order[2]) +
                      np.var(second_order[(0,1)]) + np.var(second_order[(0,2)]) + np.var(second_order[(1,2)]) +
                      np.var(third_order))
    print(f"Total variance: {total_var}")
    print(f"Sum of component variances: {component_vars}")
    assert np.allclose(total_var, component_vars, atol=1.0e-5), "Variance decomposition failed"


if __name__ == "__main__":
    ngram_n = 3
    epoch = 53_000
    num_tokens = 512

    # Create a directory to store the results
    output_dir = f'/media/External01/efron_stein_results_epoch_{epoch}'
    os.makedirs(output_dir, exist_ok=True)

    mm_array = np.memmap(
        f'ngram_{ngram_n}_outputs_epoch_{epoch}.npy',
        dtype='float32', 
        mode='r',
        shape=(num_tokens, num_tokens, num_tokens, num_tokens)
    )

    for output_token in tqdm(range(num_tokens), desc="Processing output tokens"):
        print(f'\nProcessing output token {output_token}')
        
        print(f'Getting data....')
        data = np.copy(mm_array[..., output_token])

        print(f'Beginning Efron-Stein decomposition')
        zeroth_order, first_order, second_order, third_order = efron_stein_decomposition(data)

        print(f'Running correctness checks')
        check_reconstruction(data, zeroth_order, first_order, second_order, third_order)
        check_variances(data, first_order, second_order, third_order)

        print("Checks passed successfully!")

        # Create a dictionary to store all components
        es_decomposition = {
            'zeroth_order': torch.tensor(zeroth_order),
            'first_order': {k: torch.from_numpy(v) for k, v in first_order.items()},
            'second_order': {f'{k[0]}_{k[1]}': torch.from_numpy(v) for k, v in second_order.items()},
            'third_order': torch.from_numpy(third_order)
        }

        # Save the entire decomposition as a single file
        torch.save(es_decomposition, f'{output_dir}/es_decomposition_{output_token}.pt')

    print("All output tokens processed and results saved!")