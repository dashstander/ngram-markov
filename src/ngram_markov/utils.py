import torch


def create_ngrams(tensor, n):
    assert n >= 1
    stride = tensor.stride()
    size = tensor.size()

    # Create a new stride that moves by 1 element in the original tensor
    new_stride = (stride[0], stride[0])

    # Calculate the new shape of the tensor
    new_size = (size[0] - n + 1, n)

    # Use as_strided to create a new view of the tensor with the desired shape and stride
    ngrams = torch.as_strided(tensor, size=new_size, stride=new_stride)
    return ngrams



def save_to_s3(fs, weights, optimizer, config, rng, bucket, step):
    with fs.open(f'{bucket}/{step}.pth', mode='wb') as file:
        torch.save(
            {
                'model': weights,
                'optimizer': optimizer, 
                'config': config,
                'rng': rng
            }, 
            file
        )