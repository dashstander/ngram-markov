import torch


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