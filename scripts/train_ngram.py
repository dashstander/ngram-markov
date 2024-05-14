

from concurrent.futures import ThreadPoolExecutor
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
import s3fs 
from tqdm import trange
import torch
from torch.nn.functional import log_softmax
from torch.nn.utils import clip_grad_norm_
from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformer_lens.utils import lm_accuracy, lm_cross_entropy_loss
import wandb

from ngram_markov.sampler import NgramSampler

torch.set_float32_matmul_precision('high')
allow_ops_in_compiled_graph()
fs = s3fs.S3FileSystem()


def save_to_s3(weights, optimizer, config, rng, bucket, step):
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


@torch.no_grad()
def do_validation(model, group):
    valid_msg = {}
    data, labels = [tensor.to('cuda') for tensor in group.generate()]
    #even_inds = torch.arange(2, data.shape[1], 2).to('cuda:0')
    logits = model(data, return_type='logits')
    loss = seq2seq_cross_entropy_loss(logits, labels)
    acc = seq2seq_accuracy(logits, labels)
    valid_msg[f'loss/validation'] = loss.item()
    valid_msg[f'accuracy/validation'] = acc.item()
    return valid_msg


def train(model, optimizer, scheduler, config, num_steps, group, bucket):
    #model_fn = torch.compile(model)
    msg = do_validation(model, group)
    wandb.log(msg)

    executor = ThreadPoolExecutor(max_workers=20)

    with trange(1, num_steps+1) as t:
        for i in t:
            data, labels = [tensor.to('cuda') for tensor in group.generate()]
            optimizer.zero_grad()
            logits = model(data, return_type='logits')
            loss = lm_c(logits, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            msg = {'loss/train': loss.item()}

            if i % 10 == 0:
                valid_losses = do_validation(model, group)
                msg.update(valid_losses)

            if i % 10 == 0:
                t.set_postfix(loss=loss.item())
            
            wandb.log(msg)
            if i % 100 == 0:
                executor.submit(
                    save_to_s3,
                    model.state_dict(),
                    optimizer.state_dict(),
                    config,
                    torch.random.get_rng_state(),
                    bucket,
                    i
                )
    save_to_s3(
        model.state_dict(),
        optimizer.state_dict(),
        config,
        torch.random.get_rng_state(),
        bucket,
        i
    )
            


def main(_):

    

    N = 2
    layers = 4
    context = 64
    batch_size = 1024
    seed = 0
    path = f'C{N}-seq2seq-{layers}-{seed}'
    bucket = f's3://automatic-circuits-01/{path}'
    

    cfg = {
        "d_model": 256,
        "d_head": 64,
        "n_heads": 4,
        "d_mlp": 1024,
        "n_ctx": context + 1,
        "n_layers": layers,
        "d_vocab": N + 1,
        "act_fn": "relu"
    }
    num_steps = 20_000

    wandb.init(config=cfg, entity='dstander', project='transformer-parities-seq2seq')

    torch.manual_seed(seed)

    wandb.init(config=cfg, entity='dstander', project='transformer-parity-seq2seq')

    config = HookedTransformerConfig(**cfg)
    model = HookedTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1.0e-6)


    with fs.open(f'{bucket}/0.pth', mode='wb') as file:
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'config': config,
                'rng': torch.random.get_rng_state()
            }, 
            file
        )

    dataset = CyclicGroupGenerator(context, N, batch_size)
    
    wandb.watch(model, log='all', log_freq=200)

    try:
        train(model, optimizer, scheduler, config, num_steps, dataset, bucket)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main(None)

