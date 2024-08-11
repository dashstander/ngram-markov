import cola
import torch


@torch.compile
def calc_perron_vector(matrix, n_iter=5):
    vec = torch.ones((matrix.shape[0],), device=matrix.device)
    for _ in range(n_iter):
        new_vec = matrix @ vec
        new_vec /= new_vec.pow(2).sum().sqrt()
        vec = new_vec
    return vec
    

def symmetric_directed_laplacian(adjacency_matrix, device, n_iter=5):
    perron = cola.ops.Diagonal(calc_perron_vector(adjacency_matrix, n_iter=n_iter)).to(device)
    adj = cola.ops.Dense(adjacency_matrix).to(device)
    Ident = cola.ops.I_like(perron).to(device)
    laplacian = (cola.sqrt(perron) @ adj @ cola.inv(cola.sqrt(perron)))
    laplacian +=  (cola.inv(cola.sqrt(perron)) @ adj.T @ cola.sqrt(perron))
    laplacian /= 2       
    return cola.SelfAdjoint(Ident - laplacian).to(device)


def directed_laplacian(adjacency_matrix, device, n_iter=5):
    perron = cola.ops.Diagonal(calc_perron_vector(adjacency_matrix, n_iter=n_iter)).to(device)
    adj = cola.ops.Dense(adjacency_matrix).to(device)
    Ident = cola.ops.I_like(perron).to(device)
    laplacian = (cola.sqrt(perron) @ adj @ cola.inv(cola.sqrt(perron)))
    return (Ident - laplacian).to(device)
