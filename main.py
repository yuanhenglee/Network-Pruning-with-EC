import numpy as np
import threading
import argparse

import torch
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem

from prune import ModelPruner

class PruningProblem(ElementwiseProblem):

    def __init__(self, mp: ModelPruner):
        super().__init__(n_var=mp.prunable_layer_num, n_obj=1,
                         xl=0., xu=1.0)
        self._mp = mp

    def _evaluate(self, x, out, *args, **kwargs):
        self._mp.prune_model(x)
        f = self._mp.get_fitness_score()
        out["F"] = f


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--pruning_method', type=str, default='by_parameter')
    parser.add_argument('--es_n_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model_name = args.model_name
    dataset = args.dataset
    pruning_method = args.pruning_method
    es_n_iter = args.es_n_iter
    device = args.device

    # model_name = 'resnet18'
    # dataset = 'CIFAR10'
    # pruning_method = 'by_parameter'
    # es_n_iter = 10
    # device = 'cuda'


    mp = ModelPruner(model_name, dataset, pruning_method)

    problem = PruningProblem(mp)
    print('baseline:', mp.baseline)
    algorithm = CMAES(x0=np.random.random(problem.n_var))
    res = minimize(problem,
                   algorithm,                
                   ('n_iter', es_n_iter),
                   seed=1,
                   verbose=False)

    print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")
    print(mp.get_fitness_score(verbose=True))

    pruned_model_saving_path = f"{mp.config['model_weights_root_path']}pruned/pruned_{model_name}_{dataset}.model"
    torch.save(mp.cached_model, pruned_model_saving_path)
    
    # Load pruned model
    # pruned_model_saving_path = f"{mp.config['model_weights_root_path']}pruned/pruned_{model_name}_{dataset}.model"
    # model = torch.load(pruned_model_saving_path, map_location=torch.device(device))
    # model = model.to(device)
    # print(mp.get_fitness_score(model))