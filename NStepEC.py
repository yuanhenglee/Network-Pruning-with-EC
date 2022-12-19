import numpy as np
import math
from prune import ModelPruner
import argparse


#Hyperparameters could be change here:

# HP_DECISION_INITVAL = 0.01   # Initial value for decision variable (array)
HP_STEPSIZE_INITVAL = 0.1   # Initial value for stepsize (array)
HP_ITERATIONS = 5           # Run how many iterations/epoch
HP_CHILD_LAMBDA = 2         # Run (1+child_lambda)-EC
HP_TAU_PRUM_PARAM = 0.05    # global step size changing parameter, tau_prum = HP_TAU_PRUM_PARAM/sqrt(2*sqrt(N))
HP_TAU_PARAM = 0.05         # local step size changing parameter, tau = HP_TAU_PARAM/sqrt(2*N)
HP_ELPSLON_VALUE = 0.001    # The threshold of the step size, if stepsize < elpslon, stepsize=elpslon

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
    HP_ITERATIONS = args.es_n_iter
    device = args.device

    if pruning_method == 'by_parameter':
        HP_DECISION_INITVAL = 0.5
    else:
        HP_DECISION_INITVAL = 0.01

    mp = ModelPruner(model_name, dataset, pruning_method)

    # the phony generator
    def generator(in_arr):
        pruned_model = mp.prune_model(in_arr)
        return mp.get_fitness_score(pruned_model)

    def N_step_single_pass(decision_len, deci_arr_in, stepsize_arr_in, tau_prum_in, tau_in, elpslon_zero):
        random_normal_i = np.random.normal(0, 1, size=(decision_len))
        random_normal = np.random.normal(0, 1, size=(1))
        
        #store input variable
        deci_arr = np.copy(deci_arr_in)
        stepsize_arr = np.copy(stepsize_arr_in)
        
        for i in range(decision_len):
            stepsize_arr[i]*=math.exp(tau_prum_in*random_normal[0]+tau_in*random_normal_i[i])

            if(stepsize_arr[i] > elpslon_zero):
                deci_arr[i] += stepsize_arr[i]*random_normal_i[i]
            else:
                deci_arr[i] += elpslon_zero*random_normal_i[i]

            if(deci_arr[i] <= 0.001):
                deci_arr[i] = 0.001
            elif(deci_arr[i] >= 0.999):
                deci_arr[i] = 0.999

        return deci_arr, stepsize_arr


    def ESXPlusX_NS(Decision_var_record, Stepsize_record, Best_score_record,
                    Decision_arr_init, Stepsize_arr_init,
                    Target_run, child_lambda, hp_tau_prum, hp_tau, hp_elpslon_zero):
        train_runs = 0
        
        decision_var_arr = np.copy(Decision_arr_init)
        stepsize_arr = np.copy(Stepsize_arr_init)
        best_score, best_acc = generator(decision_var_arr)

        
        tau_in = hp_tau/((1/math.sqrt(2*len(Decision_arr_init))))
        tau_prum_in = hp_tau_prum/math.sqrt(2*math.sqrt(len(Decision_arr_init)))

        while(train_runs < Target_run):
            # print(f'Training epoch: {train_runs+1}/{Target_run}')
            memorize_parent_decision_var_arr = np.copy(decision_var_arr)
            memorize_parent_stepsize_arr = np.copy(stepsize_arr)
    
            for run in range(child_lambda):
                tmp_decision_var_arr, tmp_stepsize = N_step_single_pass(len(Decision_arr_init),memorize_parent_decision_var_arr,memorize_parent_stepsize_arr,
                                                                        tau_prum_in, tau_in, hp_elpslon_zero)
                tmp_score, tmp_acc = generator(tmp_decision_var_arr)


                if(tmp_score < best_score):
                    best_score = tmp_score
                    best_acc = tmp_acc
                    decision_var_arr = np.copy(tmp_decision_var_arr)
                    stepsize_arr = tmp_stepsize
            # Decision_var_record.append(decision_var_arr)
            # Stepsize_record.append(stepsize_arr)
            # Best_score_record.append(best_score)

            # print info
            print( f'{train_runs+1:5d} | {best_score:5.3f} | {np.mean(stepsize_arr):5.3f} | {best_acc:5.3f}' )

            train_runs+=1

        return decision_var_arr, stepsize_arr, best_score, best_acc



    #auto generated parameter (Don't change)
    Decision_arr = np.array([HP_DECISION_INITVAL for i in range(mp.prunable_layer_num)]) 
    Stepsize_arr = np.array([HP_STEPSIZE_INITVAL for i in range(mp.prunable_layer_num)]) 

    # print parameters
    print('model:', model_name)
    print('dataset:', dataset)
    print('pruning method:', mp.pruning_method)
    print('device:', device)
    print('iterations:', HP_ITERATIONS)
    print('child_lambda:', HP_CHILD_LAMBDA)
    print('tau:', HP_TAU_PARAM)
    print('tau_prum:', HP_TAU_PRUM_PARAM)
    print('elpslon:', HP_ELPSLON_VALUE)
    print('stepsize_init:', HP_STEPSIZE_INITVAL)
    print('decision_init:', HP_DECISION_INITVAL)

    DV_record = [] #records the decision variable of each epoch
    SZ_record = [] #records the step size of each epoch
    BS_reocrd = [] #records the best solution of each run


    # print baseline performance
    print('baseline:', mp.baseline)

    # print title
    print( f'{"Epoch":5s} | {"Score":7s} | {"Step":5s} | {"Acc":5s}')

    #run EC algorithm
    ans_decvar, ans_stepsize, ans_bestscore, best_acc = ESXPlusX_NS(
        Decision_var_record = DV_record,
        Stepsize_record = SZ_record,
        Best_score_record = BS_reocrd, 
        Decision_arr_init = Decision_arr,
        Stepsize_arr_init = Stepsize_arr,
        Target_run = HP_ITERATIONS,
        child_lambda = HP_CHILD_LAMBDA,
        hp_tau_prum = HP_TAU_PRUM_PARAM,
        hp_tau = HP_TAU_PARAM,
        hp_elpslon_zero = HP_ELPSLON_VALUE
    )

    print(f"Best solution found: \nX = {ans_decvar}\nF = {ans_bestscore}")

    print(mp.get_fitness_score(mp.cached_model, verbose=True))