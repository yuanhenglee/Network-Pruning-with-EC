import numpy as np
import math
from prune import ModelPruner
import argparse



#Hyperparameters could be change here:

HP_DECISION_INITVAL = 0.5   # Initial value for decision variable (array)
HP_STEPSIZE_INITVAL = 0.1   # Initial value for stepsize
HP_ITERATIONS = 5           # Run how many iterations/epoch
HP_CHILD_LAMBDA = 2         # Run (1+child_lambda)-EC
HP_TAU_PARAM = 0.1          # tau = HP_TAU_PARAM/((1/sqrt(N)),
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

    mp = ModelPruner(model_name, dataset, pruning_method)

    #this is the phony generator
    def generator(in_arr):
        pruned_model = mp.prune_model(in_arr)
        return mp.get_fitness_score(pruned_model)

    def One_step_single_pass(decision_len, deci_arr_in, stepsize_in, tau_in, elpslon_zero):
        random_normal_i = np.random.normal(0, 1, size=(decision_len))
        random_normal = np.random.normal(0, 1, size=(1))
        
        #store input variable
        deci_arr = np.copy(deci_arr_in)
        stepsize = np.copy(stepsize_in)
        
        stepsize*=math.exp(tau_in*random_normal[0])
        for i in range(decision_len):

            if(stepsize > elpslon_zero):
                deci_arr[i] += stepsize*random_normal_i[i]
            else:
                deci_arr[i] += elpslon_zero*random_normal_i[i]

            if(deci_arr[i] <= 0.001):
                deci_arr[i] = 0.001
            elif(deci_arr[i] >= 1):
                deci_arr[i] = 0.999

        return deci_arr, stepsize


    def ESXPlusX_OS(Decision_var_record, Stepsize_record, Best_score_record, Decision_arr_init, Stepsize_init
                    , Target_run, child_lambda, hp_tau, hp_elpslon_zero):
        train_runs = 0
        
        decision_var_arr = np.copy(Decision_arr_init)
        stepsize = Stepsize_init
        best_score = generator(decision_var_arr)

        
        tau_in = hp_tau/((1/math.sqrt(len(Decision_arr_init))))

        while(train_runs < Target_run):
            print(f'Training epoch: {train_runs+1}/{Target_run}')
            memorize_parent_decision_var_arr = np.copy(decision_var_arr)
            memorize_parent_stepsize = stepsize
    
            for run in range(child_lambda):
                tmp_decision_var_arr, tmp_stepsize = One_step_single_pass(len(Decision_arr_init),memorize_parent_decision_var_arr,memorize_parent_stepsize,tau_in,hp_elpslon_zero)
                tmp_score = generator(tmp_decision_var_arr)


                if(tmp_score < best_score):
                    best_score = tmp_score
                    decision_var_arr = np.copy(tmp_decision_var_arr)
                    stepsize = tmp_stepsize
            Decision_var_record.append(decision_var_arr)
            Stepsize_record.append(float(stepsize))
            Best_score_record.append(best_score)

            # print info
            # print( f'{train_runs+1:3d} | {best_score:10.6f} | {stepsize:10.6f}')

            train_runs+=1

        return decision_var_arr, stepsize, best_score

    #auto generated parameter (Don't change)
    Decision_arr = np.array([HP_DECISION_INITVAL for i in range(mp.prunable_layer_num)]) 

    DV_record = [] #records the decision variable of each epoch
    SZ_record = [] #records the step size of each epoch
    BS_reocrd = [] #records the best solution of each run

    #run EC algorithm
    ans_decvar, ans_stepsize, ans_bestscore = ESXPlusX_OS(Decision_var_record=DV_record,Stepsize_record=SZ_record,Best_score_record=BS_reocrd, 
                Decision_arr_init=Decision_arr,Stepsize_init=HP_STEPSIZE_INITVAL,
                Target_run=HP_ITERATIONS,child_lambda=HP_CHILD_LAMBDA,hp_tau=HP_TAU_PARAM,hp_elpslon_zero=HP_ELPSLON_VALUE)


    print(DV_record)
    print(SZ_record)
    print(BS_reocrd)
