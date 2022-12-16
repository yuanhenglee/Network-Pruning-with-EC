import numpy as np
import math
from prune import ModelPruner
import argparse



# Hyperparameters could be change here:

# Self adaptation of step-size: 1/5 Rule
# It runs for a certain amount of runs with fix stepsize, and determine if the 1/5 
# results are better than the parent generation.
# if yes, stepsize /= a
# else stepsize *= a


HP_DECISION_INITVAL = 0.5   # Initial value for decision variable
HP_STEPSIZE_INITVAL = 0.1   # Initial value for stepsize
HP_GGENERATIONS = 10         # Run how many runs before updating the step size
HP_A = 0.817                # a magical value to update step size, repordely 0.817 <= a <= 1
HP_ITERATIONS = 15           # Run how many runs, the total pruning is HP_GGENERATIONS*HP_ITERATIONS


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

    def One_fifth_single_pass(decision_len, deci_arr_in, stepsize_in):
        random_normal_i = np.random.normal(0, 1, size=(decision_len))
        # random_normal = np.random.normal(0, 1, size=(1))
        
        #store input variable
        deci_arr = np.copy(deci_arr_in)
        stepsize = np.copy(stepsize_in)
        
        for i in range(decision_len):

            deci_arr[i] += stepsize*random_normal_i[i]
            if(deci_arr[i] <= 0.001):
                deci_arr[i] = 0.001
            elif(deci_arr[i] >= 1):
                deci_arr[i] = 0.999

        return deci_arr


    def ESXPlusX_OneFifth(  Decision_var_record, Stepsize_record, Best_score_record, PS_record,
                            Decision_arr_init, Stepsize_init, Ggenerations, hp_a_param,
                            Target_run):
        train_runs = 0
        
        decision_var_arr = np.copy(Decision_arr_init)
        stepsize = Stepsize_init
        
        best_score, best_acc = generator(decision_var_arr)

        while(train_runs < Target_run):
            # print(f'Training epoch: {train_runs+1}/{Target_run}')
            # memorize_parent_decision_var_arr = np.copy(decision_var_arr)
            # memorize_parent_stepsize = stepsize

            evolve_kids = 0
            for run in range(Ggenerations):

                tmp_decision_var_arr = One_fifth_single_pass(len(Decision_arr_init),decision_var_arr,stepsize)
                tmp_score, tmp_acc = generator(tmp_decision_var_arr)


                if(tmp_score < best_score):
                    evolve_kids+=1
                    best_score = tmp_score
                    best_acc = tmp_acc
                    decision_var_arr = np.copy(tmp_decision_var_arr)
                    
            if(evolve_kids > (Ggenerations/5)):
                stepsize = stepsize/hp_a_param
            elif(evolve_kids < (Ggenerations/5)):
                stepsize = stepsize*hp_a_param
            
            # Decision_var_record.append(decision_var_arr)
            # Stepsize_record.append(float(stepsize))
            # Best_score_record.append(best_score)
            # PS_record.append(evolve_kids/Ggenerations)

            # print info
            print( f'{train_runs+1:5d} | {best_score:5.3f} | {stepsize:5.3f} | {best_acc:5.3f}' )

            train_runs+=1

        return decision_var_arr, stepsize, best_score, best_acc

    #auto generated param(Don't change)
    Decision_arr = np.array([HP_DECISION_INITVAL for i in range(mp.prunable_layer_num)]) 
    DV_record = []
    SZ_record = []
    BS_reocrd = []
    PS_record = []

    # print baseline performance
    print('baseline:', mp.baseline)

    # print title
    print( f'{"Epoch":5s} | {"Score":7s} | {"Step":5s} | {"Acc":5s}')

    ans_decvar, ans_stepsize, ans_bestscore, best_acc = ESXPlusX_OneFifth(Decision_var_record=DV_record,Stepsize_record=SZ_record,Best_score_record=BS_reocrd, PS_record=PS_record,
                Decision_arr_init=Decision_arr,Stepsize_init=HP_STEPSIZE_INITVAL,
                Ggenerations=HP_GGENERATIONS, hp_a_param=HP_A, Target_run=HP_ITERATIONS)


    print(f"Best solution found: \nX = {ans_decvar}\nF = {ans_bestscore}")

    print(mp.get_fitness_score(mp.cached_model, verbose=True))