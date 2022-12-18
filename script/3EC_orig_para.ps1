

echo "start training by parameter"
Measure-Command {python .\OneStepEC.py --es_n_iter=200 > .\output\res_cifat10_a8_bp_200_onestep.out }
Measure-Command {python .\NStepEC.py --es_n_iter=200 > .\output\res_cifat10_a8_bp_200_nstep.out }
Measure-Command {python .\OneFifthEC.py --es_n_iter=200 > .\output\res_cifat10_a8_bp_200_onefifth.out }

echo "start training by channel"
Measure-Command {python .\OneStepEC.py --es_n_iter=200 --pruning_method=by_channel > .\output\res_cifat10_a8_bc_200_onestep.out }
Measure-Command {python .\NStepEC.py --es_n_iter=200 --pruning_method=by_channel > .\output\res_cifat10_a8_bc_200_nstep.out }
Measure-Command {python .\OneFifthEC.py --es_n_iter=200 --pruning_method=by_channel > .\output\res_cifat10_a8_bc_200_onefifth.out }