

echo "start training by parameter"
# Measure-Command {python .\OneStepEC.py --es_n_iter=100 > .\output\res_cifat10_a5_bp_100_onestep.out }
# Measure-Command {python .\NStepEC.py --es_n_iter=100 > .\output\res_cifat10_a5_bp_100_nstep.out }
# Measure-Command {python .\OneFifthEC.py --es_n_iter=100 > .\output\res_cifat10_a5_bp_100_onefifth.out }

echo "start training by channel"
Measure-Command {python .\OneStepEC.py --es_n_iter=100 --pruning_method=by_channel > .\output\res_cifat10_c5_bc_100_onestep.out }
Measure-Command {python .\NStepEC.py --es_n_iter=100 --pruning_method=by_channel > .\output\res_cifat10_c5_bc_100_nstep.out }
Measure-Command {python .\OneFifthEC.py --es_n_iter=100 --pruning_method=by_channel > .\output\res_cifat10_c5_bc_100_onefifth.out }