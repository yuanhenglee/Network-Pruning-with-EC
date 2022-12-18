

echo "start training by parameter"
# Measure-Command {python .\OneStepEC.py --es_n_iter=500 > .\output\res_cifar10_bp_500_onestep.out }
# Measure-Command {python .\NStepEC.py --es_n_iter=500 > .\output\res_cifar10_bp_500_nstep.out }
Measure-Command {python .\OneFifthEC.py --es_n_iter=500 > .\output\res_cifar10_bp_500_onefifth.out }

echo "start training by channel"
Measure-Command {python .\OneStepEC.py --es_n_iter=500 --pruning_method=by_channel > .\output\res_cifar10_bc_500_onestep.out }
Measure-Command {python .\NStepEC.py --es_n_iter=500 --pruning_method=by_channel > .\output\res_cifar10_bc_500_nstep.out }
Measure-Command {python .\OneFifthEC.py --es_n_iter=500 --pruning_method=by_channel > .\output\res_cifar10_bc_500_onefifth.out }