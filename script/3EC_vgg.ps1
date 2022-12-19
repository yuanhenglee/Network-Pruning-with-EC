

echo "start training by parameter"
# Measure-Command {python .\OneStepEC.py --model_name=vgg11 --es_n_iter=100 > .\output\vgg_cifar10_bp_100_onestep.out }
# Measure-Command {python .\NStepEC.py --model_name=vgg11 --es_n_iter=100 > .\output\vgg_cifar10_bp_100_nstep.out }
# Measure-Command {python .\OneFifthEC.py --model_name=vgg11 --es_n_iter=100 > .\output\vgg_cifar10_bp_100_onefifth.out }

echo "start training by channel"
Measure-Command {python .\OneStepEC.py --model_name=vgg11 --es_n_iter=100 --pruning_method=by_channel > .\output\vgg_cifar10_bc_100_onestep.out }
Measure-Command {python .\NStepEC.py --model_name=vgg11 --es_n_iter=100 --pruning_method=by_channel > .\output\vgg_cifar10_bc_100_nstep.out }
Measure-Command {python .\OneFifthEC.py --model_name=vgg11 --es_n_iter=100 --pruning_method=by_channel > .\output\vgg_cifar10_bc_100_onefifth.out }