
echo "start training by parameter"
Measure-Command {python .\OneStepEC.py --dataset=MNIST --es_n_iter=100 > .\output\res_mnist_bp_100_onestep.out }
Measure-Command {python .\NStepEC.py --dataset=MNIST --es_n_iter=100 > .\output\res_mnist_bp_100_nstep.out }
Measure-Command {python .\OneFifthEC.py --dataset=MNIST --es_n_iter=100 > .\output\res_mnist_bp_100_onefifth.out }

echo "start training by channel"
Measure-Command {python .\OneStepEC.py --dataset=MNIST --es_n_iter=100 --pruning_method=by_channel > .\output\res_mnist_bc_100_onestep.out }
Measure-Command {python .\NStepEC.py --dataset=MNIST --es_n_iter=100 --pruning_method=by_channel > .\output\res_mnist_bc_100_nstep.out }
Measure-Command {python .\OneFifthEC.py --dataset=MNIST --es_n_iter=100 --pruning_method=by_channel > .\output\res_mnist_bc_100_onefifth.out }