[model weights](https://drive.google.com/drive/folders/1zm2v6JbL3GACy7GA7qeZppb0uq_tpsDd?usp=sharing)

    # Example
    if __name__ == '__main__':
        mp = ModelPruner('MNIST', 'by_channel')
        print(mp.baseline)

        # TODO: Run your EC Here
        # ...
        prune_amount_list = [0.5 for i in range(mp.prunable_layer_num)] # from the result of ES
        pruned_model = mp.prune_model(prune_amount_list)
        print(mp.get_fitness_score(pruned_model))

        # Save final pruned model
        path = './model_weights/pruned_vgg11_MNIST.pt'
        torch.save(pruned_model.state_dict(), path)