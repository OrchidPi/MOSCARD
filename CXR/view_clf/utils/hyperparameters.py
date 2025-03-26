
hyperparameters = {
    "do_train": True,
    "n_iters": 50, #the number of training iterations (epochs)
    "print_every": 1, #the number of epochs after which to print the current training status
    "plot_every": 1, #the number of epochs after which to plot the current training results
    "batch_size": 32, #the number of samples in one forward/backward pass
    "model_name": "Resnet18",
    "save_dir": "/home/GK/Datacenter_storage/GK/SleepApnoea/Code/OSA_PD_Results", #the directory to save the trained model and its results
    "metric_name": "precision", #the name of the metric to evaluate the model performance
    "lr": 0.001, #small positive value used to update the model parameters
    "step_size": 7,
    # step size for learning rate
    "maximize_metric": True,
    "patience": 7, #the number of epochs to wait for improvement in the metric_name before early stopping
    "early_stop": False,
    "prev_val_loss": 1e10, #large positive value representing the initial value of the validation loss before the training starts
    "num_workers": 0, #the number of worker threads to use for loading the data.
    "data_dir": "/media/Datacenter_storage/GK/SleepApnoea/Data/",
    "train_file": "PreporcessIMG_A_Train_Binary.csv", 
    "test_file": "PreporcessIMG_Test_Binary.csv",
    "val_file": "PreporcessIMG_A_Val_Binary.csv"
}
