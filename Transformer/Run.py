from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import wandb
import numpy as np
import csv

"""parameters to select model and tricks"""
# 1 "bert", 2 "roberta", 3 "xlnet"
model_sequence = 1
# use sweep
is_sweep = False
# use early stop
is_early_stop = False


# Preparing train data
data_train_neg = pd.read_csv('../datasets/train_neg_full.txt', sep='\r', names=["text"], header=None, encoding='utf-8')
data_train_neg.insert(data_train_neg.shape[1], "labels", 0)

data_train_pos = pd.read_csv('../datasets/train_pos_full.txt', sep='\r', names=["text"], header=None, encoding='utf-8')
data_train_pos.insert(data_train_pos.shape[1], "labels", 1)

data_train = pd.concat([data_train_neg, data_train_pos], ignore_index=True)
data_train = data_train.sample(frac=1).reset_index(drop=True)

data_eval = data_train.iloc[2400000:-1, :]
data_eval = data_eval.reset_index(drop=True)
data_train = data_train.iloc[0:2400000, :]

# load the test data
data_test = []
with open("../datasets/test_data.txt", "r", encoding='utf-8', ) as filestream:
    for line in filestream:
        currentline = line.split(",", 1)
        data_test.append(currentline[1].strip())


def run():

    model_args = ClassificationArgs()

    if is_early_stop:
        # use early stop
        model_args.use_early_stopping = True
        model_args.early_stopping_delta = 0.01
        model_args.early_stopping_metric = "mcc"
        model_args.early_stopping_metric_minimize = False
        model_args.early_stopping_patience = 3
        model_args.evaluate_during_training_steps = 2000

    if is_sweep:
        # Decide hyperparameter
        sweep_config = {
            "method": "bayes",  # based on a Gaussian Process
            "metric": {"name": "train_loss", "goal": "minimize"}, # set the metric
            "parameters": {
                "num_train_epochs": {"values": [2, 3, 4]},  # sweep number of epochs
                "learning_rate": {"min": 5e-6, "max": 1e-4}, # sweep learning rate
            },
        }
        # create a sweep project
        sweep_id = wandb.sweep(sweep_config, project="Sweep hyper-parameter")
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)

        # Define parameters for the model
        model_args.reprocess_input_data = True
        model_args.overwrite_output_dir = True
        model_args.evaluate_during_training = True
        model_args.manual_seed = 4
        model_args.use_multiprocessing = True
        model_args.train_batch_size = 32
        model_args.eval_batch_size = 16
        model_args.wandb_project = "Sweep hyper-parameter"

        def train():
            # Initialize a new wandb run
            wandb.init()
            # Create a TransformerModel
            model = ClassificationModel(
                "bert",
                "bert-base-cased",
                use_cuda=True,
                args=model_args,
                sweep_config=wandb.config,
            )
            # Train the model
            model.train_model(data_train, eval_df=data_eval)
            # Evaluate the model
            model.eval_model(data_eval)
            # Sync wandb
            wandb.join()
        wandb.agent(sweep_id, train)

    else:

        if model_sequence == 1:
            model_args.save_steps = -1 # the interval to save the model (-1 means don't save)
            model_args.num_train_epochs = 2 # number of epochs, given by sweep
            model_args.train_batch_size = 32 # batch size
            model_args.learning_rate = 1.96e-5 # learning rate, given by sweep
            model_args.overwrite_output_dir = True
            model = ClassificationModel("bert", "bert-base-cased", use_cuda=True, args=model_args) # create a BERT model
            # Train the model
            model.train_model(data_train, eval_df=data_eval)
            # Evaluate the model
            model.eval_model(data_eval)
        elif model_sequence == 2:
            model_args.save_steps = -1 # the interval to save the model (-1 means don't save)
            model_args.num_train_epochs = 2 # number of epochs, given by sweep
            model_args.train_batch_size = 32 # batch size
            model_args.learning_rate = 1.96e-5 # learning rate, given by sweep
            model_args.overwrite_output_dir = True
            model = ClassificationModel("roberta", "roberta-base", use_cuda=True, args=model_args) # create a RoBERTa model
            # Train the model
            model.train_model(data_train, eval_df=data_eval)
            # Evaluate the model
            model.eval_model(data_eval)
        elif model_sequence == 3:
            model_args.save_steps = -1
            model_args.num_train_epochs = 2
            model_args.train_batch_size = 32
            model_args.learning_rate = 1.96e-5
            model_args.overwrite_output_dir = True
            model = ClassificationModel("xlnet", "xlnet-base-cased", use_cuda=True, args=model_args) # create a XLNet model
            # Train the model
            model.train_model(data_train, eval_df=data_eval)
            # Evaluate the model
            model.eval_model(data_eval)




    # Make predictions with the model
    predictions, raw_outputs = model.predict(data_test)

    predict_res = np.array(predictions)
    # convert from {0,1} to {-1,1}
    predict_res[np.where(predict_res == 0)] = -1
    test_id = np.arange(1, 10001)

    # generate the result .csv file
    with open('../prediction.csv', 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(test_id, predict_res):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})

if __name__ == "__main__":
    run()