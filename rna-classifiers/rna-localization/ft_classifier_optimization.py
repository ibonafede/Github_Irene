import datetime
import os

import util.fast_text as ft
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK

FILE_TRAIN = 'data/seqlist/training_set.csv'
FILE_VALID = 'data/seqlist/valid_set.csv'
FILE_TEST = 'data/seqlist/test_set.csv'

def datetime_now():
    return "["+str(datetime.datetime.now())+"] "

def print_results(N, p, r, k=1):
    print(datetime_now(), "Results: ", "N\t" + str(N) + ";", "P@{} {:.3f}".format(k, p) + ";", "R@{} {:.3f}".format(k, r))

def objective_function(configuration, return_model=False):
    """ Train network
    """

    print(datetime_now(), "Configuration: ", configuration)
    
    # NB: If you want same training results using same config choose thread=1, otherwise the results can ben quite different
    model = tf.train_supervised(input=FILE_TRAIN, label=u'__label__', thread=12,
        epoch=configuration['epoch'],
        lr=configuration['learning_rate'],
        dim=configuration['dimension'],
        ws=configuration['window_size'],
        wordNgrams=configuration['word_ngrams'],
        neg=configuration['neg']
        )

    samples_number, precision, recall = model.test(FILE_VALID, k=1)
    print_results(samples_number, precision, recall)

    if return_model:
        return {'loss': -precision, 'status': STATUS_OK}, model

    return {'loss': -precision, 'status': STATUS_OK }

if __name__ == "__main__":
    space = {
        'epoch': hp.choice('epoch_', [100, 200, 300, 400, 500, 600, 700]),
        'learning_rate': hp.choice('learning_rate_', [0.1, 0.3, 0.5, 0.7, 1]),
        'dimension': hp.choice('dimension_', [10, 20, 50, 70, 100, 150, 200, 250, 300]),
        'window_size': hp.choice('window_size_', range(5, 15, 1)),
        'word_ngrams': hp.choice('word_ngrams_', range(1, 5, 1)),
        'neg': hp.choice('neg_', range(1, 5, 1)),
    }

    # Optimization
    #   - Tree of Parzen Estimators (TPE): hyperopt.tpe.suggest
    #   - Random Search: hyperopt.random.suggest
    best = fmin(fn=objective_function, space=space, algo=tpe.suggest, max_evals=2) #max_evals=300

    # Create best model
    print(datetime_now(), "Best (array): ", best)

    configuration_best = space_eval(space, best)
    print(datetime_now(), "Best (configuration): ", configuration_best)

    print(datetime_now(), "Training best config...")
    _, model = objective_function(configuration_best, return_model=True)
    model.save_model('rna_localization_optimized.bin')

    print(datetime_now(), "Test set evaluation")
    samples_number, precision, recall = model.test(FILE_TEST, k=1)
    print_results(samples_number, precision, recall)