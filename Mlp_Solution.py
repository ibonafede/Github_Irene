import common
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


# ------------------------ PARAMS ---------------------------------
train_file_path = ...
validation_file_path = ...
test_file_path= ...
feature_count = 25
random_seed = 1
max_epochs = 100 
# -----------------------------------------------------------------
    
mlp = MLPClassifier(solver='adam',learning_rate_init=0.03, warm_start=True, max_iter=1, random_state=random_seed)
#MLPClassifier( hidden_layer_sizes=(100, ),
#                activation=’relu’,
#                solver=’adam’,
#                alpha=0.0001,
#                batch_size=’auto’,
#                learning_rate=’constant’,
#                learning_rate_init=0.001,
#                power_t=0.5,
#                max_iter=200,
#                shuffle=True,
#                random_state=None,
#                tol=0.0001,
#                verbose=False,
#                warm_start=False,
#                momentum=0.9,
#                nesterovs_momentum=True,
#                early_stopping=False,
#                validation_fraction=0.1,
#                beta_1=0.9,
#                beta_2=0.999,
#                epsilon=1e-08)

# carica i dataset di train e validation/test
train_x, train_y = common.load_labeled_dataset(train_file_path, feature_count)
validation_x, validation_y = common.load_labeled_dataset(validation_file_path, feature_count)
test_x, test_y = common.load_labeled_dataset(test_file_path, feature_count)
train_patterns = train_x.shape[0]
validation_patterns = validation_x.shape[0]
test_patterns = test_x.shape[0]
print('Train: ', train_file_path, ' (', train_patterns,' patterns)')
print('Validation: ', validation_file_path, ' (', validation_patterns, ' patterns)')
print('Test: ', test_file_path, ' (', test_patterns, ' patterns)')
print()

time_start = time.time()
print('Training ...')

epochs_training_loss = []
epochs_validation_accuracy = []
epochs_training_accuracy = []
epochs_test_accuracy = []
for i in range(max_epochs):
    mlp.fit(train_x, train_y)
    epochs_training_loss.append(mlp.loss_)  
    epochs_training_accuracy.append(mlp.score(train_x, train_y) * 100)
    epochs_validation_accuracy.append(mlp.score(validation_x, validation_y) * 100)
    epochs_test_accuracy.append(mlp.score(test_x, test_y) * 100)
    print("Epoch %2d: Loss = %5.4f, TrainAccuracy = %4.2f%%, ValidAccuracy = %4.2f%%, TestAccuracy = %4.2f%%" % (i, epochs_training_loss[-1], epochs_training_accuracy[-1], epochs_validation_accuracy[-1],epochs_test_accuracy[-1])) 
         
max_valacc_idx = np.array(epochs_validation_accuracy).argmax()
print('Max Accuracy on Validation = %4.2f%% at Epoch = %2d Accuracy on Test =  %4.2f%%' % (epochs_validation_accuracy[max_valacc_idx], max_valacc_idx,epochs_test_accuracy[max_valacc_idx])) 
print('Total Time: %.2f sec' % (time.time() - time_start)) 

# Plot dei risultati
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_ylim(0,epochs_training_loss[0] * 1.1)
ax2.set_ylim(85,100)
ax1.plot(range(0,len(epochs_training_loss)), epochs_training_loss, 'r')
ax2.plot(range(0,len(epochs_training_accuracy)), epochs_training_accuracy, 'b')
ax2.plot(range(0,len(epochs_validation_accuracy)), epochs_validation_accuracy, 'g')
ax2.plot(range(0,len(epochs_test_accuracy)), epochs_test_accuracy, 'c')
common.legend_drawing(1,('r','b','g','c'),("Loss","Training Accuracy","Validation Accuracy","Test Accuracy"),0)
plt.show() 