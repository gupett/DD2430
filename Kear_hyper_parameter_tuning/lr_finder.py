import numpy as np
import matplotlib.pyplot as plt

import random

# Class for approximately find the bes learning rate for a given model
# Based on: https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
# and https://arxiv.org/abs/1506.01186
class lr_finder:
    def __init__(self, model, training_X, training_y, val_X, val_y, lr_range=(0.0000001,0.1), lr_samples=20, batch_size=20):
        # The model for which we should find the best learning rate
        self.model = model
        # The data used to fit and evaluate the model
        self.training_X = training_X
        self.training_y = training_y
        self.val_X = val_X
        self.val_y = val_y
        self.batch_size = batch_size
        # Parameters for choosing which parameters lr_parameters to test
        self.lr_parameters = self.list_of_lr_parameters(lr_range, lr_samples)


    # A list of exponentialy growing learning rates is returned between the specified start and end range
    # of the same length as lr_samples specifies
    def list_of_lr_parameters(self, lr_range, lr_samples):
        lr_start = lr_range[0]
        lr_end = lr_range[1]

        # Calculate the base of the exponential so that there will be lr_samples between lr_start and lr_end
        # lr_start * base^(lr_samples-1) = lr_end | (lr_samples-1) is used because a list will star with index zero
        base = np.power((lr_end/lr_start), (1/(lr_samples-1)))

        # Calculate the values of every lr which should be tested
        lr_parameters = [lr_start * np.power(base, i) for i in range(lr_samples)]

        return lr_parameters


    def find_start_lr(self, plot=True):

        # Store the weights of the model so that the same model is used for testing all the different learning rates
        #model_weights = self.model.get_weights()
        loss_for_lr = []
        for lr in self.lr_parameters:

            #model_history = self.model.fit(self.trainingX, self.trainingY, epochs=1, batch_size=self.batch_size, verbose=0)
            #loss = model_history['loss']
            loss = random.random()
            loss_for_lr.append(loss)
            # Restore the model to the same weights as it had before the lr test
            #self.model.set_weights(model_weights)

        # Calculate the delat loss
        delta_loss = []
        avg = 3
        for i in range(avg, len(loss_for_lr)):
            s_loss =
            d_loss =


        if plot:
            # Plotting the loss for each learning rate
            plt.plot(self.lr_parameters, loss_for_lr, 'ro')

            # plotting the change in loss for each learning rate

            plt.show()

if __name__ == '__main__':
    LR_finder = lr_finder('3', '4', '4', '4', '4')
    LR_finder.find_start_lr()

    #lr_parameters = list_of_lr_parameters(lr_range=(0.00001,1), lr_samples=20)
    #print(lr_parameters)