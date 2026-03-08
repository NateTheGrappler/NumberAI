import network
import mnist_loader


if(__name__ == "__main__"):
    #set up data using the load data file
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    #set up ai with contructor call
    AI_network = network.Network([784, 30, 10])

    #begin the learning process using the stochastic gradient descent function in the ai class

    AI_network.stochasticGradientDescent(training_data, 30, 10, 3.0, test_data=test_data)