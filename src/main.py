import network
import mnist_loader
import drawer


if(__name__ == "__main__"):
    #set up data using the load data file
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    #set up ai with contructor call
    #AI_network = network.Network([784, 30, 10])
    AI_network = network.Network.load_with_metadata('mnist_network_20260307_214327.pkl')


    #begin the learning process using the stochastic gradient descent function in the ai class
    #AI_network.stochasticGradientDescent(training_data, 30, 10, 3.0, test_data=test_data)

    drawWindow = drawer.DigitDrawer(AI_network)
    drawWindow.run()

    #save data for later use
    # accuracy = AI_network.evaluate(test_data)
    # print(f"Accuracy: {accuracy}/{len(test_data)}")
    # AI_network.save_with_metadata(test_accuracy=accuracy)