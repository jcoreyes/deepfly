# flyvfly
Attempt to use neural networks for classifying socal actions of fruit flies from pose tracking data using
the [fly-vs-fly](http://www.vision.caltech.edu/Video_Datasets/Fly-vs-Fly/index.html) dataset.
I used [neon](https://github.com/NervanaSystems/neon) for training the neural networks.

## Training
First download the fly-vs-fly dataset and install Neon. Model setup and training parameters are set 
in train.py. The dataset classes for training a single class and multiclass model are in flyvfly.py and 
flyvflymulticlass.py respectively. Specify the path of the dataset in each of these files. 
To train a single class model, import flyvfly as as Fly and set
NUM_CLASSES constant to 1. Set the action number in flyvfly.py. To train a multiclass model, import
flyvflymulticlass as Fly and set the NUM_CLASSES constant to 5. Run train.py with the output save file
for the model as the first input argument.

## Testing
Default videos for training are 1-5 and for testing are 6-10. Testing a single class model is done
in test.py and testing a multiclass model is done in test_multiclass.py where the input argument
to each is the name of the saved model file. The saved model file should end in a two digit number
(00 to 99) for labeling the plots though this is not necessary.