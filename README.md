# dog_classifier
The goal of the project is to classify the breed of dog within 133 genres with the power of transfer learning. In this project, the 4 state of art models in the field of computer vision ResNet50, ResNet50V2, Xception and InceptionV3 whose weights have been trained on [ImageNet](http://www.image-net.org) are introduced.


## Usage
1. Install all the required dependencies 
```shell
$ pip install -r requirements.txt 
```
2. Download the [pre-trained models](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) to you folder
3. run the code and start classifying the dog breed
```shell
$ python test.py -p <image path> -n <name of outcome file> -m <model>
```

## Train
You can also train the models your self. Please follow the steps below:

1. The first thing you have to di is to download the dataset from the [link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
2. Run the code 
```shell=
$ python main.py
```
## Reference
https://medium.com/@psiodoros/dog-breed-classification-using-cnns-b065913527c

