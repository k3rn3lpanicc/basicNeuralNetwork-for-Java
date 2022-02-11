# BasicNeuralNetwork for Java
Very Simple and dense Neural Network from Scratch with Java


## Creating Network/Loading Network from file
```Java
//Create a Simple Dense Network
NNetwork myNetwork = new NNetwork();
myNetwork.addLayer(new Layer(2 , ActivationTypes.Sigmoid));
myNetwork.addLayer(new Layer(200 , ActivationTypes.Sigmoid));
myNetwork.addLayer(new Layer(200 , ActivationTypes.Sigmoid));
myNetwork.addLayer(new Layer(1 , ActivationTypes.Softmax));
//----------------------------
//Loading Network from file
myNetwork.QloadModel("model2.panic"); //QloadModel is faster but needs more space
myNetwork.loadModel("model.panic"); //loadModel is slower(a lot!) but needs half space
```
## Getting Network properties printed on console
```Java
myNetwork.getSummary(false); //false means we don't want to print the weights
```
## Set training data/Labels and do feed-forward to get the prediction of network
```Java
Double[][] input = new Double[][]{{2.0 , 1.0},{3.0, 2.0} , {4.0,1.0} , {2.0,0.0}};
Double[][] outPut = new Double[][]{{1.0},{0.0},{0.0},{0.0}}; //the labels
TrainingData data = new TrainingData(input , outPut);
Double[][] Output = myNetwork.feedForward(data); //predict results

```
## Saving trained Model
```Java
myNetwork.saveModel("model.panic"); //this method uses half space in compare to second method but it's very slow
myNetwork.QsaveModel("model2.panic"); //this method is fast but uses more space to save
```

## In future
-Back propagation (training part)

-More Customizeable networks

-CNN and RNN models

