# basicNeuralNetwork-for-Java
Writing a Neural Network (Dense Network) from Scratch in Java


## Creating Network/Loading Network from file
```Java
//Create a Simple Dense Network
NNetwork myNetwork = new NNetwork();
myNetwork.addLayer(new Layer(2 , ActivationTypes.Sigmoid));
myNetwork.addLayer(new Layer(200 , ActivationTypes.Sigmoid));
myNetwork.addLayer(new Layer(200 , ActivationTypes.Sigmoid));
myNetwork.addLayer(new Layer(26 , ActivationTypes.Softmax));
//----------------------------
//Loading Network from file
myNetwork.QloadModel("model2.panic"); //QloadModel is faster but needs more space
myNetwork.loadModel("model.panic"); //loadModel is slower(a lot!) but needs half space
```
## Getting Network propertiese printed on console
```Java
myNetwork.getSummary(false);
```
