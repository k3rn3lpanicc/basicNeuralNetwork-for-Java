package NN;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Layer implements Serializable {
    List<Neuron> Neurons = new ArrayList<>();
    Double biasWeight;
    ActivationTypes Activation;
    Layer(int neuronNo , ActivationTypes Activation){
        for(int i = 0; i < neuronNo; i++){
            Neuron neuron = new Neuron(Activation);
            Neurons.add(neuron);
        }
        this.Activation = Activation;
    }
    Double[] calculate(Double[] Input , Double[][] Weights) throws Exception {
        Double[] result = new Double[Neurons.size()];
        for(int i = 0 ; i < Neurons.size() ; i++) {
            result[i] = Neurons.get(i).Calculate(Input, Weights,i);
        }
        return result;
    }
}
