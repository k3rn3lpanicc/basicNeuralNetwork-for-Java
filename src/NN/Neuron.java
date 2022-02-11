package NN;

import java.io.Serializable;
import java.util.concurrent.ExecutionException;


public class Neuron implements Serializable {
    ActivationTypes Activation;
    Neuron(ActivationTypes Activation){
        this.Activation = Activation;
    }
    double calculateSigmoid(double Value){
        return 1/(1+Math.exp(Value));
    }
    double calculateRelu(double Value){
        return Math.max(0,Value);
    }
    private double addActivation(double Value,ActivationTypes Activation) throws Exception{
        switch (Activation){
            case Linear:
                return Value;
                //break;
            case Sigmoid:
                return calculateSigmoid(Value);
                //break;
            case Relu:
                return calculateRelu(Value);
                //break;
            case Softmax:
                return Value;
            default:
                throw new Exception("Activation Function Not Found");

        }
    }
    public double Calculate(Double[] previousLayerValues , Double[][] previousLayerWeights, int ii) throws Exception{
        if(previousLayerValues.length!=previousLayerWeights.length){
            throw new Exception("The Input Weights and values are not same size!");
        }
        else{
            double calculatedValue = 0;
            for(int i = 0; i<previousLayerValues.length; i++){
                calculatedValue+= ((previousLayerValues[i])*(previousLayerWeights[i][ii]));
            }
            return addActivation(calculatedValue , Activation);

        }
    }
}
