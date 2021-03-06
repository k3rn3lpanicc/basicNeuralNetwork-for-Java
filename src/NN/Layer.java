package NN;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Layer implements Serializable {
    List<Neuron> Neurons = new ArrayList<>();
    ActivationTypes Activation;
    //Double Bias;
    Layer(int neuronNo , Double Bias , ActivationTypes Activation){
        for(int i = 0; i < neuronNo; i++){
            Neuron neuron = new Neuron(Activation);
            Neurons.add(neuron);
        }
        this.Activation = Activation;
        //this.Bias = Bias;
    }
    Layer(int neuronNo , ActivationTypes Activation){
        for(int i = 0; i < neuronNo; i++){
            Neuron neuron = new Neuron(Activation);
            Neurons.add(neuron);
        }
        this.Activation = Activation;
        //this.Bias = new Random().nextGaussian();
    }
    double calculateSigmoid(double Value){
        return 1/(1+Math.exp(Value));
    }
    void applySigmoid(Matrix input){
        for(int i = 0; i< input.getFirst();i++){
            for(int j = 0; j< input.getSecond();j++){
                //input.printMatrix();

                input.setCell(i,j,calculateSigmoid(input.getCell(i,j)));
            }
        }
    }
    double calculateRelu(double Value){
        return Math.max(0,Value);
    }
    void applyRelu(Matrix input){
        for(int i = 0; i< input.getFirst();i++){
            for(int j = 0; j< input.getSecond();j++){
                input.setCell(i,j,calculateRelu(input.getCell(i,j)));
            }
        }
    }
    Double[] SoftmaxApplied(Double[] inp){
        Double[] result = new Double[inp.length];
        Double maximum = 0.0;
        for(int i = 0; i < inp.length; i++)
            if(maximum<inp[i])
                maximum = inp[i];
        for(int i = 0; i < inp.length; i++)
            inp[i]-=maximum;
        Double div = 0.0;
        for(int i = 0; i<inp.length;i++)
            div+= Math.exp(inp[i]);
        for(int i = 0 ;i<inp.length;i++)
            result[i] = Math.exp(inp[i])/div;
        return result;
    }
    void SoftmaxApplied(Matrix inp){
        for(int k = 0; k<inp.getFirst();k++) {
            inp.getData()[k] = SoftmaxApplied(inp.getData()[k]);
        }
    }
    void applyActivation(Matrix inp,ActivationTypes activation){
            switch (activation){
                case Linear:
                    return;
                case Sigmoid:
                    applySigmoid(inp);
                    break;
                case Relu:
                    applyRelu(inp);
                    break;
                case Softmax:
                     SoftmaxApplied(inp);
                    break;
            }
    }
    /*
    Matrix calculate(Matrix Input , Matrix Weights,Double previousBias) throws Exception {
        Matrix result = new Matrix(new Double[Input.getFirst()][Input.getSecond()]);
        result.setData(Input.getData().clone());
        result.Multiply(Weights);
        result.Plus(previousBias);
        applyActivation(result,Activation);
        return result;
    }
    */

}
