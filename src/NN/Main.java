package NN;

import java.io.IOException;
import java.io.InputStream;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws Exception {
        Double[][] input = new Double[][]{{0.0,0.0},{0.0,1.0} , {1.0,0.0} , {1.0,1.0}};
        Double[][] outPut = new Double[][]{{0.0,1.0},{0.0,1.0},{0.0,1.0},{1.0,0.0}};
        TrainingData data = new TrainingData(new Matrix(input).Transpose() , new Matrix(outPut).Transpose());
        NNetwork myModel = new NNetwork(0.9,0.000001);
        myModel.addLayer(new Layer(2 , ActivationTypes.Sigmoid));
        myModel.addLayer(new Layer(40 , ActivationTypes.Sigmoid));
        myModel.addLayer(new Layer(50 , ActivationTypes.Sigmoid));
        myModel.addLayer(new Layer(50 , ActivationTypes.Sigmoid));
        myModel.addLayer(new Layer(50 , ActivationTypes.Sigmoid));


        myModel.addLayer(new Layer(2 , ActivationTypes.Softmax));
        //myModel.QloadModel("model2.panic");
        //Matrix pred = myModel.feedForward(data);
        //(pred.Transpose()).printMatrix();



        myModel.getSummary(false);

        int epochs = 10000;

        myModel.Fit(data , 100*epochs,"model2.panic");
        //myModel.QsaveModel("model2.panic");
    }
}
