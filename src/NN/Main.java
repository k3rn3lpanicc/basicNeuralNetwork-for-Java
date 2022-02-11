package NN;

public class Main {
    public static void main(String[] args) throws Exception {
        NNetwork myNetwork = new NNetwork();
        myNetwork.addLayer(new Layer(2 , ActivationTypes.Sigmoid));
        myNetwork.addLayer(new Layer(30 , ActivationTypes.Sigmoid));
        myNetwork.addLayer(new Layer(20 , ActivationTypes.Sigmoid));
        myNetwork.addLayer(new Layer(2 , ActivationTypes.Softmax));
        //myNetwork.QloadModel("model2.panic");
        myNetwork.getSummary(false);
        Double[][] input = new Double[][]{{2.0 , 1.0},{3.0, 2.0} , {4.0,1.0} , {2.0,0.0}};
        Double[][] outPut = new Double[][]{{1.0},{0.0},{0.0},{0.0}};
        TrainingData data = new TrainingData(new Matrix(input) , new Matrix(outPut));

        long lasttime = System.nanoTime();
        Matrix Output = myNetwork.feedForward(data);
        System.out.println("time it took to calculate : "+String.format("%.4f",(1e-6 * (System.nanoTime()-lasttime)))+" milli seconds");
        Output.printMatrix();

        //myNetwork.saveModel("model.panic");
        //myNetwork.QsaveModel("model2.panic");

    }
}
