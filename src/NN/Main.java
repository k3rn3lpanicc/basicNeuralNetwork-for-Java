package NN;

public class Main {
    public static void main(String[] args) throws Exception {
        NNetwork myNetwork = new NNetwork();
        myNetwork.addLayer(new Layer(2 , ActivationTypes.Sigmoid));
        myNetwork.addLayer(new Layer(200 , ActivationTypes.Sigmoid));
        myNetwork.addLayer(new Layer(200 , ActivationTypes.Sigmoid));
        myNetwork.addLayer(new Layer(100 , ActivationTypes.Sigmoid));
        myNetwork.addLayer(new Layer(50 , ActivationTypes.Sigmoid));
        myNetwork.addLayer(new Layer(26 , ActivationTypes.Softmax));
        myNetwork.QloadModel("model2.panic");
        myNetwork.getSummary(false);
        Double[][] input = new Double[][]{{2.0 , 1.0},{3.0, 2.0} , {4.0,1.0} , {2.0,0.0}};
        Matrix myMatrix = new Matrix(input);
        myMatrix.printMatrix();
        TrainingData data = new TrainingData(input , new Double[][]{{1.0},{0.0},{0.0},{0.0}});
        long lasttime = System.nanoTime();
        Double[][] Output = myNetwork.feedForward(data);
        for(int i = 0; i<Output.length;i++) {
            for(int j = 0; j<Output[0].length;j++)
                System.out.print(Output[i][j] + "  ");
            System.out.println();
        }
        System.out.println();
        System.out.println("time it took to calculate : "+String.format("%.4f",(1e-6 * (System.nanoTime()-lasttime)))+"MS");
        //myNetwork.saveModel("model.panic");
        //myNetwork.QsaveModel("model2.panic");

    }
}
