package NN;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;
import java.nio.charset.StandardCharsets;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.WatchEvent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
enum ActivationTypes{
    Linear,
    Sigmoid,
    Relu,
    Softmax

}
public class NNetwork implements Serializable {
    List<Layer> Layers = new ArrayList<>();
    List<Matrix> Weights = new ArrayList<>();
    private List<Matrix> VdW= new ArrayList<>();
    private List<Matrix> VdB= new ArrayList<>();
    private List<Matrix> Bias = new ArrayList<>();
    Double Beta;
    Double learningRate;
    NNetwork(Double Beta , Double learningRate){
        this.Beta = Beta;
        this.learningRate = learningRate;
    }
    public static String ActivationTypeToString(ActivationTypes type) throws Exception {
        switch (type){
            case Linear:
                return "Linear";
            case Sigmoid:
                return "Sigmoid";
            case Softmax:
                return "Softmax";
            case Relu:
                return "Relu";
            default:
                throw new Exception("Activation not found!");
        }
    }

    Matrix feedForward(TrainingData data) throws Exception {
        Matrix Z;//new Matrix(new Double[data.InputData.getFirst()][data.InputData.getSecond()]);
        Z = Matrix.Multiply(Matrix.Copy(Weights.get(0)),data.InputData);
        Z.PlusBias(Bias.get(0));
        applyActivation(Z , Layers.get(0).Activation);
        for(int i  = 1 ; i<Layers.size()-1 -1;i++){
            //Z=np.dot(W[i],A)+b[i]
            Z = Matrix.Copy(Z);
            Z = Matrix.Multiply(Weights.get(i),Z);
            Z.PlusBias(Bias.get(i));
            applyActivation(Z , Layers.get(i).Activation);
        }
        Matrix kk = Matrix.Multiply(Weights.get(Weights.size()-1),Z);
        kk.PlusBias(Bias.get(Bias.size()-1));
        Matrix lol = Matrix.Copy(kk).Transpose();
        applyActivation(lol , ActivationTypes.Softmax);
        return lol.Transpose();
    }
    Matrix deriv(ActivationTypes activation , Matrix z) throws Exception {
        if(activation == ActivationTypes.Sigmoid){
            return Matrix.Multiply2(z,Matrix.Multiply(Matrix.Minus(z,1.0), -1.0));
        }
        Matrix result = Matrix.Copy(z);
        if(activation == ActivationTypes.Relu) {
            for (int i = 0; i < result.getFirst(); i++)
                for (int j = 0; j < result.getSecond(); j++)
                    result.setCell(i, j, result.getCell(i, j) <= 0 ? 0.0 : 1.0);
            return result;
        }
        else{
            return null;
        }
    }
    Double[][] createWeights(int thisLayer , int thatLayer){
        Double[][] weights = new Double[thisLayer][thatLayer];
        for(int i = 0; i<thisLayer ;i++){
            for(int j = 0 ; j< thatLayer ; j++)
                weights[i][j] = new Random().nextGaussian();
        }
        return weights;
    }
    void addWeights(Double[][] weights){
        Weights.add(new Matrix(weights));
        VdW.add(Matrix.Zeros(weights.length,weights[0].length));
        VdB.add(Matrix.Zeros(weights.length,1));
        Bias.add(Matrix.getRandom(weights.length,1));
    }
    public void addLayer(Layer newLayer){
        Layers.add(newLayer);
        if(Layers.size()>=2) {

            addWeights(Matrix.Multiply(new Matrix(createWeights(newLayer.Neurons.size(), Layers.get(Layers.size() - 2).Neurons.size())),Math.sqrt(2/Layers.get(Layers.size()-1).Neurons.size())).getData());

        }
    }
    public void addLayer(int neuronsNo , Double Bias , ActivationTypes Activation){
        Layer newLayer = new Layer(neuronsNo , Bias , Activation);
        Layers.add(newLayer);
        if(Layers.size()>=2) {
            addWeights(Matrix.Multiply(new Matrix(createWeights(newLayer.Neurons.size(), Layers.get(Layers.size() - 2).Neurons.size())),Math.sqrt(2/Layers.get(Layers.size()-1).Neurons.size())).getData());

        }
    }

    int calculateTrainableParameters(){
        int ans = 0;
        for(int i = 0; i < Layers.size()-1;i++){
            ans+= (Layers.get(i).Neurons.size())*(Layers.get(i+1).Neurons.size());
        }
        return ans;
    }
    public List<Layer> getLayers() { return Layers; }
    public void setLayers(List<Layer> layers) { Layers = layers; }
    public List<Matrix> getWeights() { return Weights; }
    public void setWeights(List<Matrix> weights) {
        Weights = weights;
    }
    void QsaveModel(String fileName){
        System.out.println("Saving...");
        GsonBuilder builder = new GsonBuilder();
        builder.setPrettyPrinting();
        Gson gson = builder.create();
        Path path = Paths.get(fileName);
        String someString = gson.toJson(this);
        byte[] bytes = someString.getBytes();
        try {
            Files.write(path, bytes);
            System.out.println("Saved!");
        } catch (IOException ex) {
           //TODO
        }
    }
     void QloadModel(String fileName){
         System.out.println("Loading Model ...");
        Gson gson = new Gson();
        try (Reader reader = new FileReader(fileName)) {
            NNetwork model = gson.fromJson(reader, NNetwork.class);
            this.Layers = model.Layers;
            this.Weights = model.Weights;
            this.VdW = model.VdW;
            this.VdB = model.VdB;
            System.out.println("Loaded!");
        } catch (IOException e) { e.printStackTrace(); }
    }
    void saveModel(String fileName) throws IOException {
        System.out.println("Saving...");
        FileOutputStream fileOutputStream = new FileOutputStream(fileName);
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
        objectOutputStream.writeObject(this);
        objectOutputStream.flush();
        objectOutputStream.close();
        System.out.println("Saved!");
    }
    void loadModel(String fileName) throws IOException, ClassNotFoundException {
        System.out.println("Loading Model ...");
        FileInputStream fileInputStream = new FileInputStream(fileName);
        ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
        NNetwork p2 = (NNetwork) objectInputStream.readObject();
        objectInputStream.close();
        this.Weights = p2.Weights;
        this.VdW = p2.VdW;
        this.VdB = p2.VdB;
        this.Layers = p2.Layers;
        System.out.println("Loaded!");
    }
    public void getSummary(boolean printWeights) throws Exception {
        System.out.println("-----------------Network Properties-----------------");
        System.out.println("Number of Layers : "+Layers.size());
        int cnt = 0;
        for(Layer layer : Layers){
            System.out.println("Layer "+cnt+" : "+layer.Neurons.size()+" Neurons with Activation Function of : "+ ActivationTypeToString(layer.Activation));
            cnt++;
        }
        System.out.println("number of Trainable Parameters : "+calculateTrainableParameters());
        if(!printWeights)
            return;
        System.out.println("Weights : ");
        int cntr = 0;
        for(Matrix weight : Weights){
            System.out.println("=========Layer "+cntr+"-"+(cntr+1)+" : ");
            cntr++;
            for(int i = 0;i<weight.getData().length;i++) {
                for (int j = 0; j < weight.getData()[i].length; j++) {
                    System.out.print(weight.getData()[i][j] + " ");
                }
                System.out.println();
            }
            System.out.println();
        }
    }
    Double Cost(TrainingData data) throws Exception {
        Matrix p = feedForward(data);
        Double cost = 0.0;
        int exmpleno = data.InputData.getSecond();
        int classno = data.YLabels.getFirst();
        for(int i = 0 ; i< exmpleno;i++){
            for(int j = 0 ; j < classno;j++){
               // System.out.print(cost+"+ (-"+data.YLabels.getCell(i,j)+"*log("+(p.getCell(i,j))+")) = ");
                cost += (-data.YLabels.getCell(j,i)*Math.log(p.getCell(j,i)));
               // System.out.println(cost);
            }
        }
        return cost;
    }
    Matrix rounded(Matrix input){
        Matrix result = new Matrix(new Double[input.getFirst()][input.getSecond()]);
        for(int i = 0; i<input.getFirst();i++)
            for(int j = 0; j<input.getSecond();j++)
                result.setCell(i,j, (double) Math.round(input.getCell(i,j)));
        return result;
    }
    static boolean areEqual(Double[] n1 , Double[] n2) throws Exception {
        if(n1.length!=n2.length)
            throw new Exception("arrays are not same size!");
        for(int i = 0; i<n1.length;i++){
            if(!n1[i].equals(n2[i]))
                return false;
        }
        return true;
    }
    Double Accuracy(TrainingData data) throws Exception {
        Matrix pred = rounded(feedForward(data).Transpose());
        Matrix YS = Matrix.Copy(data.YLabels.Transpose());
        int cnt = 0;
        for(int i=0;i<YS.getFirst();i++){
            if(areEqual(pred.getData()[i] , YS.getData()[i]))
                    cnt++;
        }
        return (double)cnt/(double)YS.getFirst();
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

    void backPropogate(TrainingData data) throws Exception {
        List<Matrix> As = new ArrayList<>();
        Double m = (double)data.InputData.getFirst();
        Matrix Z;//new Matrix(new Double[data.InputData.getFirst()][data.InputData.getSecond()]);
        Z = Matrix.Multiply(Matrix.Copy(Weights.get(0)),data.InputData);
        Z.PlusBias(Bias.get(0));
        applyActivation(Z , Layers.get(0).Activation);
        As.add(Matrix.Copy(Z));
        for(int i  = 1 ; i<Layers.size()-1 -1;i++){
            //Z=np.dot(W[i],A)+b[i]
            Z = Matrix.Copy(Z);
            Z = Matrix.Multiply(Weights.get(i),Z);
            Z.PlusBias(Bias.get(i));
            applyActivation(Z , Layers.get(i).Activation);
            As.add(Matrix.Copy(Z));
        }
        Matrix kk = Matrix.Multiply(Weights.get(Weights.size()-1),Z);
        kk.PlusBias(Bias.get(Bias.size()-1));
        Matrix lol = Matrix.Copy(kk).Transpose();
        applyActivation(lol , ActivationTypes.Softmax);
        As.add(Matrix.Copy(lol.Transpose()));

        Matrix dAL = Matrix.Zeros(1,1);
        Matrix dZL = Matrix.Minus(As.get(As.size()-1),data.YLabels);
        //dZL.PrintDimensions("dZL");
        //System.out.println("DZL : ");
        //dZL.printMatrix();
        Matrix dWL = Matrix.Multiply(Matrix.Multiply(dZL , As.get(As.size()-2).Transpose()),(1/m));
        //dWL.PrintDimensions("dWL");
        //System.out.println("DWL : ");
        //dWL.printMatrix();
        Matrix dBL = Matrix.Multiply(Matrix.Sum(dZL,1),(1/m));
        //dBL.PrintDimensions("dBL");
        //System.out.println("DBL :");
        //dBL.printMatrix();
        //System.out.println("VdW:");
        //VdW.get(VdW.size()-1).PrintDimensions("VdW[-1]");
        VdW.set(VdW.size()-1 , Matrix.Plus(Matrix.Multiply(VdW.get(VdW.size()-1),Beta) , Matrix.Multiply(dWL , (1-Beta))));
        //VdW.get(VdW.size()-1).printMatrix();
        //System.out.println("VdB:");
        VdB.set(VdB.size()-1,Matrix.Plus(VdB.get(VdB.size()-1) , Matrix.Multiply(dBL , (1-Beta))));
        //VdB.get(VdB.size()-1).PrintDimensions("VdB");
        //VdB.get(VdB.size()-1).printMatrix();
        //System.out.println("WEIGHTS MINUSING");
        //Weights.get(Weights.size()-1).printMatrix();
        //System.out.println("AFTER:");
        Weights.get(Weights.size()-1).Minus(Matrix.Multiply(VdW.get(VdW.size()-1),learningRate));
        //Weights.get(Weights.size()-1).printMatrix();
        //System.out.println("bias change:");
        //Bias.get(Bias.size()-1).printMatrix();
        Bias.get(Bias.size()-1).Minus(Matrix.Multiply(VdB.get(VdB.size()-1),learningRate));
        //System.out.println("after change :");
        //Bias.get(Bias.size()-1).printMatrix();

        for(int i = Layers.size()-3;i>=0 ;i--){
            //System.out.println("dAL:");
            dAL = Matrix.Multiply(Weights.get(i+1).Transpose(),dZL);
            //dAL.PrintDimensions("dAL");
            //dAL.printMatrix();
            dZL = Matrix.Multiply2(dAL , deriv(Layers.get(i).Activation,As.get(i)));
           // dZL.PrintDimensions("dZL");
            //System.out.println("dZL:");
            //dZL.printMatrix();
            if (i == 0){
                //dWL = (1 / m) * np.dot(dZL, X.T) + ((landa / m) * W[i])
                dWL = new Matrix();
                dWL = Matrix.Multiply(Matrix.Multiply(dZL , data.InputData.Transpose()),(1/m));
            }
            else{
                dWL = new Matrix();
                dWL = Matrix.Multiply(Matrix.Multiply(dZL,As.get(i-1).Transpose()),(1/m));
            }
            //System.out.println("dWL:");
            //dWL.PrintDimensions("dWL");
            //dWL.printMatrix();

            //System.out.println("DBL:");
            dBL = Matrix.Multiply(Matrix.Sum(dZL,1),(1/m));
            //dBL.printMatrix();
            //dBL.PrintDimensions("dBL");
            VdW.set(i , Matrix.Plus(Matrix.Multiply(VdW.get(i),Beta) , Matrix.Multiply(dWL , (1-Beta))));
            //VdW.get(i).PrintDimensions("VdW[i]");

            VdB.set(i, Matrix.Plus(Matrix.Multiply(VdB.get(i),Beta) , Matrix.Multiply(dBL,(1-Beta))));
            //VdB.get(i).PrintDimensions("VdB[i]");

            Weights.get(i).Minus(Matrix.Multiply(VdW.get(i),learningRate));
            Bias.get(i).Minus(Matrix.Multiply(VdB.get(i),learningRate));
        }
    }
    void Fit(TrainingData data , int epochs,String fileName) throws Exception {
        Double lastCost = Cost(data);
        for(int i = 0 ; i<epochs;i++){
            backPropogate(data);
            //(feedForward(data).Transpose()).printMatrix();
            if(i%10==0)
                System.out.println("Accuracy : "+100*Accuracy(data)+" and cost : "+Cost(data));
            if(Cost(data)>lastCost) {
                System.out.println("Accuracy : "+100*Accuracy(data)+" and cost : "+Cost(data));
                //(feedForward(data).Transpose()).printMatrix();
                //QsaveModel(fileName);
                //break;
            }
            lastCost = Cost(data);
        }
    }

}
