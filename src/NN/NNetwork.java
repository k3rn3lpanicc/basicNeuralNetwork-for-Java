package NN;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;
import java.nio.charset.StandardCharsets;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
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
    List<Double[][]> Weights = new ArrayList<>();
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
    Double[][] feedForward(TrainingData data) throws Exception {
        Double[][] res = new Double[data.InputData.length][Layers.get(Layers.size() - 1).Neurons.size()];
        int cnt = 0;
        for(int i2 = 0; i2 < data.InputData.length;i2++) {
            Double[] result = new Double[Layers.get(Layers.size() - 1).Neurons.size()];
            result = data.InputData[i2];
            Double[] tempres = new Double[result.length];
            for (int i = 1; i < Layers.size(); i++) {
                if(Layers.get(i).Activation==ActivationTypes.Softmax){
                    tempres = new Double[Layers.get(i).Neurons.size()];
                    tempres = Layers.get(i).calculate(result, Weights.get(i - 1));
                    result = tempres;
                    result = SoftmaxApplied(result);
                }
                else {
                    tempres = new Double[Layers.get(i).Neurons.size()];
                    tempres = Layers.get(i).calculate(result, Weights.get(i - 1));
                    result = tempres;
                }
            }
            res[cnt] = result;
            cnt++;
        }
        return res;
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
        Weights.add(weights);
    }
    public void addLayer(Layer newLayer){
        Layers.add(newLayer);
        if(Layers.size()>=2)
            addWeights(createWeights(Layers.get(Layers.size()-2).Neurons.size(),newLayer.Neurons.size()));
    }
    public void addLayer(int neuronsNo , ActivationTypes Activation){
        Layer newLayer = new Layer(neuronsNo , Activation);
        Layers.add(newLayer);
        if(Layers.size()>=2)
            addWeights(createWeights(Layers.get(Layers.size()-2).Neurons.size(),neuronsNo));
    }
    public void removeLayerAt(int index){
        Layers.remove(index);
        Weights.remove(index);
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
    public List<Double[][]> getWeights() { return Weights; }
    public void setWeights(List<Double[][]> weights) {
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
        for(Double[][] weight : Weights){
            System.out.println("=========Layer "+cntr+"-"+(cntr+1)+" : ");
            cntr++;
            for(int i = 0;i<weight.length;i++) {
                for (int j = 0; j < weight[i].length; j++) {
                    System.out.print(weight[i][j] + " ");
                }
                System.out.println();
            }
            System.out.println();
        }

    }

}
