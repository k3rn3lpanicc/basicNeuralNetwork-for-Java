package NN;

public class TrainingData {
    Double[][] InputData;
    Double[][] YLabels;

    public TrainingData(Double[][] inputData, Double[][] YLabels) {
        InputData = inputData;
        this.YLabels = YLabels;
    }
}
