package NN;

public class TrainingData {
    Matrix InputData;
    Matrix YLabels;

    public TrainingData(Matrix inputData, Matrix YLabels) {
        InputData = inputData;
        this.YLabels = YLabels;
    }
}
