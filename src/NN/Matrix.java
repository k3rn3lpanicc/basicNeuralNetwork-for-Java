package NN;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;


class RowMultiplyWorker implements Runnable {
    private final Double[][] result;
    private Double[][] matrix1;
    private Double[][] matrix2;
    private final int row;
    public RowMultiplyWorker(Double[][] result, Double[][] matrix1, Double[][] matrix2, int row) {
        this.result = result;
        this.matrix1 = matrix1;
        this.matrix2 = matrix2;
        this.row = row;
    }
    @Override
    public void run() {
        for (int i = 0; i < matrix2[0].length; i++) {
            result[row][i] = 0.0;
            for (int j = 0; j < matrix1[row].length; j++) {
                result[row][i] += matrix1[row][j] * matrix2[j][i];
            }
        }
    }
}
class ParallelThreadsCreator {
    public static boolean multiply(Double[][] matrix1, Double[][] matrix2, Double[][] result) {
        List threads = new ArrayList<>();
        int rows1 = matrix1.length;
        for (int i = 0; i < rows1; i++) {
            RowMultiplyWorker task = new RowMultiplyWorker(result, matrix1, matrix2, i);
            Thread thread = new Thread(task);
            thread.start();
            threads.add(thread);
            if (threads.size() % 10 == 0) {
                waitForThreads(threads);
            }
        }
        return true;
    }
    private static void waitForThreads(List<Thread> threads) {
        for (Thread thread : threads) {
            try {
                thread.join();
                System.out.println("joined");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        threads.clear();
    }
}
public class Matrix {
    private Double[][] data;
    Matrix(){}
    Matrix(Double[][] data){
        setData(data);
    }
    public Double[][] getData() {
        return data;
    }

    public void setData(Double[][] data) {
        this.data = new Double[data.length][data[0].length];
        System.arraycopy(data , 0 , this.data , 0 , data.length);
    }
    Double multiplyMatricesCell(Double[][] firstMatrix, Double[][] secondMatrix, int row, int col) {
        Double cell = 0.0;
        for (int i = 0; i < secondMatrix.length; i++) {
            cell += firstMatrix[row][i] * secondMatrix[i][col];
        }
        return cell;
    }
    Double[][] multiplyMatrices(Double[][] firstMatrix, Double[][] secondMatrix) {
        Double[][] result = new Double[firstMatrix.length][secondMatrix[0].length];
        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[row].length; col++) {
                result[row][col] = multiplyMatricesCell(firstMatrix, secondMatrix, row, col);
            }
        }
        return result;
    }

    void Multiply(Matrix to){
        Matrix lol = new Matrix();
        //lol.data = new Double[this.getFirst()][to.getSecond()];
        //System.out.println("Multiplying : "+this.data.length+"*"+this.data[0].length+" in "+to.data.length+"*"+to.data[0].length);
        //ParallelThreadsCreator.multiply(this.data, to.data, lol.data);
        //this.data = lol.data;
        this.data = multiplyMatrices(this.data,  to.data);


    }
    int getFirst(){
        return data.length;
    }
    int getSecond(){
        return data[0].length;
    }
    void setCell(int first , int second , Double value){
        data[first][second] = value;
    }
    Double getCell(int first , int second){
        return data[first][second];
    }
    void printMatrix(){
        System.out.print("=======Matrix======\n");
        for(int i = 0; i<data.length;i++){
            for(int j = 0;  j < data[i].length;j++){
                System.out.print(data[i][j]+" ");
            }
            System.out.println();
        }
        System.out.println();
    }
}
