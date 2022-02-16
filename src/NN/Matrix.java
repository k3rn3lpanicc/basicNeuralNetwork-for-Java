package NN;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
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
    static Matrix Zeros(int row , int column){
        Matrix salam = new Matrix(new Double[row][column]);
        for(int i = 0;i<salam.getFirst();i++)
            for(int j = 0; j<salam.getSecond();j++)
                salam.setCell(i,j,0.0);
        return salam;
    }
    void Multiply(Matrix to) throws Exception {
        if(this.getSecond()!=to.getFirst())
            throw new Exception("Matrices Sizes are not acceptable : ("+this.getFirst()+","+this.getSecond()+") * ("+to.getFirst()+","+to.getSecond()+")");
        this.data = multiplyMatrices(this.data,  to.data);
    }
    private static void transpose(Double A[][], Double B[][])
    {
        int i, j;
        for (i = 0; i < B.length; i++)
            for (j = 0; j < B[0].length; j++)
                B[i][j] = A[j][i];
    }
    Matrix Transpose(){
        Matrix result = new Matrix(new Double[this.getSecond()][this.getFirst()]);
        transpose(this.getData() , result.getData());
        return result;
    }
    static Matrix Sum(Matrix input , int axis){

        if(axis==0){
            Matrix result = new Matrix(new Double[1][input.getSecond()]);
            for(int i = 0; i< input.getSecond() ; i++){
                Double sum = 0.0;
                for(int j = 0;j<input.getFirst();j++){
                    sum+= input.getCell(j,i);
                }
                result.setCell(0,i , sum);
            }
            return result;
        }
        else{
            Matrix result = new Matrix(new Double[input.getFirst()][1]);
            for(int i = 0;i<input.getFirst();i++){
                Double sum = 0.0;
                for(int j = 0; j<input.getSecond();j++){
                    sum+=input.getCell(i,j);
                }
                result.setCell(i,0 , sum);
            }
            return result;
        }
    }
    static Matrix getRandom(int row , int col){
        Matrix result = new Matrix(new Double[row][col]);
        for(int i = 0;i<row ;i++){
            for(int j = 0;j<col;j++){
                result.setCell(i,j,new Random().nextGaussian());
            }
        }
        return result;
    }
    void Multiply(Double value){
        for(int i = 0; i<getFirst();i++)
            for(int j = 0;j<getSecond();j++)
                data[i][j]*=value;
    }
    void Plus(Double value){
        for(int i = 0; i<getFirst();i++)
            for(int j = 0;j<getSecond();j++)
                data[i][j]+=value;
    }
    void Minus(Matrix value) throws Exception {
        if(getFirst()!=value.getFirst() || getSecond()!=value.getSecond())
            throw new Exception("Matrices are not same size! ("+getFirst()+","+getSecond()+") , ("+value.getFirst()+","+value.getSecond()+")");
        for(int i = 0; i<getFirst();i++)
            for(int j = 0;j<getSecond();j++)
                data[i][j]-=value.getCell(i,j);
    }
    void Minus(Double value){
        for(int i = 0; i<getFirst();i++)
            for(int j = 0;j<getSecond();j++)
                data[i][j]-=value;
    }
    void PrintDimensions(String name){
        System.out.println(name + "("+getFirst()+","+getSecond()+")");
    }
    static Matrix Multiply2(Matrix inp1 , Matrix inp2) throws Exception {
        if(inp1.getSecond()!=inp2.getSecond() || inp1.getFirst()!=inp2.getFirst())
            throw new Exception("Matrix Multipication(X) needs Matrices to be same size : (" + inp1.getFirst()+","+inp1.getSecond()+") != ("+inp2.getFirst()+","+inp2.getSecond()+")");
        Matrix result = new Matrix(new Double[inp1.getFirst()][inp1.getSecond()]);
        for(int i = 0;i<inp1.getFirst();i++){
            for(int j = 0; j<inp1.getSecond();j++){
                result.setCell(i,j,inp1.getCell(i,j)*inp2.getCell(i,j));
            }
        }
        return result;
    }
    void Plus(Matrix value) throws Exception {
        if(getFirst()!=value.getFirst() || getSecond()!=value.getSecond())
            throw new Exception("Matrices are not same size! ("+getFirst()+","+getSecond()+") , ("+value.getFirst()+","+value.getSecond()+")");
        for(int i = 0; i<getFirst();i++)
            for(int j = 0;j<getSecond();j++)
                data[i][j]+=value.getCell(i,j);
    }
    static Matrix Plus(Matrix input , Double value){
        Matrix result = Matrix.Copy(input);
        result.Minus(value);
        return result;
    }
    static Matrix Minus(Matrix input , Matrix value) throws Exception {
        Matrix result = Matrix.Copy(input);
        result.Plus(value);
        return result;
    }
    static Matrix Minus(Matrix input , Double value){
        Matrix result = Matrix.Copy(input);
        result.Minus(value);
        return result;
    }
    static Matrix Plus(Matrix input , Matrix value) throws Exception {
        Matrix result = Matrix.Copy(input);
        result.Plus(value);
        return result;
    }
    static Matrix Multiply(Matrix input , Double value){
        Matrix result = Matrix.Copy(input);
        result.Multiply(value);
        return result;
    }
    static Matrix Multiply(Matrix input , Matrix value) throws Exception {
        Matrix result = Matrix.Copy(input);
        result.Multiply(value);
        return result;
    }
    void PlusBias(Matrix bias) throws Exception {
        if(bias.getFirst()!=getFirst())
            throw new Exception("First parts are not equal!("+getFirst()+","+getSecond()+") , ("+bias.getFirst()+","+bias.getSecond()+")");
        if(bias.getSecond()!=1)
            throw new Exception("Second Part is not one!("+bias.getFirst()+","+bias.getSecond()+")");
        for(int i = 0; i<getFirst();i++){
            for(int j = 0;j<getSecond();j++){
                data[i][j]+=bias.data[i][0];
            }
        }


    }

    static void CopyMatrix(Matrix input, Matrix Output){
        Output = new Matrix(new Double[input.getFirst()][input.getSecond()]);
        for(int i =0 ; i<input.getFirst();i++)
            for(int j = 0;j<input.getSecond();j++)
                Output.setCell(i,j,input.getCell(i,j));
    }
    static Matrix Copy(Matrix input){
        Matrix Output;
        Output = new Matrix(new Double[input.getFirst()][input.getSecond()]);
        for(int i =0 ; i<input.getFirst();i++)
            for(int j = 0;j<input.getSecond();j++)
                Output.setCell(i,j,input.getCell(i,j));
            return Output;
    }
    Matrix(Matrix lol){
        for(int i =0 ; i<lol.getFirst();i++)
            for(int j = 0;j<lol.getSecond();j++)
                this.setCell(i,j,lol.getCell(i,j));
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
