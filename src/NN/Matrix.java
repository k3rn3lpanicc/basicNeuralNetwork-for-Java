package NN;

import java.lang.reflect.Array;

public class Matrix {
    private Double[][] data;
    int rows;
    int cols;
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
        rows = data.length;
        cols = data[0].length;
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
