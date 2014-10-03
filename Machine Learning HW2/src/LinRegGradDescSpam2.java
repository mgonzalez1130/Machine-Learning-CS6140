import Jama.Matrix;

public class LinRegGradDescSpam2 {
    
    static Matrix spamMatrix = DataInputer.convertToMatrix(
            DataInputer.convertTo2dArray(
                    DataInputer.insertDataIntoArray("spambase.data")));
    
    static int lastRowNumber = spamMatrix.getRowDimension() - 1;
    static int testMaxRow = lastRowNumber - (lastRowNumber / 10);
    static int initColIndex = 0;
    static int finColIndex = spamMatrix.getColumnDimension() - 1;
    
    static Matrix trainMatrix = spamMatrix.getMatrix
            (0, testMaxRow, initColIndex, finColIndex);
    static Matrix testMatrix = spamMatrix.getMatrix
            (testMaxRow + 1, lastRowNumber, initColIndex, finColIndex);
    
    static Matrix parameters = new Matrix ((trainMatrix.getColumnDimension()), 1, 0);
   
    static final double LEARNING_RATE  = 0.0000001;
    
    public static void main(String[] args) {
        addBiasColumns();
        normalizeData();
        trainLinearRegression();
        testLinearRegression();
    }
    
    private static void trainLinearRegression() {

        for (int iterations = 0; iterations < 100; iterations++) {
            for (int j = 0; j < parameters.getRowDimension(); j++) {
                int increment = 0;
                
                for (int i = 0; i < trainMatrix.getRowDimension(); i++) {
                    Matrix currentRowNoLabel = trainMatrix.getMatrix
                            (i, i, 0, trainMatrix.getColumnDimension() - 2);
                    double estLabel = parameters.transpose().times
                            (currentRowNoLabel.transpose())
                            .get(0,0);
                    double actLabel = trainMatrix.get
                            (i, trainMatrix.getColumnDimension() - 1);
                    double currentValue = trainMatrix.get(i, j);
                    increment += (actLabel - estLabel) * currentValue;
                }
                
                double currentWeight = parameters.get(j, 0);
                double newWeight = currentWeight 
                        - ((LEARNING_RATE / trainMatrix.getRowDimension())
                        * increment);
                parameters.set(j, 0, newWeight);
            }
        }
        //parameters.print(1, 5);
    }

    private static void testLinearRegression() {
        int errorRate = 0;
        int truePos = 0;
        int trueNeg = 0;
        int falsePos = 0;
        int falseNeg = 0;
        for (int i = 0; i < testMatrix.getRowDimension(); i++) {
            //extract the current row as its own matrix
            int initColIndex = 0;
            int finColIndex = testMatrix.getColumnDimension() - 2;
            Matrix currentRow = testMatrix.getMatrix(i, i, initColIndex, finColIndex);

            //calculate the constant that the current row is going to be multiplied by
            double estLabel = parameters.transpose().times
                    (currentRow.transpose())
                    .get(0,0) * -1;
            estLabel = (estLabel >= 5) ? 1 : 0;
            double actLabel = testMatrix.get
                    (i, testMatrix.getColumnDimension() - 1);
//            System.err.println(estLabel);
//            System.err.println(actLabel);
//            System.err.println("");

            if (estLabel != actLabel) errorRate++;
            
            if (actLabel == 1 && estLabel == 1) {
                truePos++;
            }
            if (actLabel == 0 && estLabel == 0) {
                trueNeg++;
            }
            if (actLabel == 1 && estLabel == 0) {
                falseNeg++;
            }
            if (actLabel == 0 && estLabel == 1) {
                falsePos++;
            }
        }
        
        System.out.println ("The test accuracy is: " + errorRate + " wrong out of " 
                + testMatrix.getRowDimension() + " test rows");
        
        
        System.out.println("True pos: " + truePos);
        System.out.println("True neg: " + trueNeg);
        System.out.println("False pos: " + falsePos);
        System.out.println("False Neg: " + falseNeg);

        errorRate = 0;
        truePos = 0;
        trueNeg = 0;
        falsePos = 0;
        falseNeg = 0;
        for (int i = 0; i < trainMatrix.getRowDimension(); i++) {
            //extract the current row as its own matrix
            int initColIndex = 0;
            int finColIndex = trainMatrix.getColumnDimension() - 2;
            Matrix currentRow = trainMatrix.getMatrix(i, i, initColIndex, finColIndex);

            //calculate the constant that the current row is going to be multiplied by
            double estLabel = parameters.transpose().times
                    (currentRow.transpose())
                    .get(0,0) * -1;
            estLabel = (estLabel >= 7) ? 1 : 0;
            double actLabel = trainMatrix.get
                    (i, trainMatrix.getColumnDimension() - 1);
//            System.err.println(estLabel);
//            System.err.println(actLabel);
//            System.err.println("");

            if (estLabel != actLabel) errorRate++;
            
            if (actLabel == 1 && estLabel == 1) {
                truePos++;
            }
            if (actLabel == 0 && estLabel == 0) {
                trueNeg++;
            }
            if (actLabel == 1 && estLabel == 0) {
                falseNeg++;
            }
            if (actLabel == 0 && estLabel == 1) {
                falsePos++;
            }
        }
        
        System.out.println ("The test accuracy is: " + errorRate + " wrong out of " 
                + trainMatrix.getRowDimension() + " test rows");
        System.out.println("True pos: " + truePos);
        System.out.println("True neg: " + trueNeg);
        System.out.println("False pos: " + falsePos);
        System.out.println("False Neg: " + falseNeg);

    }

    private static void addBiasColumns() {
        Matrix biasColumn = new Matrix (trainMatrix.getRowDimension(), 1, 1);
        trainMatrix = columnAppend (biasColumn, trainMatrix);
        biasColumn = new Matrix (testMatrix.getRowDimension(), 1, 1);
        testMatrix = columnAppend (biasColumn, testMatrix);
    }
    
    public static Matrix columnAppend(Matrix m, Matrix n) {
        int mNumRows = m.getRowDimension();
        int mNumCols = m.getColumnDimension();
        int nNumRows = n.getRowDimension();
        int nNumCols = n.getColumnDimension();
        
        if (mNumRows != nNumRows)
            throw new IllegalArgumentException("Number of rows must be identical to column-append.");
        
        Matrix x = new Matrix(mNumRows,mNumCols+nNumCols);
        x.setMatrix(0,mNumRows-1,0,mNumCols-1,m);
        x.setMatrix(0,mNumRows-1,mNumCols,mNumCols+nNumCols-1,n);
        
        return x;
    }
    
    private static void normalizeData() {
        for (int j = 0; j < spamMatrix.getColumnDimension() - 1; j++) {

            //find the minimum and maximum
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;

            for (int i = 0; i < spamMatrix.getRowDimension(); i++) {
                double currentValue = spamMatrix.get(i, j);
                if (currentValue < min) min = currentValue;
                if (currentValue > max) max = currentValue;
            }

            //normalize current column in spam Matrix
            for (int i = 0; i < spamMatrix.getRowDimension(); i++) {
                double currentValue = spamMatrix.get(i,j);
                double newValue = (currentValue - min) / (max - min); 
                spamMatrix.set(i, j, newValue);
            }
        }
    }
}

