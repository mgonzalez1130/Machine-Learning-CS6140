import Jama.Matrix;

public class LinRegGradDescSpam {

	static Matrix spamMatrix = DataInputer.convertToMatrix(
			DataInputer.convertTo2dArray(
					DataInputer.insertDataIntoArray("spambase.data")));

	//input train and test data into Matrices
	static int rowNumber = spamMatrix.getRowDimension() - 1;
	static int testMaxRow = rowNumber - (rowNumber / 10);
	static int initColIndex = 0;
	static int finColIndex = spamMatrix.getColumnDimension() - 1;


	static Matrix trainMatrix = spamMatrix.getMatrix(0, testMaxRow, initColIndex, finColIndex);
	static Matrix testMatrix = spamMatrix.getMatrix(testMaxRow + 1, rowNumber, initColIndex, finColIndex);

	//Matrix with parameter values where each parameter is in its own row
	static Matrix parameters = new Matrix ((trainMatrix.getColumnDimension()) - 1, 1, 0.5);

	//independent parameter
	static Matrix indParam = new Matrix (1, 1, 0.5);

	//Bias column matrix
	static Matrix biasMatrix = new Matrix ((trainMatrix.getRowDimension()), 1, 1);

	//the learning rate
	static final double LEARNING_RATE  = 0.25;

	//Matrix with the actual labels from the trainMatrix in a single column
	static int initialRowIndex = 0;
	static int finalRowIndex = trainMatrix.getRowDimension() - 1;
	static int initialColIndex = trainMatrix.getColumnDimension() - 1;
	static int finalColIndex = initialColIndex;

	static Matrix actualLabels = trainMatrix.getMatrix(initialRowIndex, finalRowIndex,
			initialColIndex, finalColIndex);

	public static void main(String[] args) {
		normalizeData();
		trainLinearRegression();
		testLinearRegression();
	}
	
	private static void trainLinearRegression() {

		for (int i = 0; i < 10; i++) {

			//extract the current row as its own matrix
			int initColIndex = 0;
			int finColIndex = trainMatrix.getColumnDimension() - 1;
			Matrix currentRow = trainMatrix.getMatrix(i, i, initColIndex, finColIndex);

			//calculate the constant that the current row is going to be multiplied by
			double estLabel = calculateEstimatedLabel(currentRow);
			double actLabel = actualLabels.get(i , 0);
			double constant = LEARNING_RATE * (estLabel - actLabel);

			//calculate the new parameters according to the equation
			//theta := theta - learning_rate * (estimated label - actual label) * xji
			int lastColumn = currentRow.getColumnDimension() - 2;
			Matrix rowMatrix = currentRow.getMatrix(0, 0, 0, lastColumn); 
			parameters = parameters.minus(rowMatrix.times(constant).transpose());
			double newIndParam = indParam.get(0,0) - constant;
			indParam.set(0, 0, newIndParam);
		}
	}

	private static void testLinearRegression() {
		int errorRate = 0;
		for (int i = 0; i < testMatrix.getRowDimension(); i++) {
			//extract the current row as its own matrix
			int initColIndex = 0;
			int finColIndex = testMatrix.getColumnDimension() - 1;
			Matrix currentRow = testMatrix.getMatrix(i, i, initColIndex, finColIndex);

			//calculate the constant that the current row is going to be multiplied by
			double estLabel = calculateEstimatedLabel(currentRow);
//			if (estLabel >= 1) estLabel = 1;
//			if (estLabel < 1) estLabel = 0;
			double actLabel = actualLabels.get(i , 0);
//			System.err.println(estLabel);
//			System.err.println(actLabel);
//			System.err.println("");
			
			if (estLabel != actLabel) errorRate++;
	

		}
		System.out.println ("The test accuracy is: " + errorRate + " wrong out of " 
				+ testMatrix.getRowDimension() + " test rows");

		errorRate = 0;
		for (int i = 0; i < trainMatrix.getRowDimension(); i++) {
			//extract the current row as its own matrix
			int initColIndex = 0;
			int finColIndex = trainMatrix.getColumnDimension() - 1;
			Matrix currentRow = trainMatrix.getMatrix(i, i, initColIndex, finColIndex);

			//calculate the constant that the current row is going to be multiplied by
			double estLabel = calculateEstimatedLabel(currentRow);
//			if (estLabel >= 1) estLabel = 1;
//			if (estLabel < 1) estLabel = 0;
			double actLabel = actualLabels.get(i , 0);
//			System.err.println(estLabel);
//			System.err.println(actLabel);
//			System.err.println("");
			
			if (estLabel != actLabel) errorRate++;

		}
		System.out.println ("The train accuracy is: " + errorRate + " wrong out of " 
				+ trainMatrix.getRowDimension() + " train rows");

	}

	private static void normalizeData() {
		for (int j = 0; j < spamMatrix.getColumnDimension() - 1; j++) {

			//find the minimum and maximum
			double min = Double.POSITIVE_INFINITY;
			double max = Double.NEGATIVE_INFINITY;

			for (int i = 0; i < spamMatrix.getRowDimension() - 1; i++) {
				double currentValue = spamMatrix.get(i, j);
				if (currentValue < min) min = currentValue;
				if (currentValue > max) max = currentValue;
			}

			//normalize current column in spam Matrix
			for (int i = 0; i < spamMatrix.getRowDimension() - 1; i++) {
				double currentValue = spamMatrix.get(i,j);
				double newValue = (currentValue - min) / (max - min); 
				spamMatrix.set(i, j, newValue);
			}

		}

	}

	private static double calculateEstimatedLabel(Matrix currentRow) {
		int lastColumn = currentRow.getColumnDimension() - 2;
		Matrix rowMatrix = currentRow.getMatrix(0, 0, 0, lastColumn); 
		Matrix solMatrix = parameters.transpose().times(rowMatrix.transpose());
		double sol = solMatrix.get(0, 0) + indParam.get(0, 0);
		System.out.println(solMatrix.get(0,0));
		System.out.println(indParam.get(0,0));
		System.out.println("");
		return sol;
	}

}
