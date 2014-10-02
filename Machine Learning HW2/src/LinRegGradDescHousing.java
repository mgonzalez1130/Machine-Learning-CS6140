import Jama.Matrix;

public class LinRegGradDescHousing {

	//input train and test data into Matrices
	static Matrix trainMatrix = DataInputer.convertToMatrix(
			DataInputer.convertTo2dArray(
					DataInputer.insertDataIntoArray("housing_train.txt")));
	
	static Matrix testMatrix = DataInputer.convertToMatrix(
			DataInputer.convertTo2dArray(
					DataInputer.insertDataIntoArray("housing_test.txt")));
	
	//Matrix with parameter values where each parameter is in its own row
	static Matrix parameters = new Matrix ((trainMatrix.getColumnDimension()) - 1, 1, 0.5);
	
	//independent parameter
	static Matrix indParam = new Matrix (1, 1, 0.5);
	
	//Bias column matrix
	static Matrix biasMatrix = new Matrix ((trainMatrix.getRowDimension()), 1, 1);
	
	//the learning rate
	static final double LEARNING_RATE  = .03;
	
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
		
		for (int i = 0; i < trainMatrix.getRowDimension() - 1; i++) {
			
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
		double squaredErrorSum = 0;
		double MSE = 0;
		
		for (int i = 0; i < testMatrix.getRowDimension() - 1; i++) {
			//extract the current row as its own matrix
			int initColIndex = 0;
			int finColIndex = testMatrix.getColumnDimension() - 1;
			Matrix currentRow = testMatrix.getMatrix(i, i, initColIndex, finColIndex);
					
			//calculate the constant that the current row is going to be multiplied by
			double estLabel = calculateEstimatedLabel(currentRow);
			double actLabel = actualLabels.get(i , 0);
			
			squaredErrorSum += Math.pow ((estLabel - actLabel), 2);
		}
		
		MSE = squaredErrorSum / testMatrix.getRowDimension();
		System.out.println ("The test mean squared error is: " + MSE);
		
		squaredErrorSum = 0;
		MSE = 0;
		
		for (int i = 0; i < trainMatrix.getRowDimension() - 1; i++) {
			//extract the current row as its own matrix
			int initColIndex = 0;
			int finColIndex = trainMatrix.getColumnDimension() - 1;
			Matrix currentRow = trainMatrix.getMatrix(i, i, initColIndex, finColIndex);
					
			//calculate the constant that the current row is going to be multiplied by
			double estLabel = calculateEstimatedLabel(currentRow);
			double actLabel = actualLabels.get(i , 0);
			
			squaredErrorSum += Math.pow ((estLabel - actLabel), 2);
		}
		
		MSE = squaredErrorSum / trainMatrix.getRowDimension();
		System.out.println ("The train mean squared error is: " + MSE);
		
	}

	private static void normalizeData() {
		for (int j = 0; j < trainMatrix.getColumnDimension() - 1; j++) {
			
			//find the minimum and maximum
			double min = Double.POSITIVE_INFINITY;
			double max = Double.NEGATIVE_INFINITY;
			
			for (int i = 0; i < trainMatrix.getRowDimension(); i++) {
				double currentValue = trainMatrix.get(i, j);
				if (currentValue < min) min = currentValue;
				if (currentValue > max) max = currentValue;
			}
			
			//normalize current column in train data
			for (int i = 0; i < trainMatrix.getRowDimension(); i++) {
				double currentValue = trainMatrix.get(i,j);
				double newValue = (currentValue - min) / (max - min); 
				trainMatrix.set(i, j, newValue);
			}
			
			//normalize current column in test data
			for (int i = 0; i < testMatrix.getRowDimension(); i++) {
				double currentValue = testMatrix.get(i,j);
				double newValue = (currentValue - min) / (max - min); 
				testMatrix.set(i, j, newValue);
			}
		}
		
	}

	private static double calculateEstimatedLabel(Matrix currentRow) {
		int lastColumn = currentRow.getColumnDimension() - 2;
		Matrix rowMatrix = currentRow.getMatrix(0, 0, 0, lastColumn); 
		Matrix solMatrix = parameters.transpose().times(rowMatrix.transpose());
		double sol = solMatrix.get(0, 0) + indParam.get(0, 0);
		return sol;
	}
	
	
}
