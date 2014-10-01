import Jama.Matrix;


public class linearRegression {
	
	public static Matrix trainMatrix;
	public static Matrix testMatrix;
	public static Matrix labels;
	
	public static int rowNumber = 0;
	public static int columnNumber = 0;
	public static int labelIndex = 0;
	
	public static double[] coefficients;
	
	public static double[][] normalizeValues;
	
	public static void main(String[] args) {
		
		double[][] trainArray = normalize(DataInputer.convertTo2dArray
				(DataInputer.insertDataIntoArray ("housing_train.txt")));
		double[][] testArray = normalize(DataInputer.convertTo2dArray
				(DataInputer.insertDataIntoArray ("housing_test.txt")));
		
		trainMatrix = DataInputer.convertToMatrix(trainArray);
		rowNumber = trainMatrix.getRowDimension() - 1;
		labelIndex = trainMatrix.getColumnDimension() - 1;
		columnNumber = trainMatrix.getColumnDimension() - 1;
		coefficients = new double[columnNumber];
		
		testMatrix = DataInputer.convertToMatrix(testArray);
		labels = subMatrix(labelIndex);
		
		for (int i = 0; i < coefficients.length; i++) {
			coefficients[i] = computeCoefficient (subMatrix (i));
			System.out.println(coefficients[i]);
		}
		
		errorRate();
	}
	
	private static Matrix subMatrix (int columnNumber) {
		int[] columnArray = {columnNumber};
		return trainMatrix.getMatrix(0, rowNumber, columnArray);
	}
	
	private static double computeCoefficient (Matrix xMatrix) {
		Matrix xTranspose = xMatrix.transpose();
		Matrix xTXInverse = xTranspose.times(xMatrix).inverse();
		Matrix xTY = xTranspose.times(labels);
		Matrix result = xTXInverse.times(xTY);

		return result.get(0, 0);
	}
	
	private static void errorRate() {
		double squaredErrorSum = 0;
		double MSE = 0;
		
		for (int i = 0; i < testMatrix.getRowDimension() - 1; i++) {
			double predictedValue = normalizeAndPredict(i);
			double actualValue = testMatrix.get(i, labelIndex);
			squaredErrorSum += Math.pow ((predictedValue - actualValue), 2);
		}
		
		MSE = squaredErrorSum / testMatrix.getRowDimension();
		System.out.println ("The test mean squared error is: " + MSE);
		
		squaredErrorSum = 0;
		MSE = 0;
		
		for (int i = 0; i < trainMatrix.getRowDimension() - 1; i++) {
			double predictedValue = predict(i);
			double actualValue = trainMatrix.get(i, labelIndex);
			squaredErrorSum += Math.pow ((predictedValue - actualValue), 2);
		}
		
		MSE = squaredErrorSum / trainMatrix.getRowDimension();
		System.out.println ("The train mean squared error is: " + MSE);
	}
	
	private static double normalizeAndPredict(int rowIndex) {
		double predictedValue = 0;
		for (int i = 0; i < columnNumber; i++) {
			predictedValue += normalizeValue(testMatrix.get(rowIndex, i), i) 
					* coefficients[i];
		}
		return predictedValue;
	}
	
	private static double predict(int rowIndex) {
		double predictedValue = 0;
		for (int i = 0; i < columnNumber; i++) {
			predictedValue += trainMatrix.get(rowIndex, i) * coefficients[i];
		}
		return predictedValue;
	}
	
	private static double normalizeValue(double originalValue, int columnNumber) {
		double normalizedValue = 0;
		normalizedValue = (originalValue - normalizeValues[columnNumber][0])
				/ normalizeValues[columnNumber][1];
		return normalizedValue;
	}

	public static double[][] normalize (double[][] originalArray) {
		int numOfFeatures = 12;
		double[][] normalizingFactors = new double[13][2];
		
		for (int i = 0; i < numOfFeatures; i++) {
			int column = i;
			java.util.Arrays.sort(originalArray,
					new java.util.Comparator<double[]>() {
			    public int compare(double[] a, double[] b) {
			        return Double.compare(a[column], b[column]);
			    }
			});
			
			double min = originalArray[0][i];
			double max = originalArray[originalArray.length - 1][i];
			normalizingFactors[i][0] = min;
			normalizingFactors[i][1] = max - min;
		}
		
		normalizeValues = normalizingFactors;
		
		for (int i = 0; i < numOfFeatures; i++) {
			for (int j = 0; j < originalArray.length; j++) {
				originalArray[j][i] = (originalArray[j][i] - normalizingFactors[i][0])
						/ normalizingFactors[i][1];
			}
		}
		
		return originalArray;
	}
		
}





