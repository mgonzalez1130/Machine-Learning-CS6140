import Jama.Matrix;


public class spamLinearRegression {
	
	public static Matrix spamMatrix;
	public static Matrix testMatrix;
	public static Matrix trainMatrix;
	public static Matrix labels;
	
	public static int rowNumber = 0;
	public static int columnNumber = 0;
	public static int labelIndex = 0;
	
	public static double[] coefficients;
	
	public static double[][] normalizeValues;
	
	public static void main(String[] args) {
	
		double[][] spamArray = normalize(DataInputer.convertTo2dArray
				(DataInputer.insertDataIntoArray("spambase.data")));
		
		spamMatrix = DataInputer.convertToMatrix(spamArray);
		rowNumber = spamMatrix.getRowDimension() - 1;
		labelIndex = spamMatrix.getColumnDimension() - 1;
		columnNumber = spamMatrix.getColumnDimension() - 1;
		coefficients = new double[columnNumber];
		labels = subMatrix(labelIndex);
	
		int counter = 0;
		for (int i = 0; i < coefficients.length; i++) {
			coefficients[i] = computeCoefficient (subMatrix (i));
			System.out.println(coefficients[i] + ": " + counter);
			counter++;
		}
		
		int testMaxRow = rowNumber - (rowNumber / 10);
		int[] columnArray = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
				16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
				34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 
				51, 52, 53, 54, 55, 56, 57};
		
		trainMatrix = spamMatrix.getMatrix(0, testMaxRow, columnArray);
		testMatrix = spamMatrix.getMatrix(testMaxRow + 1, rowNumber, columnArray);
		errorRate();
	}
	
	private static Matrix subMatrix (int columnNumber) {
		int[] columnArray = {columnNumber};
		return spamMatrix.getMatrix(0, rowNumber, columnArray);
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
			double predictedValue = predict(i);
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
	
	private static double predict(int rowIndex) {
		double predictedValue = 0;
		for (int i = 0; i < columnNumber; i++) {
			predictedValue += trainMatrix.get(rowIndex, i) * coefficients[i];
		}
		return predictedValue;
	}
	
	public static double[][] normalize (double[][] originalArray) {
		int numOfFeatures = 13;
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
