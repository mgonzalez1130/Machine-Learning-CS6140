
public class Impurity {
	
	public static double entropy (double[][] data) {
		int spamTotal = 0;
		for (int i = 0; i < data.length; i++) {
			spamTotal += data[i][data[i].length -1];
		}
		double rows = data.length;
		double leftP= spamTotal / rows;
		double leftlogP = Math.log(leftP) / Math.log(2);
		if (leftlogP == Double.NEGATIVE_INFINITY) {
			leftlogP = 0;
		}
		
		double rightP= (data.length - spamTotal) / rows;
		double rightlogP = Math.log(rightP) / Math.log(2);
		if (rightlogP == Double.NEGATIVE_INFINITY) {
			rightlogP = 0;
		}
		
		double negResult = leftP * leftlogP + rightP * rightlogP;
		
		double entropy = 0 - negResult;
		
		return entropy;
		
	}

	public static double meanSquaredError (double[][] data) {
		int indexOfLabel = 13;
		double sumOfLabels = 0;
		double average = 0;
		double squaredErrorSum = 0;
		double meanSquaredError = 0;
		
		//compute the average
		for (int i = 0; i < data.length; i++) {
			sumOfLabels += data[i][indexOfLabel];
		}
		average = sumOfLabels / data.length;
		
		//computer the mean squared error
		for (int i = 0; i< data.length; i++) {
			squaredErrorSum += 
					Math.pow((average - data[i][indexOfLabel]), 2);
		}
		meanSquaredError = squaredErrorSum / data.length;
		
		return meanSquaredError;
	}
	
	public static double meanSquaredError (int[] data, double[][] sortedTrainData) {
		if (data.length != 0) {
			int indexOfLabel = 13;
			double sumOfLabels = 0;
			double average = 0;
			double squaredErrorSum = 0;
			double meanSquaredError = 0;
			
			//compute the average
			for (int i = 0; i < data.length; i++) {
				sumOfLabels += sortedTrainData[data[i]][indexOfLabel];
			}
			average = sumOfLabels / data.length;
			
			//computer the mean squared error
			for (int i = 0; i< data.length; i++) {
				squaredErrorSum += 
						Math.pow((average - 
								sortedTrainData[data[i]][indexOfLabel]), 2);
			}
			meanSquaredError = squaredErrorSum / data.length;
			
			return meanSquaredError;
		} else {
			return Double.POSITIVE_INFINITY;
		}
	}
	
}


//Entropy = -p+ log (p+)  -  p- log p-