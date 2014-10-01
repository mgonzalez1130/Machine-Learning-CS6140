import Jama.Matrix;


public class spamDecisionTree {
	
	private static final int labelIndex = 57;
	private static final int spam = 1;
	private static final int notSpam = 0;
	private static int numberOfAttributes;
	
	private static spamNode rootNode;
	private static double[][] trainSpamArray;
	private static double[][] testSpamArray;


	private static int[] columnArray = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
		34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 
		51, 52, 53, 54, 55, 56, 57};

	public static void main(String[] args) {
		double[][] spamArray = DataInputer.convertTo2dArray
				(DataInputer.insertDataIntoArray ("spambase.data"));
		
		int rowNumber = spamArray.length - 1;
		int testMaxRow = rowNumber - (rowNumber / 10);
		numberOfAttributes = columnArray.length - 2; 
		
		Matrix spamMatrix = DataInputer.convertToMatrix(spamArray);
		Matrix trainMatrix = spamMatrix.getMatrix(0, testMaxRow, columnArray);
		Matrix testMatrix = spamMatrix.getMatrix(testMaxRow + 1, rowNumber, columnArray);

		

		trainSpamArray = trainMatrix.getArray();
		testSpamArray = testMatrix.getArray();
		
		
		rootNode = new spamNode();
		rootNode.setEntropy(Impurity.entropy(trainSpamArray));
		rootNode.setData (trainSpamArray);
		
		train(rootNode);
		errorRate(testSpamArray);

	}
	
	private static void train(spamNode currentNode) {
		double[][] currentData = currentNode.getData();
		int maxTreeDepth = 10;
		int minSamplesPerNode = 10;
		int[] bestSplit = determineBestSplit (currentData);	
		Matrix currentDataMatrix = DataInputer.convertToMatrix(currentData);
		
		double[][] leftData = currentDataMatrix.getMatrix(0, bestSplit[0], 
				columnArray).getArray();
		double[][] rightData = currentDataMatrix.getMatrix(bestSplit[0] + 1, 
				currentDataMatrix.getRowDimension() - 1, columnArray).getArray();
		
		spamNode leftChild = new spamNode();
		spamNode rightChild = new spamNode();
		
		leftChild.setData(leftData);
		leftChild.setParent(currentNode);
		leftChild.setTreeDepth(currentNode.getTreeDepth() + 1);
		leftChild.setEntropy(Impurity.entropy(leftData));
		leftChild.setEstimatedLabel();
		
		rightChild.setData(rightData);
		rightChild.setParent(currentNode);
		rightChild.setTreeDepth(currentNode.getTreeDepth() + 1);
		rightChild.setEntropy(Impurity.entropy(rightData));
		rightChild.setEstimatedLabel();
		
		currentNode.setSplit(bestSplit);
		currentNode.setChildren(leftChild, rightChild);
		
		if (leftChild.getTreeDepth() < maxTreeDepth &&
				leftChild.getData().length > minSamplesPerNode &&
				leftChild.getEntropy() > 0) {
			train(leftChild);
		} else {
			leftChild.setTerminal(true);
		}
		if (rightChild.getTreeDepth() < maxTreeDepth &&
				rightChild.getData().length > minSamplesPerNode &&
				rightChild.getEntropy() > 0) {
			train(rightChild);	
		} else {
			rightChild.setTerminal(true);
		}
	}
	
	private static int[] determineBestSplit (double[][] currentData) {
		int[] bestSplit = new int[2];
		double bestEntropyDrop = 0;
		double currentEntropy = Impurity.entropy(currentData);
		int rowNumber = currentData.length - 1;
			
		for (int i = 0; i < numberOfAttributes; i++) {
			
			int column = i;
			java.util.Arrays.sort(currentData, new java.util.Comparator<double[]>() {
				public int compare(double[] a, double[] b) {
					return Double.compare(a[column], b[column]);
				}
			});
			
			Matrix currentDataMatrix = DataInputer.convertToMatrix(currentData);
			
			for (int j = 20; j < rowNumber; j += 40) {
				double[][] leftData = (currentDataMatrix.getMatrix(0, j, 
						columnArray )).getArray();
				double[][] rightData = (currentDataMatrix.getMatrix(j+1, rowNumber, 
						columnArray)).getArray();
				
				double leftEntropy = Impurity.entropy(leftData);
				double rightEntropy = Impurity.entropy(rightData);
				double Pl = j / rowNumber;
				
				double entropyDrop = currentEntropy - ((Pl * leftEntropy)) -
						((1 - Pl) * rightEntropy);
				
				if (entropyDrop > bestEntropyDrop) {
					bestEntropyDrop = entropyDrop;
					bestSplit[0] = i;
					bestSplit[1] = j;
				}
			}
		}
		return bestSplit;
	}
	
	public static void errorRate(double[][] testArray) {
		int totalError = 0;
		
		for (int i = 0; i < testArray.length; i++) {
			double predictedValue = predict(testArray[i]);
			double actualValue = testArray[i][labelIndex];
			if (predictedValue != actualValue)
				totalError ++;
		}
		System.out.println("The error rate is: " + totalError);
	}
	
	public static int predict (double[] row) {
		spamNode currentNode = rootNode;
		int[] currentSplit = currentNode.getSplit();
		
		while(true) {
			if (currentNode.isTerminal()) {
				return currentNode.getEstimatedLabel();
			} else {
				int splitIndex = currentSplit[1];
				int splitAttribute = currentSplit[0];
				
				java.util.Arrays.sort(currentNode.getData(), 
						new java.util.Comparator<double[]>() {
							public int compare(double[] a, double[] b) {
								return Double.compare(a[splitAttribute], b[splitAttribute]);
							}
				});
				
				double comparedValue = trainSpamArray[splitIndex][splitAttribute];
				
				if (currentNode.getChildren()[0] == null && 
						currentNode.getChildren()[1] == null) {
					return currentNode.getEstimatedLabel();
				} else if (row[splitAttribute] < comparedValue) {
					currentNode = currentNode.getChildren()[0];
				} else {
					currentNode = currentNode.getChildren()[1];
				}
				
			}
		}
	}
	
}
