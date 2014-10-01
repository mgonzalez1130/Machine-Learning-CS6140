import java.util.ArrayDeque;
import java.util.ArrayList;
import java.lang.System;

public class housingRegressionTree {

	private static String[] attributes = {"CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
		"DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"};

	public static double[][] trainArray;
	public static double[][] testArray;

	private static Node rootNode;

	private static final int splitAttribute = 0;
	private static final int splitIndex = 1;
	private static final int indexOfLabel = 13;

	public static void main(String[] args) {
		trainArray = DataInputer.convertTo2dArray
				(DataInputer.insertDataIntoArray ("housing_train.txt"));			
		testArray = DataInputer.convertTo2dArray
				(DataInputer.insertDataIntoArray ("housing_test.txt"));

		rootNode = train(trainArray);
		errorRateTest();
		errorRateTrain();

	}


	private static Node train(double[][] trainArray) {

		int maxTreeDepth = 3;
		int minSamplesPerNode = 10;

		ArrayDeque<Node> nodeQ = new ArrayDeque<Node>();
		Node rootNode = Node.makeRootNode(trainArray);
		nodeQ.add(rootNode);

		while (! nodeQ.isEmpty()) {
			Node nextNode = nodeQ.remove();

			if (nextNode.getTreeDepth() <= maxTreeDepth) {
				double[][] currentData = nextNode.getData();
				int[] bestSplit = determineBestSplit (currentData, nextNode);
				nextNode.setSplit(bestSplit);
				nextNode.setAttribute(bestSplit[splitAttribute]);

				int column = bestSplit[splitAttribute];

				java.util.Arrays.sort(housingRegressionTree.trainArray, 
						new java.util.Comparator<double[]>() {
							public int compare(double[] a, double[] b) {
								return Double.compare(a[column], b[column]);
							}
						});

				java.util.Arrays.sort(currentData,
						new java.util.Comparator<double[]>() {
							public int compare(double[] a, double[] b) {
								return Double.compare(a[column], b[column]);
							}
						});

				ArrayList<double[]> leftData = new ArrayList<double[]>();
				ArrayList<double[]> rightData = new ArrayList<double[]>();


				splitData(currentData, bestSplit, leftData, rightData);

				makeAndCheckChild(minSamplesPerNode, nodeQ, nextNode,
						bestSplit, leftData);

				makeAndCheckChild(minSamplesPerNode, nodeQ, nextNode,
						bestSplit, rightData);
				
				System.out.println(nextNode.toString());
			}
		}

		return rootNode;
	}


	private static void makeAndCheckChild(int minSamplesPerNode,
			ArrayDeque<Node> nodeQ, Node nextNode, int[] bestSplit,
			ArrayList<double[]> leftData) {
		Node leftChild = new Node(DataInputer.convertTo2dArray(leftData), 
				nextNode, bestSplit);
		nextNode.addChild(leftChild);

		if (! (leftData.size() < minSamplesPerNode) &&
				! (leftChild.getMSE() < 0)) {
			nodeQ.add(leftChild);
		} else {
			leftChild.setTerminalTrue();
		}
	}


	private static void splitData(double[][] currentData, int[] bestSplit,
			ArrayList<double[]> leftData, ArrayList<double[]> rightData) {
		for (int i = 0; i < currentData.length; i++) {
			if (i < bestSplit[splitIndex]) {
				leftData.add(currentData[i]);
			} else {
				rightData.add(currentData[i]);
			}
		}
	}

	private static int[] determineBestSplit (double[][] currentData, Node currentNode) {

		//Array with two elements, the first is the index corresponding to the
		//attribute that is split on,
		//the second is an integer representing the index where the given data set
		//should be split into two sub data sets.
		int[] bestSplit = new int[2];
		double bestMSE = Double.POSITIVE_INFINITY;

		for (int k = 0; k < attributes.length - 1; k++) {

			//sort the current node's data array based on the current column k
			int column = k;
			java.util.Arrays.sort(currentData, new java.util.Comparator<double[]>() {
				public int compare(double[] a, double[] b) {
					return Double.compare(a[column], b[column]);
				}
			});

			for (int i = 5; i < currentData.length; i+= 5) {
				if (currentData.length - 5 >= i) {
					double[][] leftData = new double[i][];
					double[][] rightData = new double[currentData.length - i][];

					for (int j = 0; j < leftData.length; j++) {
						leftData[j] = currentData[j];
					}

					for (int j = 0; j < rightData.length; j++) {
						rightData[j] = currentData[i + j];
					}

					double leftMSE = Impurity.meanSquaredError(leftData);
					double rightMSE = Impurity.meanSquaredError(rightData);
					double newMSE = leftMSE + rightMSE; 

					if (newMSE < bestMSE) {
						bestMSE = newMSE;
						bestSplit[splitAttribute] = k;
						bestSplit[splitIndex] = i;
					}
				}
			}
		}
		currentNode.setMSE(bestMSE);
		return bestSplit;
	}	


	private static double errorRateTest() {
		double MSE = 0;
		double squareErrorSum = 0;
		for (int i = 0; i < testArray.length; i++) {
			double predictedLabel = predict(i);
			double actualLabel = testArray[i][indexOfLabel]; 
			squareErrorSum += Math.pow((predictedLabel - actualLabel), 2);
		}
		MSE = squareErrorSum / testArray.length;
		System.out.println("The MSE for the regression tree is: " + MSE);
		return MSE;
	}

	private static double errorRateTrain() {
		double MSE = 0;
		double squareErrorSum = 0;
		for (int i = 0; i < trainArray.length; i++) {
			double predictedLabel = predictTrain(i);
			double actualLabel = trainArray[i][indexOfLabel]; 
			squareErrorSum += Math.pow((predictedLabel - actualLabel), 2);
		}
		MSE = squareErrorSum / trainArray.length;
		System.out.println("The MSE for the regression tree is: " + MSE);
		return MSE;
	}


	private static double predict(int featureRow) {
		Node currentNode = rootNode;
		int[] currentSplit = currentNode.getSplit();
		int leftChild = 0;
		int rightChild = 1;

		while(true) {
		    if (currentNode.getChildren().isEmpty()) {
		        currentNode.setTerminalTrue();
		    }
			if (currentNode.getTerminal()) {
				return currentNode.getAverage();
			} else {
				currentSplit = currentNode.getSplit();
				int currentAttribute = currentSplit[splitAttribute]; 
				int currentSplitIndex = currentSplit[splitIndex];
				
				java.util.Arrays.sort(housingRegressionTree.trainArray, 
						new java.util.Comparator<double[]>() {
							public int compare(double[] a, double[] b) {
								return Double.compare(a[currentAttribute], b[currentAttribute]);
							}
						});
				
				double comparedValue = trainArray[currentSplitIndex][currentAttribute];

				if (testArray[featureRow][currentAttribute] < comparedValue) {
					if (currentNode.getChildren().isEmpty()) {
						return currentNode.getAverage();
					} else {
						currentNode = currentNode.getChildren().get(leftChild);
					}
				} else {
					currentNode = currentNode.getChildren().get(rightChild);
				}
			}
		}
	}

	private static double predictTrain(int featureRow) {
		Node currentNode = rootNode;
		int[] currentSplit = currentNode.getSplit();
		int leftChild = 0;
		int rightChild = 1;

		while(true) {
			if (currentNode.getTerminal()) {
				return currentNode.getAverage();
			} else {
				int currentAttribute = currentSplit[splitAttribute]; 
				int currentSplitIndex = currentSplit[splitIndex];
				
				java.util.Arrays.sort(housingRegressionTree.trainArray, 
						new java.util.Comparator<double[]>() {
							public int compare(double[] a, double[] b) {
								return Double.compare(a[currentAttribute], b[currentAttribute]);
							}
						});
				
				double comparedValue = trainArray[currentSplitIndex][currentAttribute];

				if (trainArray[featureRow][currentAttribute] < comparedValue) {
					currentNode = currentNode.getChildren().get(leftChild);
				} else {
					currentNode = currentNode.getChildren().get(rightChild);
				}
			}
		}
	}


	public String[] getAttributes() {
		return attributes;
	}

	public void setAttributes(String[] attributes) {
		housingRegressionTree.attributes = attributes;
	}
}
