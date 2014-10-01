import java.util.ArrayList;
import java.util.Arrays;


public class Node {
	
	//Pointer to the parent node of this node, null if this is the root node
	private Node parent = null;
	
	//Pointers to the two children nodes of this node, empty if this is a 
	//terminal node
	private ArrayList<Node> children = new ArrayList<Node>();
	
	//A list of integers referring to the data points sorted into this node.
	//The integers correspond to each data points row index in the original
	//2d Array.
	private double[][] data;
	
	private double meanSquaredError;
	
	//true if this node is a terminal node
	private boolean terminal = false;
	
	//treeDepth = 0 if root node;
	private int treeDepth;
	
	private int[] split;
	
	private int attribute;
	
	private double average = 0;
	
	private static final int splitAttribute = 0;
	
	//constructor
	
	public Node(double[][] indexArray, Node parent, int[] split) {
		this.data = indexArray;
		this.parent = parent;
		this.split = split;
		this.attribute = split[splitAttribute];
		this.treeDepth = parent.getTreeDepth() + 1;
		this.meanSquaredError = Impurity.meanSquaredError(indexArray);

		if (indexArray.length != 0) {
			int sumOfLabels = 0;
			int indexOfLabel = 13;
			for (int i = 0; i < indexArray.length; i++) {
				sumOfLabels += indexArray[i][indexOfLabel];
			}
			this.setAverage(sumOfLabels / indexArray.length);
		} else {
			this.setAverage (0);
		}
	}
	
	//specifically for making the root node
	public Node (double[][] indexArray, double[][] dataArray) {
		this.data = indexArray;
		this.treeDepth = 0;
		this.meanSquaredError = Double.POSITIVE_INFINITY;
		
		int sumOfLabels = 0;
		int indexOfLabel = 13;
		for (int i = 0; i < indexArray.length; i++) {
			sumOfLabels += indexArray[i][indexOfLabel];
		}
		this.setAverage(sumOfLabels / indexArray.length);
	}
	
	public static Node makeRootNode (double[][] dataArray) {
		double[][] rootData = new double[dataArray.length][];
		for (int i = 0; i < dataArray.length; i++) {
			rootData[i] = (housingRegressionTree.trainArray[i]);
		}
		
		return new Node(rootData, dataArray);
		
	}
	
	public void setParent (Node parent) {
		this.parent = parent;
	}
	
	public void addChild (Node child) {
		this.children.add(child);
	}
	
	public void setData (double[][] data) {
		this.data = data;
	}
	
	public Node getParent () {
		return parent;
	}
	
	public ArrayList<Node> getChildren () {
		return children;
	}
	
	public double[][] getData () {
		return data;
	}
	
	public double getMSE () {
		return meanSquaredError;
	}
	
	public void setMSE(double mse) {
		this.meanSquaredError = mse;
	}
	
	public void setTerminalTrue() {
		terminal = true;
	}
	
	public boolean getTerminal () {
		return terminal;
	}
	
	public int getTreeDepth () {
		return treeDepth;
	}
	
	public void setTreeDepth (int i) {
		this.treeDepth = i;
	}

	public int[] getSplit() {
		return split;
	}

	public void setSplit(int[] split) {
		this.split = split;
	}

	public int getAttribute() {
		return attribute;
	}

	public void setAttribute(int attribute) {
		this.attribute = attribute;
	}

	public double getAverage() {
		return average;
	}

	public void setAverage(double average) {
		this.average = average;
	}
	
	public String toString() {
		String string = "";
		for (int i = 1; i <= treeDepth; i++) {
			string += "|_____";
		}
		return string + Arrays.toString(split) + " " + treeDepth + " " + meanSquaredError; 
	}

}
