
public class spamNode {
	
	private spamNode parent;
	private double[][] data;
	private boolean terminal = false;
	private int treeDepth = 0;
	private int[] split;
	private spamNode[] children = new spamNode[2];
	private double entropy;
	private int estimatedLabel;
	
	public spamNode () {
		
	}

	public spamNode getParent() {
		return parent;
	}

	public void setParent(spamNode parent) {
		this.parent = parent;
	}

	public double[][] getData() {
		return data;
	}

	public void setData(double[][] data) {
		this.data = data;
	}

	public boolean isTerminal() {
		return terminal;
	}

	public void setTerminal(boolean terminal) {
		this.terminal = terminal;
	}

	public int getTreeDepth() {
		return treeDepth;
	}

	public void setTreeDepth(int treeDepth) {
		this.treeDepth = treeDepth;
	}

	public int[] getSplit() {
		return split;
	}

	public void setSplit(int[] split) {
		this.split = split;
	}

	public spamNode[] getChildren() {
		return children;
	}

	public void setChildren(spamNode leftChild, spamNode rightChild) {
		this.children[0] = leftChild;
		this.children[1] = rightChild;
	}

	public double getEntropy() {
		return entropy;
	}

	public void setEntropy(double entropy) {
		this.entropy = entropy;
	}

	public void setEstimatedLabel() {
		int spamTotal = 0;
		for (int i = 0; i < data.length; i++) {
			spamTotal += data[i][data[i].length -1];
		}
		
		if ((data.length / 2) < spamTotal) {
			this.estimatedLabel = 0;
		} else {
			this.estimatedLabel = 1;
		}
		
	}
	
	public int getEstimatedLabel() {
		return estimatedLabel;
	}


}
