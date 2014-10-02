import Jama.Matrix;

/*
 * Notes: 
 * The pseudocode for this algorithm came from: 
 * http://web.engr.oregonstate.edu/~xfern/classes/cs534/notes/perceptron-4-11.pdf
 * 
 * The code for the columnAppend and colsum methods was borrowed from the package
 * found here: 
 * http://www.seas.upenn.edu/~eeaton/software/Utils/javadoc/edu/umbc/cs/maple/utils/JamaUtils.html
 */


public class perceptron {

	static Matrix perceptronMatrix = DataInputer.convertToMatrix(
			DataInputer.convertTo2dArray(
					DataInputer.insertDataIntoArray("perceptronData.txt")));
	
	//initialize parameters, with an extra row for the bias parameter
	static Matrix parameters = new Matrix ((perceptronMatrix.getColumnDimension()), 1, .5);
	
	private static final double LEARNING_RATE = .5; 
	private static final double UPDATE_THRESHOLD = 0;
	
	public static void main(String[] args) {
		
		//add the bias column to the matrix
		Matrix biasColumn = new Matrix ((perceptronMatrix.getRowDimension()), 1, 1);
		perceptronMatrix = columnAppend(biasColumn, perceptronMatrix);
		
		//initialize variables to track iterations and total mistakes
		int iteration = 1;
		int totalMistakes = 0;
		
		do {
			
			//reset total mistakes variable
			totalMistakes = 0;
			
			//create the matrix that will keep track of the update value 
			Matrix update = new Matrix ((perceptronMatrix.getColumnDimension()) - 1, 1, 0);
			
			//loop through the rows and add associated values from rows with incorrect
			//estimated labels to the update matrix
			
			for (int row = 0; row < perceptronMatrix.getRowDimension(); row++) {
				Matrix currentRowNoLabel = perceptronMatrix.getMatrix
						(row, row, 0, perceptronMatrix.getColumnDimension() - 2);
				Matrix currentRowWithLabel = perceptronMatrix.getMatrix
						(row, row, 0, perceptronMatrix.getColumnDimension() - 1);
				
				double estLabel = parameters.transpose().times
						(currentRowNoLabel.transpose()).get(0, 0);
				double actLabel = currentRowWithLabel.get
						(0, currentRowWithLabel.getColumnDimension() - 1);
				
				if (estLabel * actLabel <= 0) {
					update = update.minus(currentRowNoLabel.transpose().times(actLabel));
					totalMistakes ++;
				}
			}
			
			//multiply the learning rate, then subtract the update values from the
			//parameters
			update = update.times(LEARNING_RATE);
			parameters = parameters.minus(update);
			
			System.out.println("Iteration " + iteration + ", total_mistakes " + totalMistakes);
			iteration ++;
			
		} while (totalMistakes != 0);
		
		System.out.println("");
		System.out.println("Classifier weights: ");
		parameters.print(1, 5);
		
		Matrix normalizedParameters = parameters.getMatrix
				(1, parameters.getRowDimension() - 1, 0, 0);
		normalizedParameters = normalizedParameters.times(1 / parameters.get(0,0));
		
		System.out.println("Normalized classifier weights: ");
		normalizedParameters.print(1, 5);
		
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
	
	public static double colsum(Matrix m, int col) {
		// error check the column index
		if (col < 0 || col >= m.getColumnDimension()) {
			throw new IllegalArgumentException("col exceeds the column indices [0,"+(m.getColumnDimension()-1)+"] for m.");
		}
		
		double colsum = 0;
		
		// loop through the rows for this column and compute the sum
		int numRows = m.getRowDimension();
		for (int i=0; i<numRows; i++) {
			colsum += m.get(i,col);
		}
		
		return colsum;
	}

	
}
