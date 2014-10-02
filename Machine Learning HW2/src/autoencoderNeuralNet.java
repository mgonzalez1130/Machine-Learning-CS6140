import Jama.Matrix;

public class autoencoderNeuralNet {

	private static final double ONE_THRESHOLD = 0.8;
	private static final double ZERO_THRESHOLD = 0.2;
	private static final double LEARNING_RATE = 0.2;
	
	static Matrix neuralNetworkValues = DataInputer.convertToMatrix(
			DataInputer.convertTo2dArray(
					DataInputer.insertDataIntoArray("neural_train.txt")));
	
	static Matrix inputToHiddenWeights = new Matrix (8, 3);
	static Matrix hiddenToOutputWeights = new Matrix (3, 8);
	
	static Matrix inputToHiddenBiases = new Matrix (3, 1);
	static Matrix hiddenToOutputBiases = new Matrix (8, 1);
	
	static Matrix hiddenErrors = new Matrix (3, 1);
	static Matrix outputErrors = new Matrix (8, 1);
	
	static Matrix inputValues = new Matrix (8, 1);
	static Matrix hiddenValues = new Matrix (3, 1);
	static Matrix hiddenNetInputs = new Matrix (3, 1);
	static Matrix outputValues = new Matrix (8, 1);
	static Matrix outputNetInputs = new Matrix (8, 1);
	
	static int currentRowIndex = -1;
	
	
	public static void main(String[] args) {
		
		//need to do this for each row in the neuralNetworkValues matrix
		for (int row = 0; row < neuralNetworkValues.getRowDimension(); row++) {
			initalizeWeights();
			currentRowIndex++;
			Matrix currentRow = neuralNetworkValues.getMatrix
					(row, row, 0, neuralNetworkValues.getColumnDimension() - 1);
			
			while(conditionNotMet()) {
				forwardPrapogate(currentRow);
				backPropagate();	
			}
			
			//print out the hiddenValues matrix as a row
			hiddenValues.transpose().print(1, 5);
		}
		
	}


	private static boolean conditionNotMet() {
		for (int row = 0; row < outputValues.getRowDimension(); row++) {
			if (outputValues.get(row, 0) <= ZERO_THRESHOLD) {
				outputValues.set(row, 0, 0);
			} else if (outputValues.get(row, 0) >= ONE_THRESHOLD) {
				outputValues.set(row, 0, 1);
			}
		}
		
		Matrix currentRow = neuralNetworkValues.getMatrix
				(currentRowIndex, currentRowIndex, 0, neuralNetworkValues.getColumnDimension() - 1);
		Matrix outputRow = outputValues.transpose();
		
		for (int column = 0; column < currentRow.getColumnDimension(); column++) {
			if (currentRow.get(0, column) != outputRow.get(0, column)) return true;
		}
		
		return false;
	}


	private static void initalizeWeights() {
		//These 2 matrices need to be initialized:
		//inputToHiddenWeights
		//hiddenToOutputWeights
		//biases don't need to be initialized to values other than 0 
		
		for (int row = 0; row < inputToHiddenWeights.getRowDimension(); row++) {
			for (int column = 0; column < inputToHiddenWeights.getColumnDimension(); column++) {
				inputToHiddenWeights.set(row, column, Math.random());
			}
		}
		
		for (int row = 0; row < hiddenToOutputWeights.getRowDimension(); row++) {
			for (int column = 0; column < hiddenToOutputWeights.getColumnDimension(); column++) {
				hiddenToOutputWeights.set(row, column, Math.random());
			}
		}
		
	}


	private static void forwardPrapogate(Matrix currentRow) {
		propagateToHiddenLayer(currentRow);
		propagateToOutputLayer();
		
	}

	private static void propagateToHiddenLayer(Matrix currentRow) {
		
		for (int row = 0; row < hiddenValues.getRowDimension(); row++) {
			Matrix currentWeights = inputToHiddenWeights.getMatrix 
					(0, inputToHiddenWeights.getRowDimension() - 1, row, row);
			double netInput = currentWeights.transpose().times(currentRow.transpose()).get(0, 0)
					+ inputToHiddenBiases.get(row, 0);
			double result = 1 / (1 + Math.pow(Math.E, (netInput * -1)));
			
			hiddenValues.set(row, 0, result);
			hiddenNetInputs.set(row, 0, netInput);
		}
		
	}

	
	private static void propagateToOutputLayer() {
		
		for (int row = 0; row < outputValues.getRowDimension(); row++) {
			Matrix currentWeights = hiddenToOutputWeights.getMatrix
					(0, hiddenToOutputWeights.getRowDimension() - 1, row, row);
			Matrix currentRow = hiddenValues;
			double netInput = currentWeights.transpose().times(currentRow).get(0, 0)
					+ hiddenToOutputBiases.get(row, 0);
			double result = 1 / (1 + Math.pow(Math.E, (netInput * -1)));
			
			outputValues.set(row, 0, result);
			outputNetInputs.set(row, 0, netInput);
		}
		
	}


	private static void backPropagate() {
		computeOutputErrors();
		computeHiddenErrors();
		updateParameters();
		
	}


	private static void computeOutputErrors() {
		// error = (actual value - estimated value) * estimated value * (1 - estimated value)
		
		for (int row = 0; row < outputErrors.getRowDimension(); row++) {
			double actualValue = neuralNetworkValues.get(currentRowIndex, row);
			double estimatedValue = outputValues.get(row, 0);
			double error = (actualValue - estimatedValue) * estimatedValue * (1 - estimatedValue);
			
			outputErrors.set(row, 0, error);
		}
		
	}


	private static void computeHiddenErrors() {
		for (int row = 0; row < hiddenErrors.getRowDimension(); row++) {
			double weight = hiddenToOutputWeights.get(row, 0);
			double estimatedValue = hiddenValues.get(row, 0);
			Matrix errorMatrix = outputErrors.times(weight);
			double error = colsum(errorMatrix, 0) * estimatedValue * (1 - estimatedValue);
			
			hiddenErrors.set(row, 0, error);
		}
		
	}


	private static void updateParameters() {
		updateInputToHiddenWeights();
		updateHiddenToOutputWeights();
		
	}
	
	private static void updateInputToHiddenWeights() {
		for (int column = 0; column < inputToHiddenWeights.getColumnDimension(); column++) {
			for (int row = 0; row < inputToHiddenWeights.getRowDimension(); row++) {
				double increment = LEARNING_RATE * hiddenErrors.get(column, 0) 
						* hiddenValues.get(column, 0);
				double originalWeight = inputToHiddenWeights.get(row, column);
				double newWeight = originalWeight + increment;
				inputToHiddenWeights.set(row, column, newWeight);
			}
		}
		
		for (int row = 0; row < inputToHiddenBiases.getRowDimension(); row++) {
			double increment = LEARNING_RATE * hiddenErrors.get(row, 0);
			double originalBias = inputToHiddenBiases.get(row, 0);
			double newBias = originalBias + increment;
			inputToHiddenBiases.set(row, 0, newBias);
		}
		
	}


	private static void updateHiddenToOutputWeights() {
		for (int column = 0; column < hiddenToOutputWeights.getColumnDimension(); column++) {
			for (int row = 0; row < hiddenToOutputWeights.getRowDimension(); row++) {
				double increment = LEARNING_RATE * outputErrors.get(column, 0)
						* outputValues.get(column, 0);
				double originalWeight = hiddenToOutputWeights.get(row, column);
				double newWeight = originalWeight + increment;
				hiddenToOutputWeights.set(row, column, newWeight);
			}
		}
		
		for (int row = 0; row < hiddenToOutputBiases.getRowDimension(); row++) {
			double increment = LEARNING_RATE * outputErrors.get(row, 0);
			double originalBias = hiddenToOutputBiases.get(row, 0);
			double newBias = originalBias + increment;
			hiddenToOutputBiases.set(row, 0, newBias);
		}
		
	}


	private static double colsum(Matrix m, int col) {
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
