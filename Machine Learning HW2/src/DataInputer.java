import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import Jama.Matrix;


public class DataInputer {
	
	public static ArrayList<double[]> insertDataIntoArray (String document) {
		
		ArrayList<double[]> housingData = new ArrayList<double[]>();
		
		//Create Reader for extracting data from file
		BufferedReader dataReader = null;
		try {
			dataReader = new BufferedReader(new FileReader(document));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		try {
			while(true){
				String line = dataReader.readLine();
				if (line == null) break;
				if (line.trim() == "") break;   //ignore lines with no values on them
				String delims = "\\s+|,";
				String[] lineValueStrings = line.trim().split(delims);
				//System.out.println(Arrays.toString(lineValueStrings) 
				//		+ " " + lineValueStrings.length);

				double[] lineValueDoubles = new double[lineValueStrings.length];
				for (int i = 0; i < lineValueStrings.length; i++){
					lineValueDoubles[i] = Double.parseDouble(lineValueStrings[i]);
				}
		
				//System.out.println(Arrays.toString(lineValueDoubles) + ": " + counter);

				housingData.add(lineValueDoubles);
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		try {
			dataReader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}	
		
		return housingData;	
	}
	
	public static double[][] convertTo2dArray (ArrayList<double[]> arrayList) {
		int numOfColumns = arrayList.get(0).length;
		double[][] data2dArray = new double[arrayList.size()][numOfColumns];
		for (int i = 0; i < arrayList.size(); i++) {
			for (int j = 0; j < numOfColumns; j++) {
				data2dArray[i][j] = (arrayList.get(i))[j];
			}
		}
		return data2dArray;
	}
	
	public static Matrix convertToMatrix (double[][] data2dArray) {
		return new Matrix (data2dArray);
		
	}	
}
