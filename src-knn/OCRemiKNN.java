/**
 *
 * @author Ah-Kwet Rémi Wong Suk Hee
 * @version 1.0
 * Main Class for Weighted K-NN problem for Digit Recognition
 *  
 */

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

class OCRemiKNN {


	public static void main(String args[]) {
		//preset number of neighbours and number of weights
		int k = 7;
		int numWeights = 3;
		
		//define two-fold test sets, if 2810 for both, takes all data sets		
		double trainingFold = 2810;
		double testFold = 2810  ;
		
		//hold sets from CSV data files
		int[][] trainingSet = new int[2810][65];
		int[][] testSet = new int[2810][65];


		// list to save training data
		List<Train> trainList = new ArrayList<Train>();
		// list to save distance result
		List<Results> distanceList = new ArrayList<Results>();
		// list to save testing data
		List<Test> testList = new ArrayList<Test>();

		

		String csvFile = "cw2DataSet1.csv";
		String csvFile2 = "cw2DataSet2.csv";
		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";

		//Reads training sets
		trainingSet = ReadCSV(csvFile, br, line, cvsSplitBy);

		double[][] injectInputs = new double[2810][64];
		int[] trainClass = new int[2810];
		for (int numTrains = 0; numTrains < 2810; numTrains++) {
			for (int numInputs = 0; numInputs < 64; numInputs++) {
				injectInputs[numTrains][numInputs] = trainingSet[numTrains][numInputs];
				// System.out.println(injectInputs[numInputs]);
			}
			trainClass[numTrains] = trainingSet[numTrains][64];

		}
		for (int allTrainingLines = 0; allTrainingLines < trainingFold; allTrainingLines++) {
			trainList.add(new Train(injectInputs[allTrainingLines], Integer.toString(trainClass[allTrainingLines])));
		}

		//Reads test sets
		testSet = ReadCSV(csvFile2, br, line, cvsSplitBy);
		int[] testClass = new int[2810];
		double[][] testInputs = new double[2810][64];

		for (int numTests = 0; numTests < 2810; numTests++) {
			for (int numInputs = 0; numInputs < 64; numInputs++) {
				testInputs[numTests][numInputs] = testSet[numTests][numInputs];
				// System.out.println(injectInputs[numInputs]);
			}
			testClass[numTests] = testSet[numTests][64];

		}
		
		//add all test sets to the testList object constructors
		for (int allTestLines = 0; allTestLines < testFold; allTestLines++) {
			testList.add(new Test(testInputs[allTestLines], Integer.toString(testClass[allTestLines])));
		}

		// counters for expected and classified digits
		int expected0 = 0, expected1 = 0, expected2 = 0, expected3 = 0, expected4 = 0, expected5 = 0, expected6 = 0, expected7 = 0, expected8 = 0, expected9 = 0;
		int classified0 = 0, classified1 = 0, classified2 = 0, classified3 = 0, classified4 = 0, classified5 = 0, classified6 = 0, classified7 = 0, classified8 = 0, classified9 = 0;
		double testMatches = 0;
	
		
		//starts the classification process for all test sets required 
		for (int classifyTest = 0; classifyTest < testFold; classifyTest++) {
			double[] query = new double[64];
			
			//selects a batch of 64 test set inputs to be used against training points
			for (int populateTestPoints = 0; populateTestPoints < 64; populateTestPoints++) {
				query[populateTestPoints] = testInputs[classifyTest][populateTestPoints];
			}
			
			
			// calculate Euclidean Distance calculation of test point to all training points and add them to the results list
			for (Train train : trainList) {

				double sumDotProducts = 0.0;

				for (int j = 0; j < train.trainAttributes.length; j++) {
					sumDotProducts += Math.pow(train.trainAttributes[j] - query[j], 2);
					// System.out.print(train.trainAttributes[j]+" ");
				}
				double EDistance = Math.sqrt(sumDotProducts);
				distanceList.add(new Results(EDistance, train.nearestName));

			}
			
			//sorting function to sort the distance obtained form euclidean distance
			Collections.sort(distanceList, new DistanceComparator());
			
			//prints out the nearest neighbours based on value of k and adds them to a selection pool
			String[] nearestPool = new String[k];
			for (int x = 0; x < k; x++) {
				System.out.println("Possible Class: " + distanceList.get(x).nearestName + " Distance from test point " + distanceList.get(x).distance);
				// get classes of k nearest instances (train names) from the list into an array
				nearestPool[x] = distanceList.get(x).nearestName;
			}
			
			//the most frequent class output is selected with the function findMostFrequent
			String selectedClass = findMostFrequent(nearestPool);
			
			//prints the classified digit obtained from occurence
			System.out.println("Class selection by frequency: " + selectedClass);
			System.out.println("Expected Class: " + testClass[classifyTest]);
			
			//defines how many digits are expected from classification
			switch(testClass[classifyTest]) {
			case 0: expected0++; break;
			case 1: expected1++; break;
			case 2: expected2++; break;
			case 3: expected3++; break;
			case 4: expected4++; break;
			case 5: expected5++; break;
			case 6: expected6++; break;
			case 7: expected7++; break;
			case 8: expected8++; break;
			case 9: expected9++; break;
			}

			//Selection of the appropriate class by using weightage if the occurence does not match the 1st nearset neighbour obtained previously
			if (!distanceList.get(0).nearestName.equals(selectedClass)) {
				System.out.println("Nearest: " + distanceList.get(0).distance);

				double[] weights = new double[10];
				int maxWeightPosition = 0;
				double maxWeightValue = 0;
				for (int weightage = 0; weightage < numWeights; weightage++) {
					switch (distanceList.get(weightage).nearestName) {
					case "0":
						weights[0] += 1 / distanceList.get(weightage).distance;
						break;
					case "1":
						weights[1] += 1 / distanceList.get(weightage).distance;
						break;
					case "2":
						weights[2] += 1 / distanceList.get(weightage).distance;
						break;
					case "3":
						weights[3] += 1 / distanceList.get(weightage).distance;
						break;
					case "4":
						weights[4] += 1 / distanceList.get(weightage).distance;
						break;
					case "5":
						weights[5] += 1 / distanceList.get(weightage).distance;
						break;
					case "6":
						weights[6] += 1 / distanceList.get(weightage).distance;
						break;
					case "7":
						weights[7] += 1 / distanceList.get(weightage).distance;
						break;
					case "8":
						weights[8] += 1 / distanceList.get(weightage).distance;
						break;
					case "9":
						weights[9] += 1 / distanceList.get(weightage).distance;
						break;
					}
				}
				//compares weights to obtain highest value
				for (int allWeights = 0; allWeights < 10; allWeights++) {
					if (weights[allWeights] > maxWeightValue) {
						maxWeightValue = weights[allWeights];
						maxWeightPosition = allWeights;
					}
				}
				System.out.println("Max weight val: " + maxWeightValue + " max weight position: " + maxWeightPosition);
				selectedClass = String.valueOf(maxWeightPosition);
				
				
			}
			//measures classified class against the expected value of the test dataset
			if (testClass[classifyTest] == Integer.parseInt(selectedClass)) {
				testMatches++;
				switch(Integer.parseInt(selectedClass)) {
				case 0: classified0++; break;
				case 1: classified1++; break;
				case 2: classified2++; break;
				case 3: classified3++; break;
				case 4: classified4++; break;
				case 5: classified5++; break;
				case 6: classified6++; break;
				case 7: classified7++; break;
				case 8: classified8++; break;
				case 9: classified9++; break;
				}
			} else {
				System.out.println("Incorrect");
			}
			//clears the distance list
			distanceList.clear();

		}
		
		//finally outputs the results of the weighted  K-NN classifier problem and accuracy benchmarks
		System.out.println("\n");
		System.out.println("Digit-0 training: " + expected0+ " test: " + classified0);
		System.out.println("Digit-1 training: " + expected1+ " test: " + classified1);
		System.out.println("Digit-2 training: " + expected2+ " test: " + classified2);
		System.out.println("Digit-3 training: " + expected3+ " test: " + classified3);
		System.out.println("Digit-4 training: " + expected4+ " test: " + classified4);
		System.out.println("Digit-5 training: " + expected5+ " test: " + classified5);
		System.out.println("Digit-6 training: " + expected6+ " test: " + classified6);
		System.out.println("Digit-7 training: " + expected7+ " test: " + classified7);
		System.out.println("Digit-8 training: " + expected8+ " test: " + classified8);
		System.out.println("Digit-9 training: " + expected9+ " test: " + classified9);
		System.out.println("\n");
		double accuracy = testMatches / testFold * 100;
		System.out.println("Detected: " + testMatches);
		System.out.println("Accuracy: " + accuracy + " %");
	}// end main


	//CSV Reader method to convert from CSV format to array matrix
	public static int[][] ReadCSV(String file, BufferedReader b, String l, String s) {
		int[][] currentSet = new int[2810][65];
		int numInputs = 0;
		try {
			b = new BufferedReader(new FileReader(file));
			while ((l = b.readLine()) != null) {

				// use comma as separator
				String[] pixel = l.split(s);

				//
				int c = 0;
				for (int a = 0; a < 65; a++) {

					currentSet[numInputs][a] = Integer.parseInt(pixel[c]);
					c++;

				}
				numInputs++;

			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (b != null) {
				try {
					b.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return currentSet;

	}
	
	
	//Finds the most occurrence of a possible digit class - Dr Noureddin Sadawi's occurence finder solution
	private static String findMostFrequent(String[] array) {
		// add the String array to a HashSet to get unique String values
		Set<String> testSets = new HashSet<String>(Arrays.asList(array));
		// convert the HashSet back to array
		String[] uniqueValues = testSets.toArray(new String[0]);
		// counts for unique strings
		int[] counts = new int[uniqueValues.length];
		// loop thru unique strings and count how many times they appear in original
		// array
		for (int i = 0; i < uniqueValues.length; i++) {
			for (int j = 0; j < array.length; j++) {
				if (array[j].equals(uniqueValues[i])) {
					counts[i]++;
				}
			}
		}

		for (int i = 0; i < uniqueValues.length; i++)
			System.out.println("unique values found: " + uniqueValues[i]);
		for (int i = 0; i < counts.length; i++)
			System.out.println("no of unique values: " + counts[i]);

		int max = counts[0];
		for (int counter = 1; counter < counts.length; counter++) {
			if (counts[counter] > max) {
				max = counts[counter];
			}
		}
		System.out.println("max # of occurences: " + max);

		// how many times max appears
		// we know that max will appear at least once in counts
		// so the value of freq will be 1 at minimum after this loop
		int freq = 0;
		for (int counter = 0; counter < counts.length; counter++) {
			if (counts[counter] == max) {
				freq++;
			}
		}

		// index of most freq value if we have only one mode
		int index = -1;
		if (freq == 1) {
			for (int counter = 0; counter < counts.length; counter++) {
				if (counts[counter] == max) {
					index = counter;
					break;
				}
			}
			// System.out.println("one majority class, index is: "+index);
			return uniqueValues[index];
		} else {// we have multiple modes
			int[] ix = new int[freq];// array of indices of modes
			System.out.println("multiple majority classes: " + freq + " classes");
			int ixi = 0;
			for (int counter = 0; counter < counts.length; counter++) {
				if (counts[counter] == max) {
					ix[ixi] = counter;// save index of each max count value
					ixi++; // increase index of ix array
				}
			}

			for (int counter = 0; counter < ix.length; counter++)
				System.out.println("class index: " + ix[counter]);

			// now choose one at random if a specific output cannot be determined
			Random generator = new Random();
			// get random number 0 <= rIndex < size of ix
			int rIndex = generator.nextInt(ix.length);
			System.out.println("random index: " + rIndex);
			int nIndex = ix[rIndex];
			// return unique value at that index
			return uniqueValues[nIndex];
		}

	}
}
