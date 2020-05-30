/**
 *
 * @author Rémi Wong
 * @version 1.0
 * Main Class for Multi-Layer Perceptron problem for Digit Recognition
 *  
 */

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;


public class OCRemi {

	public static void main(String[] args) {
		
		//set network parameters
		int epoch = 100;

		double trainingFold = 2810;
		double testFold = 2810;	
		
		//initialise matrix arrays for training and test sets
		int[][] trainingSet = new int[2810][65];
		int[][] testSet = new int[2810][65];
		int[] trainingClass = new int[2810];
		int[] testClass = new int[2810];
		System.out.println(
				"[WELCOME TO DIGIT RECOGNITION PROBLEM]\nIn Progres .... ");

		
		String csvFile = "cw2DataSet1.csv";
		String csvFile2 = "cw2DataSet2.csv";
		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";
		
		//reads training set lines and adds to training matrix array
		trainingSet = ReadCSV(csvFile, br, line, cvsSplitBy);

		for (int countClass = 0; countClass < 2810; countClass++) {
			trainingClass[countClass] = trainingSet[countClass][64];

		}

		//reads test set lines and adds to training matrix array
		testSet = ReadCSV(csvFile2, br, line, cvsSplitBy);
		for (int setTClass = 0; setTClass < 2810; setTClass++) {
			
			testClass[setTClass] = testSet[setTClass][64];
		}


		Network net = new Network();
		net.Net();
		// net.printWeights();
		// net.printHiddenWeights();
		
		//Based on defined no of epoch, runs the training of the network through a loop of forward and backward propagation to claibrate weights
		for(int countEpoch=0;countEpoch<epoch; countEpoch++) {
			for (int trainingSetLine = 0; trainingSetLine < trainingFold; trainingSetLine++) {
				int targetClass = trainingClass[trainingSetLine];
	
				net.ForwardPropagation(trainingSet, targetClass, trainingSetLine);
				net.Backpass(trainingSet, trainingSetLine);
				net.backpassHiddenWeight();
				net.BackpassInputWeight(trainingSet, trainingSetLine);
				//System.out.println("target: " + targetClass);
			}
		// net.printHiddenWeights();
		}
		
		//counters to measure expected and classified digits 
		int expected0 = 0, expected1 = 0, expected2 = 0, expected3 = 0, expected4 = 0, expected5 = 0, expected6 = 0, expected7 = 0, expected8 = 0, expected9 = 0;
		int classified0 = 0, classified1 = 0, classified2 = 0, classified3 = 0, classified4 = 0, classified5 = 0, classified6 = 0, classified7 = 0, classified8 = 0, classified9 = 0;
		double totalClassified = 0;
		double totalEx=0;
		
		//conducts the testing of the test dataset based on defined two-fold through forward propagation based on calibrated weights 
		for(int testSetLine = 0; testSetLine< testFold; testSetLine++) {
			totalEx++;
			int testOutput = testClass[testSetLine];
			switch(testOutput) {
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
			net.ForwardPropagation(testSet, testOutput, testSetLine);
			//System.out.println("expected class: " + testOutput);
			int forwardOutput = net.sortingTestSet(testOutput);
			//System.out.println("Test Class" + testSetLine+ "output: " + forwardOutput);
			
			//measure classified digits
			if(testOutput == forwardOutput) {
				switch(testOutput) {
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
				totalClassified++;
			}
		}
		
		
		//display results and accuracy of the algorithm
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
		double accuracy = totalClassified/totalEx*100;
		System.out.println("Total Classified: "+ totalClassified + " Total Expected : "+totalEx+ " Overall Accuracy: " + accuracy + " %");
	}

	
	//method to convert CSV sources values to matrix array elements
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
}
