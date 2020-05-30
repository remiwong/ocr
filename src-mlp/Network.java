class Network {
	//set number of neurons
	int numHiddenNeurons=32;
	//set learning rate value
	double learningRate=0.1;

	
	////define the links between input, hidden, and output layers in/////
	// 64 for inputs + 1 for bias
	double[][] inputWeights = new double[numHiddenNeurons][65];
	// 10 for hiddenNeuron + 1 for bias
	double[][] hiddenWeights = new double[10][numHiddenNeurons+1];
	int inputBias = 1, hiddenBias = 1;
	
	
	//stores the values of summative weighted perceptron values and activated neuron values
	
	double[] activatedOutputNeurons = new double[10];
	double[] activatedHiddenNeurons = new double[numHiddenNeurons];

	double[] netOutputNeurons = new double[10];
	double[] netHiddenNeurons = new double[numHiddenNeurons];

	// error memory variables
	double[] outputError = new double[10];
	double totalError = 0;

	double targetOut = 0;
	int outClass;

	//Randomises all the weights in the designed network
	public void Net() {

		double rand;
		for (int m = 0; m < numHiddenNeurons; m++) {
			for (int o = 0; o < 65; o++) {
				rand = (double) Math.random();
				inputWeights[m][o] = rand;

			}
		}
		for (int n = 0; n < 10; n++) {

			for (int p = 0; p < numHiddenNeurons+1; p++) {
				rand = (double) Math.random();
				hiddenWeights[n][p] = rand;
			}
		}

	}

	//conducts the calculation of neuron net values and activated neuron values
	public void ForwardPropagation(int[][] dataSet, int targetClass, int dataSetLine) {
		//sets the expected class for future error calculation
		outClass = targetClass;

		// calculate hidden neuron outputs
		for (int a = 0; a < numHiddenNeurons; a++) {
			double netHidden = 0;
			for (int b = 0; b < 64; b++) {
				double inputSum = (dataSet[dataSetLine][b]) / 16 * inputWeights[a][b];
				netHidden += inputSum;
			}
			double biasProduct = inputBias * inputWeights[a][64];

			netHidden += biasProduct;
			//System.out.println("nethiddenValue " + netHidden);

			double outHidden = sigmoid(netHidden);
			activatedHiddenNeurons[a] = outHidden;

		}
		for (int ab = 0; ab < numHiddenNeurons; ab++) {
			//System.out.println("ActivatedOutHiddenValue" + ab + ": " + activatedHiddenNeurons[ab]);
		}

		// calculate MLP outputs
		for (int c = 0; c < 10; c++) {
			double outputValue = 0;
			for (int d = 0; d < numHiddenNeurons; d++) {
				double outProduct = activatedHiddenNeurons[d] * hiddenWeights[c][d];
				outputValue += outProduct;
			}
			double biasProduct = hiddenBias * hiddenWeights[c][numHiddenNeurons];

			outputValue += biasProduct;
			//System.out.println("outputValue " + outputValue);

			double activatedOutputValue = sigmoid(outputValue);
			activatedOutputNeurons[c] = activatedOutputValue;

		}
		for (int cd = 0; cd < 10; cd++) {
			//System.out.println("ActivatedOutputValue" + cd + ": " + activatedOutputNeurons[cd]);
		}
		
		
	}

	//Method using backpass principle to calculate the error between expected output and obtain output from the network activated outputs
	public void Backpass(int[][] trainingSet, int trainingSetLine) {
		// calculate output errors
		for (int outs = 0; outs < 10; outs++) {

			if (outs == outClass) {
				targetOut = 0.99;
			} else {
				targetOut = 0.01;
			}
			//System.out.println("targetOuts" + outs + ": " + targetOut);
			////////// output error based on target
			outputError[outs] = changeInOutputError(targetOut, activatedOutputNeurons[outs]);
			//System.out.println("Error" + outs + ": " + outputError[outs]);



			// printInputWeights();

		}
	}

	//Method calibrates the weights on the input to hidden layers by using the cost functions
	public void BackpassInputWeight(int[][] trainingSet, int trainingSetLine) {
	
		// for all input weights, impact of hidden weight on total output error

		for (int inputsW = 0; inputsW < 65; inputsW++) {

			for (int hW = 0; hW < numHiddenNeurons; hW++) {
				//per input weight
				

				
					double totalErrorContributionofOutputHiddenNeuronhW=0;
					// A dEtotal/doutHn for impact of Hn on all 10 o/p
					
					for (int Eon = 0; Eon < 10; Eon++) {
						double errorContributionofOutputHiddenNeuronhW = outputError[Eon]
								* partialDerivativeOfOutput(activatedOutputNeurons[Eon]) * hiddenWeights[Eon][hW];
						totalErrorContributionofOutputHiddenNeuronhW += errorContributionofOutputHiddenNeuronhW;
					}
				
					// d(Etotal)/d(Weightn) = d(Etotal)/d(outHn) * d(outHn)/d(netH1) * d(netHn)/d(Weightn)
					double hiddenNeuronSelector;
					if (inputsW == 64) {
						hiddenNeuronSelector = 1.0;
					} else {
						hiddenNeuronSelector = activatedHiddenNeurons[hW];
					}
					double changeOnWeightN = totalErrorContributionofOutputHiddenNeuronhW
							* partialDerivativeOfOutput(hiddenNeuronSelector) * trainingSet[trainingSetLine][inputsW];

					//System.out.println("changeOnWeightN: " + changeOnWeightN);
				inputWeights[hW][inputsW] = inputWeights[hW][inputsW] - (learningRate * changeOnWeightN);
			}
			//System.out.println("Reached");
		}
	}

	//Method calibrates the weights on the hidden to output layers by using the cost functions
	public void backpassHiddenWeight() {

		for (int bHW = 0; bHW < 10; bHW++) {
			if (bHW == outClass) {
				targetOut = 0.99;
			} else {
				targetOut = 0.01;
			}
			// backward pass - output layer
			for (int hiddenWeightCount = 0; hiddenWeightCount < numHiddenNeurons; hiddenWeightCount++) {
				double differentialWeightChangeHidden = backPass(targetOut, activatedOutputNeurons[bHW],
						activatedHiddenNeurons[hiddenWeightCount]);
				// --appendedWeights for hidden
				hiddenWeights[bHW][hiddenWeightCount] = hiddenWeights[bHW][hiddenWeightCount] - (learningRate * differentialWeightChangeHidden);
			}
			// bias-hidden
			double differentialWeightChangeHidden = backPass(targetOut, activatedOutputNeurons[bHW], 1);
			hiddenWeights[bHW][numHiddenNeurons] = hiddenWeights[bHW][numHiddenNeurons] - (learningRate * differentialWeightChangeHidden);
		}
	}
	
	//finds the most fired neuron out of the ten outputs to determine class
	public int sortingTestSet(int expectedClass) {
		int selectOut=0;
		double largestOut=0.0;
		for(int checkOutputs=0; checkOutputs<10; checkOutputs++) {

			if(activatedOutputNeurons[checkOutputs] > largestOut) {
				largestOut = activatedOutputNeurons[checkOutputs];
				selectOut = checkOutputs;
			}
		}
		return selectOut;

	}
	
	//prints input weights
	public void printInputWeights() {
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 64; j++) {
				System.out.println("inputweight in" + i + "put" + j + ": " + inputWeights[i][j]);
			}
		}

	}

	//prints hidden neuron weights
	public void printHiddenWeights() {
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 11; j++) {
				System.out.println("hiddeweight out" + i + "hid" + j + ": " + hiddenWeights[i][j]);
			}
		}
	}

	//activates the hidden and output neurons using sigmoid function
	public static double sigmoid(double x) {
		return (1 / (1 + Math.pow(Math.E, (-1 * x))));
	}

	//finds the difference between expected and obtained output for the training part of the back propagation
	public static double changeInOutputError(double targetOut, double activatedOutputNeurons) {
		return activatedOutputNeurons - targetOut;
	}

	//cost function for the hidden to output layer
	public static double backPass(double targetOut, double activatedOutputNeurons, double activatedHiddenNeurons) {
		return -1 * (targetOut - activatedOutputNeurons) * activatedOutputNeurons * (1 - activatedOutputNeurons)
				* activatedHiddenNeurons;
	}

	//calculate partial derivative of output
	public static double partialDerivativeOfOutput(double activatedOutputNeurons) {
		return activatedOutputNeurons * (1 - activatedOutputNeurons);
	}

}