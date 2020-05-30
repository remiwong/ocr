
class Results {
	//euclidean distance between current test inputs and training sets  constructor class
	double distance;
	String nearestName;

	public Results(double distance, String nearestName) {
		this.nearestName = nearestName;
		this.distance = distance;
	}
}