import java.util.Comparator;

class DistanceComparator implements Comparator<Results> {
	//compares distances to sort them 
		@Override
		public int compare(Results a, Results b) {
			return a.distance < b.distance ? -1 : a.distance == b.distance ? 0 : 1;
		}
	}