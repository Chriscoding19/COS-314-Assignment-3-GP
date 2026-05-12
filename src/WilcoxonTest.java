package src;

import java.util.*;

public class WilcoxonTest {
    public static double test(double[] resultsA, double[] resultsB) {
        int n = resultsA.length;
        double[] diffs = new double[n];
        for (int i = 0; i < n; i++) diffs[i] = resultsA[i] - resultsB[i];
        
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(Math.abs(diffs[a]), Math.abs(diffs[b])));
        
        double Wplus = 0, Wminus = 0;
        for (int rank = 0; rank < n; rank++) {
            int idx = indices[rank];
            if (diffs[idx] > 0) Wplus += (rank + 1);
            else if (diffs[idx] < 0) Wminus += (rank + 1);
        }
        
        double W = Math.min(Wplus, Wminus);
        System.out.println("\n================================================");
        System.out.println("WILCOXON TEST RESULT");
        System.out.println("================================================");
        System.out.println("W statistic: " + W);
        double meanA = 0, meanB = 0;
        for (int i = 0; i < n; i++) { meanA += resultsA[i]; meanB += resultsB[i]; }
        meanA /= n; meanB /= n;
        System.out.println("Arithmetic GP Mean: " + meanA);
        System.out.println("Decision Tree GP Mean: " + meanB);
        return W;
    }
}
