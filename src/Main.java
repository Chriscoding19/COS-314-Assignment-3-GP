package src;

import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter seed value: ");
        long seed = scanner.nextLong();
        scanner.close();
        
        double[] arithAcc = new double[GPParameters.NUM_RUNS];
        double[] dtAcc = new double[GPParameters.NUM_RUNS];
        
        for (int run = 0; run < GPParameters.NUM_RUNS; run++) {
            long runSeed = seed + run;
            System.out.println("Run " + (run + 1) + " (Seed: " + runSeed + ")");
            
            ArithmeticGP arith = new ArithmeticGP(runSeed);
            arith.train("data/Breast_train.csv", "data/Breast_test.csv");
            arithAcc[run] = arith.getTestAccuracy();
            
            DecisionTreeGP dt = new DecisionTreeGP(runSeed);
            dt.train("data/Breast_train.csv", "data/Breast_test.csv");
            dtAcc[run] = dt.getTestAccuracy();
            
            System.out.printf("  Arithmetic GP Acc: %.4f, Decision Tree Acc: %.4f%n", arithAcc[run], dtAcc[run]);
        }
        
        double meanArith = 0, meanDT = 0;
        for (int i = 0; i < GPParameters.NUM_RUNS; i++) {
            meanArith += arithAcc[i];
            meanDT += dtAcc[i];
        }
        meanArith /= GPParameters.NUM_RUNS;
        meanDT /= GPParameters.NUM_RUNS;
        
        System.out.println("\n================================================");
        System.out.println("FINAL RESULTS (30 runs)");
        System.out.println("================================================");
        System.out.printf("Arithmetic GP Mean Accuracy: %.4f (%.2f%%)%n", meanArith, meanArith * 100);
        System.out.printf("Decision Tree GP Mean Accuracy: %.4f (%.2f%%)%n", meanDT, meanDT * 100);
        
        WilcoxonTest.test(arithAcc, dtAcc);
    }
}
