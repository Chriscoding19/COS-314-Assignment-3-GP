package src;

import java.util.*;

public class TestArithmeticGP {
    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter seed value: ");
        long seed = scanner.nextLong();
        scanner.close();
        
        System.out.println("\n================================================");
        System.out.println("TESTING ONLY ARITHMETIC GP");
        System.out.println("================================================\n");
        
        long startTime = System.currentTimeMillis();
        
        ArithmeticGP arith = new ArithmeticGP(seed);
        arith.train("data/Breast_train.csv", "data/Breast_test.csv");
        
        long endTime = System.currentTimeMillis();
        
        System.out.println("\n================================================");
        System.out.println("ARITHMETIC GP RESULTS (Seed: " + seed + ")");
        System.out.println("================================================");
        System.out.printf("Training Accuracy: %.4f (%.2f%%)%n", 
            arith.getTrainingAccuracy(), arith.getTrainingAccuracy() * 100);
        System.out.printf("Test Accuracy: %.4f (%.2f%%)%n", 
            arith.getTestAccuracy(), arith.getTestAccuracy() * 100);
        System.out.printf("F-Measure: %.4f%n", arith.getFMeasure());
        System.out.printf("Runtime: %d ms (%.2f seconds)%n", 
            arith.getRuntime(), arith.getRuntime() / 1000.0);
        System.out.printf("Total Time: %d ms (%.2f seconds)%n", 
            (endTime - startTime), (endTime - startTime) / 1000.0);
    }
}
