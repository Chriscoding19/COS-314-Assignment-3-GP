package src;

public class Metrics {
    public static double accuracy(int[] predictions, int[] actual) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) if (predictions[i] == actual[i]) correct++;
        return (double) correct / predictions.length;
    }
    
    public static double accuracy(double[] predictions, int[] actual) {
        int[] predInt = new int[predictions.length];
        for (int i = 0; i < predictions.length; i++) predInt[i] = predictions[i] > GPParameters.CLASSIFICATION_THRESHOLD ? 1 : 0;
        return accuracy(predInt, actual);
    }
    
    public static double fMeasure(double[] predictions, int[] actual) {
        int[] predInt = new int[predictions.length];
        for (int i = 0; i < predictions.length; i++) predInt[i] = predictions[i] > GPParameters.CLASSIFICATION_THRESHOLD ? 1 : 0;
        int tp = 0, fp = 0, fn = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predInt[i] == 1 && actual[i] == 1) tp++;
            else if (predInt[i] == 1 && actual[i] == 0) fp++;
            else if (predInt[i] == 0 && actual[i] == 1) fn++;
        }
        double p = (tp + fp == 0) ? 0 : (double) tp / (tp + fp);
        double r = (tp + fn == 0) ? 0 : (double) tp / (tp + fn);
        return (p + r == 0) ? 0 : 2 * p * r / (p + r);
    }
}
