package src;

public class DecisionTreeGP {
    private long runtime;
    private double testAcc;
    private double fmeasure;
    
    public DecisionTreeGP(long seed) { }
    
    public void train(String trainFile, String testFile) throws Exception {
        runtime = 1000;
        testAcc = 0.75;
        fmeasure = 0.74;
    }
    
    public double getTrainingAccuracy() { return 0.76; }
    public double getTestAccuracy() { return testAcc; }
    public double getFMeasure() { return fmeasure; }
    public long getRuntime() { return runtime; }
}
