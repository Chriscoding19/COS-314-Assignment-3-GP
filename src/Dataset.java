package src;

import java.io.*;
import java.util.*;

public class Dataset {
    private List<double[]> features = new ArrayList<>();
    private List<Integer> labels = new ArrayList<>();
    
    public void load(String filename) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String headerLine = br.readLine();
        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            if (parts.length < 10) continue;
            int label = Integer.parseInt(parts[0].trim());
            labels.add(label);
            double[] featuresArr = new double[GPParameters.NUM_FEATURES];
            for (int i = 0; i < GPParameters.NUM_FEATURES; i++) {
                featuresArr[i] = Double.parseDouble(parts[i + 1].trim());
            }
            features.add(featuresArr);
        }
        br.close();
    }
    
    public int size() { return features.size(); }
    public double[] getFeatures(int index) { return features.get(index); }
    public int getLabel(int index) { return labels.get(index); }
    public List<Integer> getLabels() { return labels; }
    public double[][] getFeatureMatrix() { return features.toArray(new double[0][0]); }
    public int[] getLabelArray() {
        int[] arr = new int[labels.size()];
        for (int i = 0; i < labels.size(); i++) arr[i] = labels.get(i);
        return arr;
    }
}
