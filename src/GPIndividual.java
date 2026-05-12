package src;

public class GPIndividual implements Comparable<GPIndividual> {
    private Node root;
    private double fitness;
    private double[] predictions;
    
    public GPIndividual(Node root) { this.root = root; this.fitness = 0.0; }
    public Node getRoot() { return root; }
    public double getFitness() { return fitness; }
    public void setFitness(double fitness) { this.fitness = fitness; }
    public double[] getPredictions() { return predictions; }
    public void setPredictions(double[] predictions) { this.predictions = predictions; }
    public GPIndividual copy() { return new GPIndividual(root.copy()); }
    
    @Override
    public int compareTo(GPIndividual other) { return Double.compare(other.fitness, this.fitness); }
    
    public static abstract class Node {
        public abstract double evaluate(double[] features);
        public abstract Node copy();
        public abstract String toString();
        public abstract int getDepth();
        public abstract int getNodeCount();
    }
    
    public static class TerminalNode extends Node {
        private boolean isFeature;
        private int featureIndex;
        private double constant;
        
        public TerminalNode(int featureIndex) { this.isFeature = true; this.featureIndex = featureIndex; }
        public TerminalNode(double constant) { this.isFeature = false; this.constant = constant; }
        
        public double evaluate(double[] features) {
            if (isFeature) return features[featureIndex];
            else return constant;
        }
        
        public Node copy() {
            if (isFeature) return new TerminalNode(featureIndex);
            else return new TerminalNode(constant);
        }
        
        public String toString() {
            if (isFeature) return GPParameters.FEATURES[featureIndex];
            else return String.format("%.2f", constant);
        }
        
        public int getDepth() { return 1; }
        public int getNodeCount() { return 1; }
    }
    
    public static class FunctionNode extends Node {
        public String operator;
        public Node left;
        public Node right;
        
        public FunctionNode(String operator, Node left, Node right) {
            this.operator = operator;
            this.left = left;
            this.right = right;
        }
        
        public double evaluate(double[] features) {
            double leftVal = left.evaluate(features);
            if (operator.equals("sin")) return Math.sin(leftVal);
            if (operator.equals("cos")) return Math.cos(leftVal);
            if (operator.equals("exp")) return Math.exp(Math.min(leftVal, 10));
            if (operator.equals("log")) return leftVal > 0 ? Math.log(leftVal) : 0;
            if (operator.equals("sqrt")) return Math.sqrt(Math.abs(leftVal));
            double rightVal = right.evaluate(features);
            if (operator.equals("+")) return leftVal + rightVal;
            if (operator.equals("-")) return leftVal - rightVal;
            if (operator.equals("*")) return leftVal * rightVal;
            if (operator.equals("/")) return rightVal == 0 ? 0 : leftVal / rightVal;
            return leftVal;
        }
        
        public Node copy() { return new FunctionNode(operator, left.copy(), right == null ? null : right.copy()); }
        
        public String toString() {
            if (operator.equals("sin") || operator.equals("cos") || operator.equals("exp") || operator.equals("log") || operator.equals("sqrt"))
                return "(" + operator + " " + left.toString() + ")";
            return "(" + operator + " " + left.toString() + " " + right.toString() + ")";
        }
        
        public int getDepth() { return 1 + Math.max(left.getDepth(), right == null ? 0 : right.getDepth()); }
        public int getNodeCount() { return 1 + left.getNodeCount() + (right == null ? 0 : right.getNodeCount()); }
    }
}
