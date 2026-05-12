package src;

import java.util.*;

public class ArithmeticGP {
    private Random random;
    private GPIndividual best;
    private Dataset trainData, testData;
    private long runtime;
    private double trainAcc, testAcc, fmeasure;
    
    private static final String[] FUNCTIONS = {"+", "-", "*", "/"};
    
    public ArithmeticGP(long seed) { 
        this.random = new Random(seed);
    }
    
    public void train(String trainFile, String testFile) throws Exception {
        long start = System.currentTimeMillis();
        
        trainData = new Dataset();
        testData = new Dataset();
        trainData.load(trainFile);
        testData.load(testFile);
        
        System.out.println("Training data size: " + trainData.size());
        System.out.println("Test data size: " + testData.size());
        System.out.println();
        
        List<GPIndividual> population = new ArrayList<>();
        for (int i = 0; i < GPParameters.POPULATION_SIZE; i++) {
            GPIndividual ind = new GPIndividual(growTree(GPParameters.INITIAL_TREE_DEPTH));
            population.add(ind);
        }
        
        best = null;
        
        for (int gen = 0; gen < GPParameters.MAX_GENERATIONS; gen++) {
            // Evaluate fitness
            for (GPIndividual ind : population) {
                double[] predictions = evaluate(ind, trainData);
                double acc = Metrics.accuracy(predictions, trainData.getLabelArray());
                double fm = Metrics.fMeasure(predictions, trainData.getLabelArray());
                double fitness = (acc + fm) / 2.0;
                ind.setFitness(fitness);
                ind.setPredictions(predictions);
            }
            
            Collections.sort(population);
            
            GPIndividual currentBest = population.get(0);
            if (best == null || currentBest.getFitness() > best.getFitness()) {
                // Create a deep copy and preserve the fitness value
                best = new GPIndividual(copyTree(currentBest.getRoot()));
                best.setFitness(currentBest.getFitness());
            }
            
            if (gen % 10 == 0) {
                // Force recalc of fitness for display
                double[] bestPreds = evaluate(best, trainData);
                double bestAcc = Metrics.accuracy(bestPreds, trainData.getLabelArray());
                double bestF1 = Metrics.fMeasure(bestPreds, trainData.getLabelArray());
                double bestFit = (bestAcc + bestF1) / 2.0;
                System.out.printf("Gen %d: Best Fitness = %.4f (Train Acc: %.4f, F1: %.4f)%n",
                    gen, bestFit, bestAcc, bestF1);
            }
            
            List<GPIndividual> newPopulation = new ArrayList<>();
            
            int eliteCount = (int)(GPParameters.POPULATION_SIZE * GPParameters.ELITISM_RATE);
            for (int i = 0; i < eliteCount; i++) {
                GPIndividual elite = new GPIndividual(copyTree(population.get(i).getRoot()));
                elite.setFitness(population.get(i).getFitness());
                newPopulation.add(elite);
            }
            
            while (newPopulation.size() < GPParameters.POPULATION_SIZE) {
                GPIndividual parent1 = tournamentSelect(population);
                GPIndividual parent2 = tournamentSelect(population);
                
                GPIndividual offspring;
                if (random.nextDouble() < GPParameters.CROSSOVER_RATE) {
                    offspring = crossover(parent1, parent2);
                } else {
                    offspring = new GPIndividual(copyTree(parent1.getRoot()));
                }
                
                mutate(offspring);
                newPopulation.add(offspring);
            }
            
            population = newPopulation;
        }
        
        double[] finalTrainPred = evaluate(best, trainData);
        double[] finalTestPred = evaluate(best, testData);
        
        trainAcc = Metrics.accuracy(finalTrainPred, trainData.getLabelArray());
        testAcc = Metrics.accuracy(finalTestPred, testData.getLabelArray());
        fmeasure = Metrics.fMeasure(finalTestPred, testData.getLabelArray());
        
        runtime = System.currentTimeMillis() - start;
        
        System.out.println("\n=== BEST EVOLVED EXPRESSION ===");
        System.out.println(best.getRoot().toString());
        System.out.println();
    }
    
    private GPIndividual.Node copyTree(GPIndividual.Node node) {
        if (node == null) return null;
        if (node instanceof GPIndividual.TerminalNode) {
            GPIndividual.TerminalNode tn = (GPIndividual.TerminalNode) node;
            // Need to check if it's feature or constant - use toString to determine
            String str = tn.toString();
            if (str.matches("-?\\d+\\.?\\d*")) {
                return new GPIndividual.TerminalNode(Double.parseDouble(str));
            } else {
                for (int i = 0; i < GPParameters.FEATURES.length; i++) {
                    if (str.equals(GPParameters.FEATURES[i])) {
                        return new GPIndividual.TerminalNode(i);
                    }
                }
                return new GPIndividual.TerminalNode(0);
            }
        } else {
            GPIndividual.FunctionNode fn = (GPIndividual.FunctionNode) node;
            GPIndividual.Node leftCopy = copyTree(fn.left);
            GPIndividual.Node rightCopy = copyTree(fn.right);
            return new GPIndividual.FunctionNode(fn.operator, leftCopy, rightCopy);
        }
    }
    
    private GPIndividual.Node growTree(int maxDepth) {
        if (maxDepth <= 1 || random.nextDouble() < 0.3) {
            if (random.nextDouble() < 0.7) {
                int featureIdx = random.nextInt(GPParameters.NUM_FEATURES);
                return new GPIndividual.TerminalNode(featureIdx);
            } else {
                double constant = GPParameters.MIN_CONSTANT + 
                    random.nextDouble() * (GPParameters.MAX_CONSTANT - GPParameters.MIN_CONSTANT);
                return new GPIndividual.TerminalNode(constant);
            }
        } else {
            String op = FUNCTIONS[random.nextInt(FUNCTIONS.length)];
            GPIndividual.Node left = growTree(maxDepth - 1);
            GPIndividual.Node right = growTree(maxDepth - 1);
            return new GPIndividual.FunctionNode(op, left, right);
        }
    }
    
    private double[] evaluate(GPIndividual ind, Dataset data) {
        double[][] features = data.getFeatureMatrix();
        double[] predictions = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            try {
                predictions[i] = ind.getRoot().evaluate(features[i]);
            } catch (Exception e) {
                predictions[i] = 0;
            }
        }
        return predictions;
    }
    
    private GPIndividual tournamentSelect(List<GPIndividual> population) {
        GPIndividual bestInd = null;
        for (int i = 0; i < GPParameters.TOURNAMENT_SIZE; i++) {
            int idx = random.nextInt(population.size());
            GPIndividual contender = population.get(idx);
            if (bestInd == null || contender.getFitness() > bestInd.getFitness()) {
                bestInd = contender;
            }
        }
        GPIndividual copy = new GPIndividual(copyTree(bestInd.getRoot()));
        copy.setFitness(bestInd.getFitness());
        return copy;
    }
    
    private GPIndividual crossover(GPIndividual parent1, GPIndividual parent2) {
        GPIndividual offspring = new GPIndividual(copyTree(parent1.getRoot()));
        GPIndividual.Node subtree = getRandomSubtree(parent2.getRoot());
        replaceRandomSubtree(offspring.getRoot(), subtree);
        return offspring;
    }
    
    private GPIndividual.Node getRandomSubtree(GPIndividual.Node node) {
        List<GPIndividual.Node> nodes = new ArrayList<>();
        collectNodes(node, nodes);
        int idx = random.nextInt(nodes.size());
        return copyTree(nodes.get(idx));
    }
    
    private void collectNodes(GPIndividual.Node node, List<GPIndividual.Node> nodes) {
        nodes.add(node);
        if (node instanceof GPIndividual.FunctionNode) {
            GPIndividual.FunctionNode fn = (GPIndividual.FunctionNode) node;
            collectNodes(fn.left, nodes);
            if (fn.right != null) {
                collectNodes(fn.right, nodes);
            }
        }
    }
    
    private void replaceRandomSubtree(GPIndividual.Node root, GPIndividual.Node subtree) {
        List<GPIndividual.Node> nodes = new ArrayList<>();
        collectNodes(root, nodes);
        if (nodes.isEmpty()) return;
        
        int idx = random.nextInt(nodes.size());
        GPIndividual.Node target = nodes.get(idx);
        
        if (target instanceof GPIndividual.FunctionNode) {
            GPIndividual.FunctionNode fn = (GPIndividual.FunctionNode) target;
            if (random.nextBoolean()) {
                fn.left = subtree;
            } else if (fn.right != null) {
                fn.right = subtree;
            }
        }
    }
    
    private void mutate(GPIndividual individual) {
        if (random.nextDouble() < GPParameters.MUTATION_RATE) {
            GPIndividual.Node newSubtree = growTree(GPParameters.MAX_TREE_DEPTH / 2);
            replaceRandomSubtree(individual.getRoot(), newSubtree);
        }
    }
    
    public double getTrainingAccuracy() { return trainAcc; }
    public double getTestAccuracy() { return testAcc; }
    public double getFMeasure() { return fmeasure; }
    public long getRuntime() { return runtime; }
}
