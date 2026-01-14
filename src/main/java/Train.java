import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.DefaultTrainingConfig;
import java.nio.file.Paths;

public class Train {

    public static void main(String[] args) throws Exception {
        int epochs = 10;
        Shape inputShape = new Shape(1, 100, 3); 

        try (NDManager manager = NDManager.newBaseManager()) {

            System.out.println("--- Phase 1: Loading Data ---");
            Model model = Model.newInstance("crypto-predictor");
            model.setBlock(MyModel.buildModel());

            ArrayDataset dataset = DataLoader.createDataset(manager);
            System.out.println("Data Loaded Successfully.");

            // 4. Configure Training
            Loss loss = Loss.l1Loss();

            // Slower learning rate to stop the '3212' error
            Optimizer optimizer = Optimizer.adam()
                .optLearningRateTracker(Tracker.fixed(0.00001f)) 
                .build();

            DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optOptimizer(optimizer);
                // We removed the .logging() listener since your SLF4J is missing

            // 5. Training Execution
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(inputShape);

                System.out.println("--- Phase 2: Starting Training Loop ---");

                for (int epoch = 1; epoch <= epochs; epoch++) {
                    float epochLoss = 0;
                    int batches = 0;

                    System.out.print("Epoch " + epoch + " processing: ");

                    for (Batch batch : dataset.getData(manager)) {
                        NDList features = batch.getData();
                        NDList labels = batch.getLabels();

                        try (GradientCollector gc = trainer.newGradientCollector()) {
                            NDList predictions = trainer.forward(features);
                            NDArray lossValue = trainer.getLoss().evaluate(labels, predictions);
                            
                            gc.backward(lossValue);
                            
                            epochLoss += lossValue.getFloat();
                            batches++;
                        }
                        
                        // FIX: Manual Gradient Clipping to prevent the 3212 error
                        // This prevents any weight update from being too aggressive
                        for (Parameter p : model.getBlock().getParameters().values()) {
                            NDArray grad = p.getArray().getGradient();
                            if (grad != null) {
                                grad.clip(1.0f, -1.0f);
                            }
                        }

                        trainer.step();
                        
                        // Visual feedback (one dot every 500 batches)
                        if (batches % 500 == 0) System.out.print(".");
                        
                        batch.close();
                    }

                    float avgLoss = epochLoss / batches;
                    System.out.printf("%n[DONE] Epoch %d | Average Loss: %.6f%n", epoch, avgLoss);
                    
                    if (epoch % 5 == 0) {
                        model.save(Paths.get("model"), "crypto-checkpoint");
                    }
                }

                model.save(Paths.get("model"), "trained-model");
                System.out.println("--- Phase 3: Training Finished ---");
            }
        }
    }
}