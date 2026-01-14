import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.translate.NoopTranslator;
import java.nio.file.Paths;

public class TestModel {

    public static void main(String[] args) throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            
            Model model = Model.newInstance("crypto-predictor");
            model.setBlock(MyModel.buildModel());
            model.load(Paths.get("model"), "trained-model");

            ArrayDataset testDataset = DataLoader.createDataset(manager);

            try (Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator())) {
                
                double totalAbsoluteError = 0;
                long totalSamples = 0;
                
                System.out.println("Calculating Average Performance...");

                for (Batch batch : testDataset.getData(manager)) {
                    NDList features = batch.getData(); // Shape (32, 100, 3)
                    NDList labels = batch.getLabels();   // Shape (32, 3)

                    // Get prediction for the whole batch
                    NDList prediction = predictor.predict(features);

                    NDArray predArray = prediction.singletonOrThrow();
                    NDArray actualArray = labels.singletonOrThrow();

                    // FIX: We calculate the sum of errors for the entire batch at once
                    // .sub() handles (32, 3) - (32, 3) perfectly.
                    // .abs().sum() squashes all 96 values (32*3) into one total error number.
                    NDArray batchError = actualArray.sub(predArray).abs().sum();
                    
                    totalAbsoluteError += batchError.getFloat();
                    totalSamples += actualArray.getShape().get(0);

                    batch.close();
                }

                // Final Performance Calculation
                // Divided by 3 because each sample has 3 predicted values (Price, Qty, Side)
                double mae = totalAbsoluteError / (totalSamples * 3);
                
                System.out.println("\n------------------------------------");
                System.out.println("Test Samples Processed: " + totalSamples);
                System.out.printf("Average Model Error (MAE): %.6f%n", mae);
                System.out.println("Note: This is in Z-Score units.");
                System.out.println("------------------------------------");
            }
        }
    }
}