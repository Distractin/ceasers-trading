import ai.djl.ndarray.*;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.ArrayDataset;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class DataLoader {

    public static ArrayDataset createDataset(NDManager manager,String filepath) {
        Path folder = Paths.get(filepath);
        List<float[][]> xList = new ArrayList<>();
        List<float[]> yList = new ArrayList<>();
        int windowSize = 100;

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(folder, "*.csv")) {
            for (Path csvFile : stream) {
                List<float[]> rawData = new ArrayList<>();
                double priceSum = 0;
                double priceSqSum = 0;
                int count = 0;

                // --- PASS 1: Calculate Mean and Variance ---
                try (BufferedReader br = Files.newBufferedReader(csvFile)) {
                    String line; br.readLine(); // skip header
                    while ((line = br.readLine()) != null) {
                        String[] f = line.split(",");
                        if (f.length < 5) continue;
                        float p = Float.parseFloat(f[2]);
                        priceSum += p;
                        priceSqSum += (p * p);
                        count++;
                        // Store temporarily to avoid 3rd file read
                        rawData.add(new float[]{p, Float.parseFloat(f[3]), f[4].equalsIgnoreCase("BUY") ? 1f : 0f});
                    }
                }

                if (count == 0) continue;

                float mean = (float) (priceSum / count);
                float variance = (float) ((priceSqSum / count) - (mean * mean));
                float stdDev = (float) Math.sqrt(variance + 1e-6); // 1e-6 prevents division by zero

                // --- PASS 2: Apply Normalization ---
                List<float[]> normalizedRows = new ArrayList<>();
                for (float[] row : rawData) {
                    float[] normalized = new float[3];
                    normalized[0] = (row[0] - mean) / stdDev; // Price Z-Score
                    normalized[1] = (float) Math.log1p(row[1]); // Log scale for Qty (better for outliers)
                    //normalized[2] = row[2]; // Side (0 or 1)
                    normalizedRows.add(normalized);
                }

                // Sliding window
                for (int i = windowSize; i < normalizedRows.size(); i++) {
                    float[][] window = new float[windowSize][];
                    for (int j = 0; j < windowSize; j++) {
                        window[j] = normalizedRows.get(i - windowSize + j);
                    }
                    xList.add(window);
                    yList.add(normalizedRows.get(i));
                }
                System.out.println("Processed " + csvFile.getFileName() + " | Mean: " + mean + " | StdDev: " + stdDev);
            }
        } catch (IOException e) { e.printStackTrace(); }

        return buildDataset(manager, xList, yList, windowSize);
    }

    private static ArrayDataset buildDataset(NDManager manager, List<float[][]> xList, List<float[]> yList, int windowSize) {
        int numSamples = xList.size();
        int numCols = 2;
        float[] flatX = new float[numSamples * windowSize * numCols];
        int offset = 0;
        for (float[][] window : xList) {
            for (float[] row : window) {
                System.arraycopy(row, 0, flatX, offset, numCols);
                offset += numCols;
            }
        }
        float[] flatY = new float[yList.size() * numCols];
        offset = 0;
        for (float[] row : yList) {
            System.arraycopy(row, 0, flatY, offset, numCols);
            offset += numCols;
        }

        return new ArrayDataset.Builder()
                .setData(manager.create(flatX).reshape(new Shape(numSamples, windowSize, numCols)))
                .optLabels(manager.create(flatY).reshape(new Shape(yList.size(), numCols)))
                .setSampling(32, true)
                .build();
    }
}