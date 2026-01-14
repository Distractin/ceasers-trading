import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.JsonNode;
import com.mashape.unirest.http.Unirest;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.nio.file.Paths;
import java.util.concurrent.*;
import java.util.ArrayList;

// DJL Imports
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.translate.TranslateException;
import ai.djl.translate.NoopTranslator;
import ai.djl.ndarray.NDList;

public class RealWorld {

    private static final String CSV_FILE = "real_world/trading_data.csv";
    private static final String HEADERS = "tickerid,trade_id,trade_time,price,size,side";
    private static Model model;
    private static Predictor<NDList, NDList> predictor;

    public static void main(String[] args) {
        try {
            initCSV();
            loadDJLModel();

            ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
            Unirest.setTimeouts(0, 0);

            System.out.println("Starting 20s test...");

            scheduler.scheduleAtFixedRate(() -> {
                try {
                    // 1. API Call
                    HttpResponse<JsonNode> response = Unirest.get("https://api.kucoin.com/api/v1/market/allTickers").asJson();
                    JSONObject firstTicker = response.getBody().getObject().getJSONObject("data").getJSONArray("ticker").getJSONObject(0);

                    // 2. Format based on your headers
                    String csvLine = String.format("%s,%s",
                        firstTicker.getString("last"),          // price
                        firstTicker.getString("vol")           // size
                    );

                    // 3. Write & Read Back
                    writeToCSV(csvLine);
                    String lastData = readLastTen();

                    // 4. Predict
                    if (lastData != null) {
                        runPrediction(lastData);
                    }

                } catch (Exception e) {
                    System.err.println("Loop Error: " + e.getMessage());
                }
            }, 0, 100, TimeUnit.MILLISECONDS);

            Thread.sleep(20000);
            scheduler.shutdown();
            Unirest.shutdown();
            System.out.println("Test Complete.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void loadDJLModel() throws Exception {
        model = Model.newInstance("price-model");
        // Ensure the "model" folder exists and contains "trained-model.params"
        model.load(Paths.get("model"), "trained-model");
        predictor = model.newPredictor(new NoopTranslator());
    }

    private static void runPrediction(String csvData) {
        try {
            // Logic to convert CSV string to NDList goes here
            // NDList input = ... 
            // predictor.predict(input);
        } catch (TranslateException e) {
            System.err.println("Model failed to translate data: " + e.getMessage());
        }
    }

    private static void initCSV() throws IOException {
        File f = new File(CSV_FILE);
        if (!f.exists()) {
            try (PrintWriter pw = new PrintWriter(new FileWriter(f))) {
                pw.println(HEADERS);
            }
        }
    }

    private static synchronized void writeToCSV(String line) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(CSV_FILE, true))) {
            bw.write(line);
            bw.newLine();
        }
    }

    private static String readLastTen() throws IOException {
        ArrayList<String> returnList;
        String toReturn = "", current;
        try (BufferedReader br = new BufferedReader(new FileReader(CSV_FILE))) {
            while ((current = br.readLine()) != null) {returnList.add(current);}
        }
        return toReturn;
    }
}