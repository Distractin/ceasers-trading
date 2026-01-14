import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;

public class MyModel {

public static Block buildModel() {
    SequentialBlock block = new SequentialBlock();
    
    // Flatten (Batch, 100, 3) -> (Batch, 300)
    block.add(Blocks.batchFlattenBlock()); 

    // Layer 1
    block.add(Linear.builder().setUnits(128).build());
    block.add(Activation.leakyReluBlock(0.2f)); // Add the 0.2f parameter here

    // Layer 2
    block.add(Linear.builder().setUnits(64).build());
    block.add(Activation.leakyReluBlock(0.2f)); // And here

    // Final Output Layer
    block.add(Linear.builder().setUnits(3).build()); 
    
    return block;
}
}
