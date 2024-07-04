import * as tf from '@tensorflow/tfjs-node-gpu';
import encodeImages from './encode.mjs';
import { writeFileSync } from 'fs';
import { createCanvas, ImageData } from 'canvas';

// Function to normalize tensor values to 0-255
function normalizeTensor(tensor) {
  const min = tensor.min();
  const max = tensor.max();
  return tensor.sub(min).div(max.sub(min)).mul(255).toInt();
}

// Function to convert a tensor to an image and save it
async function tensorToImage(tensor, imageName) {
  // Normalize tensor
  const normalizedTensor = normalizeTensor(tensor);
  const [height, width, channels] = normalizedTensor.shape;
  
  // Create a canvas element
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  
  // Convert tensor to Uint8ClampedArray
  const imageData = new ImageData(new Uint8ClampedArray(await normalizedTensor.data()), width, height);
  
  // Put image data on canvas and save as PNG
  ctx.putImageData(imageData, 0, 0);
  writeFileSync(imageName, canvas.toBuffer('image/png'));
}

// Step 1: Inspect your model to find the names of the convolutional layers
// This step is manual and depends on your specific model architecture

const model = await tf.loadLayersModel('file://model_result/model.json');

const [X, Y, Z] = await encodeImages('cartoon/convolutional');

// Step 2: Create a model for each convolutional layer to observe its output
const convLayerNames = ['conv2d_Conv2D1', 'conv2d_Conv2D2', 'conv2d_Conv2D3']; // Example layer names, replace with actual names from your model
const convModels = convLayerNames.map(layerName => {
  return tf.model({inputs: model.inputs, outputs: model.getLayer(layerName).output});
});

// Step 3: Use the new models to predict and observe the convolutional layers' outputs
convModels.forEach(async (convModel, index) => {
  const output = convModel.predict(X); // Assuming X is your input data
  const layerName = convLayerNames[index];

  // console.log(output.shape.length);

  // Assuming the output is a 4D tensor [batch, height, width, channels]
  // Convert each image in the batch
  for (let i = 0; i < output.shape[0]; i++) {
    const singleOutput = output.slice([i, 0, 0, 0], [1, output.shape[1], output.shape[2], output.shape[3]]).squeeze();
    await tensorToImage(singleOutput, `convolutional/output_${layerName}_${i}.png`);
  }
});



tf.dispose([X, Y, model]);
