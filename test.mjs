import * as tf from '@tensorflow/tfjs-node-gpu';
import * as fs from 'fs';
import encodeImages from './encode.mjs';

const model = await tf.loadLayersModel('file://model_result/model.json');

const [X, Y, Z] = await encodeImages('cartoon/test');

console.log(X.arraySync().length);

const prediction = model.predict(X);
prediction.arraySync().forEach((element, index) => {
  console.log(`Prediction: ${element} Answer: ${Y.arraySync()[index]}, File: ${Z[index]}`);
});

const predictionJsonObject = prediction.arraySync().map((element, index) => {
  return { prediction: element, answer: Y.arraySync()[index], file: Z[index] };
});

fs.writeFileSync('prediction.json', JSON.stringify(predictionJsonObject, null, 2));

tf.dispose([X, Y, model]);
