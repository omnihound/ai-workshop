// Try to use GPU where possible
// import * as tf from '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs-node-gpu'
import getModel from './model.mjs'
import encodeImages from './encode.mjs'

async function doTraining() {
  // Read images
  const [X, Y, Z] = await encodeImages()

  // Create layers model
  const model = getModel()

  // Train
  await model.fit(X, Y, {
    batchSize: 256,
    validationSplit: 0.1,
    epochs: 20,
    shuffle: true
  })

  model.save('file://model_result/')

  // Cleanup!
  tf.dispose([X, Y, model])
  console.log('Tensors in memory', tf.memory().numTensors)
}

doTraining()
