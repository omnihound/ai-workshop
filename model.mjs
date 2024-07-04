// import * as tf from '@tensorflow/tfjs-node'
import * as tf from '@tensorflow/tfjs-node-gpu'

const getModel = () => {
  const model = tf.sequential()

  // Conv + Pool combo
  model.add(
    tf.layers.conv2d({
      filters: 16,
      kernelSize: 3,
      strides: 1,
      padding: 'same',
      activation: 'relu',
      kernelInitializer: 'heNormal',
      inputShape: [256, 256, 3],
    })
  )
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    })
  )

  // Conv + Pool combo
  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      strides: 1,
      padding: 'same',
      activation: 'relu',
    })
  )
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    })
  )

  // Conv + Pool combo
  model.add(
    tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      strides: 1,
      padding: 'same',
      activation: 'relu',
    })
  )
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    })
  )

  // Flatten for connecting to deep layers
  model.add(tf.layers.flatten())

  // One hidden deep layer
  model.add(
    tf.layers.dense({
      units: 128,
      activation: 'relu',
    })
  )
  // Output
  model.add(
    tf.layers.dense({
      units: 6,
      activation: 'softmax',
    })
  )

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  })

  model.summary()

  return model
}

export default getModel;