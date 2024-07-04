import * as tf from '@tensorflow/tfjs-node-gpu'
import * as fs from 'fs'
import { glob } from 'glob'

const encodeDir = (filePath) => {
  if (filePath.includes('donald')) return 0
  if (filePath.includes('mickey')) return 1
  if (filePath.includes('minion')) return 2
  if (filePath.includes('olaf')) return 3
  if (filePath.includes('pooh')) return 4
  if (filePath.includes('pumba')) return 5
};

const shuffleCombo = (array, array2, array3) => {
  let counter = array.length
  console.assert(array.length === array2.length)
  console.assert(array.length === array3.length)
  let temp, temp2, temp3
  let index = 0
  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = (Math.random() * counter) | 0
    // Decrease counter by 1
    counter--
    // And swap the last element with it
    temp = array[counter]
    temp2 = array2[counter]
    temp3 = array3[counter]
    array[counter] = array[index]
    array2[counter] = array2[index]
    array3[counter] = array3[index]
    array[index] = temp
    array2[index] = temp2
    array3[index] = temp3
  }
}

const encodeImages = async (path = 'cartoon/train') => {
  // create stack in JS
  const XS = []
  const YS = []
  const ZS = []

  const files = await glob(`${path}/**/*.*`);

  console.log(`${files.length} Files Found`)
  files.forEach((file) => {
    let imageTensor;
    let answer;
    let imageBase64;
    try {
      const imageData = fs.readFileSync(file);
      imageBase64 = imageData.toString('base64');
      answer = encodeDir(file)
      imageTensor = tf.node.decodeImage(imageData, 3).resizeBilinear([256, 256]);
    } catch (error) {
      console.log('Error reading file', file, error);
    }

    // Store in memory
    if (imageTensor.shape[0] === 256 && imageTensor.shape[1] === 256 && imageTensor.shape[2] === 3) {
      YS.push(answer)
      XS.push(imageTensor)
      ZS.push(imageBase64)
    }

  });

  // Shuffle the data (keep XS[n] === YS[n])
  shuffleCombo(XS, YS, ZS)

  // normalize the images?
  

  // Stack values
  console.log('Stacking')
  console.log('XS', XS.length)
  const X = tf.stack(XS)
  const Y = tf.oneHot(YS, 6)

  console.log('Images all converted to tensors:')
  console.log('X', X.shape)
  console.log('Y', Y.shape)

  // Normalize X to values 0 - 1
  const XNORM = X.div(255);
  // cleanup
  tf.dispose([XS, X]);

  return [XNORM, Y, ZS];
};

export default encodeImages;