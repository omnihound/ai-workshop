import * as React from "react";
import { ReactSketchCanvas } from "react-sketch-canvas";
import * as tf from "@tensorflow/tfjs";

const predictionMatrix = ["donald", "mickey", "minion", "olaf", "pooh", "pumba"];

const base64ToImageData = (base64) => {
  return new Promise((resolve, reject) => {
    // Create a new Image object
    const img = new Image();
    
    // Set the src to the base64 string
    img.src = base64;
    
    // Once the image is loaded, draw it on the canvas
    img.onload = () => {
      // Create a canvas
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      
      // Get the context of the canvas
      const ctx = canvas.getContext('2d');
      
      // Draw the image onto the canvas
      ctx.drawImage(img, 0, 0);
      
      // Get the ImageData from the canvas
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      
      // Resolve the promise with the ImageData
      resolve(imageData);
    };
    
    // Handle errors
    img.onerror = (error) => reject(error);
  });
}

const getKeyOfMaxValue = (arr) => {
  if (arr.length === 0) return -1; // Return -1 if the array is empty

  let maxIndex = 0; // Assume the first element is the max value initially
  arr.forEach((value, index) => {
    console.log("value", value, "index", index, "maxIndex", maxIndex);
    if (value > arr[maxIndex]) {
      maxIndex = index; // Update maxIndex if a larger value is found
    }
  });
  return maxIndex; // Return the index (key) of the maximum value
}

const Canvas = () => {
  const ref = React.useRef(null);
  const [model, setModel] = React.useState(null);
  const [guess, setGuess] = React.useState(null); 

  React.useEffect(() => {
    const loadModel = async () => {
      const model = await tf.loadLayersModel("model.json");
      setModel(model);
    };

    loadModel();  
  }, []);

  const onChange = async () => {
    const paths = await ref.current.exportPaths();
    if (paths.length < 10) return;

    if (model) {
      const image = await ref.current.exportImage("png");
      const imageData = await base64ToImageData(image);
      const tensor = tf.expandDims(tf.browser.fromPixels(imageData), 0);
      const resizedTensor = tf.image.resizeBilinear(tensor, [256, 256]);
  
      const prediction = model.predict(resizedTensor);
      const array = prediction.arraySync()[0];
      setGuess(predictionMatrix[getKeyOfMaxValue(array)]);

      tf.dispose([tensor, resizedTensor, prediction, array]);
    }
  };

  const onClear = () => {
    ref.current.clearCanvas();
    setGuess(null);
  }

  return (
    <div>
      <ReactSketchCanvas
        ref={ref}
        strokeWidth={5}
        strokeColor="black"
        width="512px"
        height="512px"
        onChange={onChange}
      />
      {guess && <span>{guess}</span>}
      <button onClick={onClear}>Clear</button>
    </div>
  );
};

export default Canvas;