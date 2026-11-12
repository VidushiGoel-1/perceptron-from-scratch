# Perceptron from Scratch (NumPy)

A single-neuron classifier (the perceptron) built from scratch with NumPy, without using a machine learning library for the model itself.

I built this while learning deep learning, following CampusX's perceptron trick video as a guide.

## What this project does
- Creates a fake 2D dataset with two classes using scikit-learn's `make_classification`
- Trains a perceptron using the **perceptron trick**
- Converts the learned weights into a line and plots it over the data

## How the perceptron trick works
1. Add a column of 1s to the data so the bias is treated as just another weight.
2. Start with all weights set to 1.
3. Repeat 1000 times:
   - Pick a random data point.
   - Predict its class: 1 if the weighted sum `w0 + w1·x1 + w2·x2` is above 0, else 0.
   - Update the weights:

   `weights = weights + learning_rate × (y − y_hat) × x`

   - If the prediction is right, `(y − y_hat)` is 0 and nothing changes.
   - If a positive point is predicted negative, the point is added to the weights.
   - If a negative point is predicted positive, the point is subtracted.
4. The final weights define the decision line.

## Result
The learned line separates the two classes. Accuracy on the training data: 1.0.


## What I learned
- A perceptron can only learn a **linear** decision boundary.
- It only converges if the data is **linearly separable**.
- It cannot solve **XOR**, which is why multi-layer networks with non-linear activations are needed.
- Because training picks random points, the final line changes slightly on each run.

## Tech used
Python, NumPy, scikit-learn (for generating data), Matplotlib

## How to run
1. Install the libraries: `pip install numpy scikit-learn matplotlib`
2. Open `perceprton.ipynb` in Jupyter Notebook
3. Run the cells from top to bottom

## Next steps
- Multi-layer perceptron (MLP)
- Backpropagation
- Neural network from scratch in NumPy

## Credits
Based on CampusX's "100 Days of Machine Learning" perceptron video. The code was typed and run by me, and the explanations above are in my own words.