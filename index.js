const { Matrix } = require("./src/matrix");
const {
  Linear,
  Sigmoid,
  Sequential,
  Tanh,
  ReLU,
  Softmax,
} = require("./src/nn");
const { MeanSquareErrorLoss, CrossEntropyLoss } = require("./src/loss");
const { GradientDescent } = require("./src/optim");

let model = new Sequential([
  new Linear(2, 5),
  new Tanh(),
  new Linear(5, 2),
  new Tanh(),
  new Softmax(),
]);
let loss = new CrossEntropyLoss();
let optim = new GradientDescent(model.parameters(), 0.1);

let x = new Matrix(4, 2);
let y = new Matrix(4, 2);

x.set_data([0, 0, 0, 1, 1, 0, 1, 1]);
y.set_data([1, 0, 0, 1, 0, 1, 1, 0]);

for (let i = 0; i < 20; i++) {
  optim.zero_grad();
  let o = model.forward(x);
  let l = loss.loss(o, y);
  l.backward(1);
  console.log(l.data);
  optim.step();
}

let o = model.forward(x);
console.log(o);
