const { Matrix } = require("../src/matrix");
const { Linear, Sigmoid, Sequential, Softmax } = require("../src/nn");
const { CrossEntropyLoss } = require("../src/loss");
const { GradientDescent } = require("../src/optim");

let model = new Sequential([
  new Linear(2, 5),
  new Sigmoid(),
  new Linear(5, 2),
  new Softmax(),
]);
let loss = new CrossEntropyLoss();
let optim = new GradientDescent(model.parameters(), 1);

let x = new Matrix(4, 2);
let y = new Matrix(4, 2);

x.set_data([0, 0, 0, 1, 1, 0, 1, 1]);
y.set_data([1, 0, 0, 1, 0, 1, 1, 0]);

for (let i = 0; i < 1000; i++) {
  optim.zero_grad();
  let o = model.forward(x);
  let l = loss.loss(o, y);
  l.backward(1);
  console.log(l.data);
  optim.step();
}

let o = model.forward(x);
console.log(o);
