const { Matrix } = require("../src/matrix");
const { Linear, Sigmoid, Sequential, ReLU } = require("../src/nn");
const { MeanSquareErrorLoss } = require("../src/loss");
const { GradientDescent } = require("../src/optim");

let model = new Sequential([
  new Linear(2, 5),
  new Sigmoid(),
  new Linear(5, 5),
  new Sigmoid(),
  new Linear(5, 1),
]);
let loss = new MeanSquareErrorLoss();
let optim = new GradientDescent(model.parameters(), 0.1);

let x = new Matrix(50, 2);
let y = new Matrix(50, 1);

let x_data = [];
let y_data = [];

for (let i = 0; i < 100; i++) {
  let a_ = Math.random() * 5;
  let b_ = Math.random() * 5;
  let c_ = Math.sin(a_) * Math.cos(b_) + a_ * b_;
  x_data.push(a_, b_);
  y_data.push(c_);
}

x.set_data(x_data);
y.set_data(y_data);

for (let i = 0; i < 1000; i++) {
  optim.zero_grad();
  let o = model.forward(x);
  let l = loss.loss(o, y);
  l.backward(1);
  console.log(l.data);
  optim.step();
}

let o = model.forward(x);
for (let i = 0; i < 50; i++) {
  console.log(y_data[i] + " " + o.data[i].data);
}
