const { Variable } = require("./variable");

class SigmoidBackward {
  constructor(x, o) {
    this.x = x;
    this.o = o;
  }

  call(loss) {
    this.x.backward(loss * this.o * (1 - this.o));
  }
}

class TanhBackward {
  constructor(x, o) {
    this.x = x;
    this.o = o;
  }

  call(loss) {
    this.x.backward(loss * 0.5 * (1 - this.o ** 2));
  }
}

class ReLUBackward {
  constructor(x) {
    this.x = x;
  }

  call(loss) {
    this.x.backward(loss * (this.x.data > 0 ? 1 : 0));
  }
}

function sigmoid(x) {
  let res = 1 / (1 + Math.exp(-x.data));
  return new Variable(res, new SigmoidBackward(x, res));
}

function tanh(x) {
  let res = (1 - Math.exp(-x.data)) / (1 + Math.exp(-x.data));
  return new Variable(res, new TanhBackward(x, res));
}

function relu(x) {
  let res = x > 0 ? x : 0;
  return new Variable(res, new ReLUBackward(x, res));
}

module.exports = {
  sigmoid,
  tanh,
  relu,
};
