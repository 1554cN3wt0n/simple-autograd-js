const { Matrix } = require("./matrix");
const { sigmoid, tanh, relu } = require("./act_fun");

class Linear {
  constructor(n_inputs, n_outputs) {
    this.W = new Matrix(n_inputs, n_outputs);
    this.b = new Matrix(1, n_outputs);
    this.W.random();
    this.b.random();
  }
  forward(x) {
    return x.dot(this.W).add(this.b);
  }

  parameters() {
    return [...this.W.data, ...this.b.data];
  }
}

class Sigmoid {
  constructor() {}

  forward(x) {
    return x.apply_unitary_fn(sigmoid);
  }
  parameters() {
    return [];
  }
}

class Tanh {
  constructor() {}

  forward(x) {
    return x.apply_unitary_fn(tanh);
  }
  parameters() {
    return [];
  }
}

class ReLU {
  constructor() {}

  forward(x) {
    return x.apply_unitary_fn(relu);
  }
  parameters() {
    return [];
  }
}

class Sequential {
  constructor(layers) {
    this.layers = layers;
  }
  forward(x) {
    let o = x;
    this.layers.forEach((layer) => {
      o = layer.forward(o);
    });
    return o;
  }

  parameters() {
    let res = [];
    this.layers.forEach((layer) => {
      res = res.concat(layer.parameters());
    });
    return res;
  }
}

module.exports = {
  Linear,
  Sigmoid,
  Tanh,
  ReLU,
  Sequential,
};
