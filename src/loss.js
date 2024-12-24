const { Variable } = require("./variable");

class MeanSquareErrorLoss {
  constructor() {}

  loss(o, t) {
    return o.sub(t).pow(2).mean().mul(new Variable(2));
  }
}

class CrossEntropyLoss {
  constructor() {}

  loss(o, t) {
    return t.mul(o.log()).mean().mul(new Variable(-1));
  }
}

module.exports = {
  MeanSquareErrorLoss,
  CrossEntropyLoss,
};
