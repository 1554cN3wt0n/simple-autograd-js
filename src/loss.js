const { Variable } = require("./variable");

class MeanSquareErrorLoss {
  constructor() {}

  loss(o, t) {
    return o.sub(t).pow(2).mean().mul(new Variable(2));
  }
}

module.exports = {
  MeanSquareErrorLoss,
};
