class GradientDescent {
  constructor(params, lr) {
    this.params = params;
    this.lr = lr;
  }
  step() {
    this.params.forEach((param) => {
      param.data -= this.lr * param.grad;
    });
  }
  zero_grad() {
    this.params.forEach((param) => {
      param.zero_grad();
    });
  }
}

module.exports = {
  GradientDescent,
};
