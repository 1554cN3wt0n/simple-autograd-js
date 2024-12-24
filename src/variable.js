class AddBackward {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  call(loss) {
    this.x.backward(loss);
    this.y.backward(loss);
  }
}

class SubBackward {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  call(loss) {
    this.x.backward(loss);
    this.y.backward(-loss);
  }
}

class MulBackward {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  call(loss) {
    this.x.backward(this.y.data * loss);
    this.y.backward(this.x.data * loss);
  }
}

class DivBackward {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  call(loss) {
    this.x.backward(loss / this.y.data);
    this.y.backward((-this.x.data * loss) / this.y.data ** 2);
  }
}

class PowBackward {
  constructor(x, n) {
    this.x = x;
    this.n = n;
  }

  call(loss) {
    this.x.backward(this.n * this.x.data ** (this.n - 1) * loss);
  }
}

class LogBackward {
  constructor(x) {
    this.x = x;
  }

  call(loss) {
    this.x.backward(loss / this.x.data);
  }
}

class ExpBackward {
  constructor(x) {
    this.x = x;
  }

  call(loss) {
    this.x.backward(loss * Math.exp(this.x.data));
  }
}

class Variable {
  constructor(data, backward_fn) {
    this.data = data;
    this.backward_fn = backward_fn;
    this.grad = 0;
  }

  add(v) {
    return new Variable(this.data + v.data, new AddBackward(this, v));
  }

  sub(v) {
    return new Variable(this.data - v.data, new SubBackward(this, v));
  }

  mul(v) {
    return new Variable(this.data * v.data, new MulBackward(this, v));
  }

  div(v) {
    return new Variable(this.data / v.data, new DivBackward(this, v));
  }

  pow(n) {
    return new Variable(this.data ** n, new PowBackward(this, n));
  }

  log() {
    return new Variable(Math.log(this.data), new LogBackward(this));
  }

  exp() {
    return new Variable(Math.exp(this.data), new ExpBackward(this));
  }

  zero_grad() {
    this.grad = 0;
  }

  backward(loss) {
    if (this.backward_fn) {
      this.backward_fn.call(loss);
    } else {
      this.grad += loss;
    }
  }
}

module.exports = {
  Variable,
};
