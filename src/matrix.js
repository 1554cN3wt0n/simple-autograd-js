const { Variable } = require("./variable");
class Matrix {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = new Array(this.rows * this.cols);
  }

  random() {
    for (let i = 0; i < this.rows * this.cols; i++) {
      this.data[i] = new Variable(Math.random());
    }
  }

  ones() {
    for (let i = 0; i < this.rows * this.cols; i++) {
      this.data[i] = new Variable(1);
    }
  }

  zeros() {
    for (let i = 0; i < this.rows * this.cols; i++) {
      this.data[i] = new Variable(0);
    }
  }

  set_data(raw_data) {
    for (let i = 0; i < this.rows * this.cols; i++) {
      this.data[i] = new Variable(raw_data[i]);
    }
  }

  add(m) {
    let res = new Matrix(this.rows, this.cols);
    let i1 = this.rows != 1 && m.rows == 1 ? 0 : 1;
    let j1 = this.cols != 1 && m.cols == 1 ? 0 : 1;
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        let idx1 = i * this.cols + j;
        let idx2 = i * i1 * m.cols + j * j1;
        res.data[idx1] = this.data[idx1].add(m.data[idx2]);
      }
    }
    return res;
  }

  sub(m) {
    let res = new Matrix(this.rows, this.cols);
    let i1 = this.rows != 1 && m.rows == 1 ? 0 : 1;
    let j1 = this.cols != 1 && m.cols == 1 ? 0 : 1;
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        let idx1 = i * this.cols + j;
        let idx2 = i * i1 * m.cols + j * j1;
        res.data[idx1] = this.data[idx1].sub(m.data[idx2]);
      }
    }
    return res;
  }

  mul(m) {
    let res = new Matrix(this.rows, this.cols);
    let i1 = this.rows != 1 && m.rows == 1 ? 0 : 1;
    let j1 = this.cols != 1 && m.cols == 1 ? 0 : 1;
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        let idx1 = i * this.cols + j;
        let idx2 = i * i1 * m.cols + j * j1;
        res.data[idx1] = this.data[idx1].mul(m.data[idx2]);
      }
    }
    return res;
  }

  div(m) {
    let res = new Matrix(this.rows, this.cols);
    let i1 = this.rows != 1 && m.rows == 1 ? 0 : 1;
    let j1 = this.cols != 1 && m.cols == 1 ? 0 : 1;
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        let idx1 = i * this.cols + j;
        let idx2 = i * i1 * m.cols + j * j1;
        res.data[idx1] = this.data[idx1].div(m.data[idx2]);
      }
    }
    return res;
  }

  sum() {
    let s = new Variable(0);
    this.data.forEach((elem) => {
      s = s.add(elem);
    });
    return s;
  }

  mean() {
    let s = this.sum();
    return s.div(new Variable(this.rows * this.cols));
  }

  pow(n) {
    let res = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows * this.cols; i++) {
      res.data[i] = this.data[i].pow(n);
    }
    return res;
  }

  dot(m) {
    let res = new Matrix(this.rows, m.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        let s = new Variable(0);
        for (let k = 0; k < this.cols; k++) {
          s = s.add(this.data[i * this.cols + k].mul(m.data[k * m.cols + j]));
        }
        res.data[i * m.cols + j] = s;
      }
    }
    return res;
  }

  apply_unitary_fn(fn) {
    let res = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows * this.cols; i++) {
      res.data[i] = fn(this.data[i]);
    }
    return res;
  }
}

module.exports = {
  Matrix,
};
