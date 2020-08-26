use ndarray::prelude::*;

type Int = i8;
type Vector = Array1<Int>;
type Matrix = Array2<Int>;

pub trait Classifier {
  fn train(self, x_train: Matrix, y_train: Vector) -> Self;
  fn predict(self, x: Matrix) -> Vector;
  fn predict_one(self, x: Vector) -> Int;
}

pub struct LinearClassifier {
  weights: Vector,
  bias: Int,
  iterations: Int,
}

impl LinearClassifier {
  pub fn new(iterations: Int) -> LinearClassifier {
    return LinearClassifier { weights: Array::zeros(0), bias: 0 , iterations}
  }
}

impl Classifier for LinearClassifier {
  fn train(self, x_train: Matrix, y_train: Vector) -> LinearClassifier {
    let (n, d) = x_train.dim();
    let mut weights = Array::zeros(d);
    let mut bias = 0;

    for _ in 0..self.iterations {
      for i in 0..n {
        let mut label = y_train[i];
        label = if label == 0 { -1 } else { label }; 

        let input = x_train.index_axis(Axis(0), i);

        let prediction = label * (input.dot(&weights) + bias);

        if prediction <= 0 {
          weights = weights + &input * label;
          bias = bias + label;
        }
      }
    }   

    return LinearClassifier { weights, bias, iterations: self.iterations}

  } 

  fn predict(self, x: Matrix) -> Vector {
    let dotted = x.dot(&self.weights) + self.bias;
    dotted.map(|&outcome| if outcome > 0 { 1 } else { 0 })
  }

  fn predict_one(self, x: Vector) -> Int {
    let outcome = x.dot(&self.weights) + self.bias;
    match outcome > 0 {
      true => return 1,
      false => return 0
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;  

  #[test]
  fn it_learns_or_behavior() {
    let x_train = array![
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0]
    ];

    let y_train = array![1, 1, 1, 0];

    let classifier = LinearClassifier::new(100);  
    let trained_classifier = classifier.train(x_train, y_train);
    let results = trained_classifier.predict(array![[0, 0], [0, 1], [1, 1], [0, 0], [1, 0]]);
    
    assert_eq!(results[0], 0);
    assert_eq!(results[1], 1);
    assert_eq!(results[2], 1);
    assert_eq!(results[3], 0);
    assert_eq!(results[4], 1);

  }

  #[test]
  fn it_learns_and_behavior() {
    let x_train = array![
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0]
    ];

    let y_train = array![1, 0, 0, 0];

    let classifier = LinearClassifier::new(100);  
    let trained_classifier = classifier.train(x_train, y_train);
    let results = trained_classifier.predict(array![[0, 0], [0, 1], [1, 1], [0, 0], [1, 0]]);
    
    assert_eq!(results[0], 0);
    assert_eq!(results[1], 0);
    assert_eq!(results[2], 1);
    assert_eq!(results[3], 0);
    assert_eq!(results[4], 0);

  }
}
