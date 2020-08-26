use ndarray::prelude::*;
use std::io::Result;
use std::fs;
use log::{info};

type Int = u8;
type Vector = Array1<Int>;
type Matrix = Array2<Int>;


pub fn main() {
  let (x_train, y_train) = load_data();

  let classifier = Classifier::new(10);  
  let trained_classifier = classifier.train(x_train, y_train);
  let results = trained_classifier.predict(array![[1, 1],[1, 0],[0, 1],[0, 0]]);
  println!("{:?}", results);
  
}

pub struct Classifier {
  theta: Vector,
  t: Int,
}

impl Classifier {
  pub fn new(t: Int) -> Classifier {
    return Classifier { theta: Array::zeros(0), t }
  } 
  
  pub fn train(self, x_train: Matrix, y_train: Vector) -> Classifier {
    let (n, d) = x_train.dim();
    let mut theta = Array::zeros(d);

    for _ in 0..self.t {
      for i in 0..n {
        let label = y_train[i];
        let x = x_train.index_axis(Axis(0), i);

        if label * (x.dot(&theta)) <= 0 {
          theta = theta + &x * label;
        }
      }
    }   

    return Classifier { theta, t: self.t }

  }

  pub fn predict(self, x: Matrix) -> Vector {
    let dotted = x.dot(&self.theta);
    dotted.map(|&outcome| if outcome > 0 { 1 } else { 0 })
  }

  pub fn predict_one(self, x: Vector) -> Int {
    let outcome = x.dot(&self.theta);
    match outcome > 0 {
      true => return 1,
      false => return 0
    }
  }
}

fn load_data() -> (Array2<u8>, Array1<u8>) {
  let data = read("./data/or.txt").unwrap();
  (data.to_owned().slice_move(s![.., 0..2]), data.to_owned().slice_move(s![.., 2]))
}

fn read(filename: &str) -> Result<Array2<u8>> {
  let content: Vec<u8> = fs::read_to_string(filename)?
    .lines()
    .flat_map(|line| line.split_whitespace())
    .map(|integer| integer.parse().unwrap())
    .collect();

  let data = Array2::from_shape_vec((4,3), content.to_vec()).unwrap();

  info!("{}", data);
  Ok(data)
}

fn nand(first: bool, second: bool) -> bool {
  !(first && second)
}

pub fn or(first: bool, second: bool) -> bool {
  nand(nand(first, first), nand(second, second))
}

#[cfg(test)]
mod tests {
  use super::*;  

  #[test]
  fn it_executes_or_operation() {
    assert_eq!(or(true, true), true);
    assert_eq!(or(true, false), true);
    assert_eq!(or(false, true), true);
    assert_eq!(or(false, false), false);
  }   

  #[test]
  fn it_learns_or_behavior() {
    let x_train = array![
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0]
    ];

    let y_train = array![1, 1, 1, 0];

    let classifier = Classifier::new(10);
    let trained_classifier = classifier.train(x_train, y_train);
    let results = trained_classifier.predict(array![[0, 0], [0, 1], [1, 1], [0, 0], [1, 0]]);
    
    assert_eq!(results[0], 0);
    assert_eq!(results[1], 1);
    assert_eq!(results[2], 1);
    assert_eq!(results[3], 0);
    assert_eq!(results[4], 1);

  }
  
}
