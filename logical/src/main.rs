use ndarray::prelude::*;
use std::io::Result;
use std::fs;
use log::{info};

use logical::classifier::Classifier;
use logical::classifier::LinearClassifier;

pub fn main() {
  let (x_train, y_train) = load_data();

  let classifier = LinearClassifier::new(100);  
  let trained_classifier = classifier.train(x_train, y_train);
  let results = trained_classifier.predict(array![[1, 1],[1, 0],[0, 1],[0, 0]]);
  println!("{:?}", results);
  
}

fn load_data() -> (Array2<i8>, Array1<i8>) {
  let data = read("./data/or.txt").unwrap();
  (data.to_owned().slice_move(s![.., 0..2]), data.to_owned().slice_move(s![.., 2]))
}

fn read(filename: &str) -> Result<Array2<i8>> {
  let content: Vec<i8> = fs::read_to_string(filename)?
    .lines()
    .flat_map(|line| line.split_whitespace())
    .map(|integer| integer.parse().unwrap())
    .collect();

  let data = Array2::from_shape_vec((4,3), content.to_vec()).unwrap();

  info!("{}", data);
  Ok(data)
}


