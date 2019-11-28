use ndarray::prelude::*;
use std::time::Instant;

pub struct KNN {
  x_train: Vec<Array2<u8>>,
  y_train: Vec<u8>,
}

impl KNN {
  pub fn new() -> KNN {
    KNN {x_train: Vec::new(), y_train: Vec::new()}
  }

  pub fn train(self, x_train: Vec<Array2<u8>>, y_train: Vec<u8>) -> KNN {
    KNN {x_train, y_train}
  }

  pub fn predict(self, input: Vec<Array2<u8>>) {
    println!("Training dim {:?}", self.x_train.len());
    println!("Data dimensions {:?}", input.len());

    let now = Instant::now();
  
    let transposed = self.x_train.transpose();
    println!("Transposed shape {:?} {:?}", transposed.shape, now.elapsed().as_millis());
    
    //let first = &input * transposed;
    //println!("Dotted shape {:?} {:?}", first.shape, now.elapsed().as_millis());

    //let first = first * -2.0;
    //println!("Escalar multiplication {:?}", now.elapsed().as_millis());
    
    //let second = self.x_train.map(|e| e * e);
    //let second = second.column_sum();
    //println!("Train input squared {:?} {:?}", second.shape, now.elapsed().as_millis());
    
    //let third = input.map(|e| e * e);
    //let third = third.column_sum();
    //println!("Test input squared {:?} {:?}", third.shape, now.elapsed().as_millis())
    
  }
}
