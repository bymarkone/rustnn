use ndarray::prelude::*;
use std::time::Instant;

pub struct KNN {
  x_train: Array2<f32>,
  y_train: Array1<f32>,
}

impl KNN {
  pub fn new() -> KNN {
    KNN {x_train: Array::zeros((0,0)), y_train: Array::zeros(0)}
  }

  pub fn train(self, x_train: Array2<f32>, y_train: Array1<f32>) -> KNN {
    KNN {x_train, y_train}
  }

  pub fn predict(self, input: Array2<f32>) {
    println!("Training dim {:?}", self.x_train.len());
    println!("Data dimensions {:?}", input.len());

    let now = Instant::now();

    // -2xy 
    let transposed = self.x_train.t();
    println!("Transposed shape {:?} {:?}", transposed.dim(), now.elapsed().as_millis());
    let first = input.dot(&transposed);
    println!("Dotted shape {:?} {:?}", first.dim(), now.elapsed().as_millis());
    let first = first * -2.0;
    println!("Escalar multiplication {:?}", now.elapsed().as_millis());
    
    // x^2
    let second = self.x_train.map(|e| e * e).sum_axis(Axis(1));
    println!("Train input squared {:?} {:?}", second.dim(), now.elapsed().as_millis());
    
    // y^2
    let third = input.map(|e| e * e).sum_axis(Axis(1)).insert_axis(Axis(1));
    println!("Test input squared {:?} {:?}", third.dim(), now.elapsed().as_millis());
     
    // (x-y)^2 = -2xy + x^2 +y^2
    let result = first + second + third; 
    println!("Result {:?} {:?}", result.dim(), now.elapsed().as_millis())
  }
}
