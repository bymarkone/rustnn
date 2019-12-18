use log::{info,debug};

use ndarray::prelude::*;
use std::time::Instant;
use std::cmp::Ordering::Equal;

pub struct KNN {
  x_train: Array2<f32>,
  y_train: Array1<f32>,
}

fn count(e: &usize, list: &Array1<f32>) -> usize {
  return list.iter().filter(|f| (**f).eq(&(*e as f32))).count();
}

fn argmax(list: Vec<usize>) -> f32 {
  return 0.0;
}

impl KNN {
  pub fn new() -> KNN {
    KNN {x_train: Array::zeros((0,0)), y_train: Array::zeros(0)}
  }

  pub fn train(self, x_train: Array2<f32>, y_train: Array1<f32>) -> KNN {
    KNN {x_train, y_train}
  }

  pub fn predict(self, input: Array2<f32>, labels: Array1<f32>) {
    info!("Training dim {:?}", self.x_train.dim());
    info!("Test dimensions {:?}", input.len());

    let now = Instant::now();

    // -2xy 
    let transposed = self.x_train.t();
    debug!("Transposed shape {:?} {:?}", transposed.dim(), now.elapsed().as_millis());
    let first = input.dot(&transposed);
    debug!("Dotted shape {:?} {:?}", first.dim(), now.elapsed().as_millis());
    let first = first * -2.0;
    debug!("Escalar multiplication {:?}", now.elapsed().as_millis());
    
    // x^2
    let second = self.x_train.map(|e| e * e).sum_axis(Axis(1));
    debug!("Train input squared {:?} {:?}", second.dim(), now.elapsed().as_millis());
    
    // y^2
    let third = input.map(|e| e * e).sum_axis(Axis(1)).insert_axis(Axis(1));
    debug!("Test input squared {:?} {:?}", third.dim(), now.elapsed().as_millis());
     
    // (x-y)^2 = -2xy + x^2 +y^2
    let result = first + second + third; 
    debug!("Result {:?} {:?}", result.dim(), now.elapsed().as_millis());

    let predicted = self.predict_labels(result);
    let matching = predicted.iter().zip(&labels).filter(|&(a, b)| a == b).count();
    info!("Accuracy is {:?}%", matching);
    
  }

  fn predict_labels(self, dists: Array2<f32>) -> Array1<f32> {
    let dim = dists.dim();    

    let mut y_pred: Vec<f32> = Vec::new();
 
    for i in 0..dim.0 {

      let row = dists.index_axis(Axis(0), i);

      let mut argsorted = row.iter().enumerate().collect::<Vec<_>>().to_vec();
      argsorted.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Equal));
      let sorted_indexes = Array::from(argsorted.iter().map(|(a,b)| a).collect::<Vec<&usize>>());

      let sorted_indexes_sliced = sorted_indexes.slice(s![..1]);

      let closest_labels = sorted_indexes_sliced.map(|e| self.y_train[**e]);

      let bincount = (0..9).into_iter().map(|e| count(&e, &closest_labels)).collect::<Vec<usize>>();
      debug!("Count is {:?}", bincount);
      y_pred.push(closest_labels[0]);
      
    }
    return Array::from(y_pred);
  } 

}
