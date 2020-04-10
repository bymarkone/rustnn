use log::{info,debug};

use ndarray::prelude::*;
use std::time::Instant;
use std::cmp::Ordering::Equal;

pub struct KNN {
  x_train: Array2<f32>,
  y_train: Array1<f32>,
  k: usize,
}

fn count(e: &usize, list: &Array1<f32>) -> usize {
  return list.iter().filter(|f| (**f).eq(&(*e as f32))).count();
}

impl KNN {
  pub fn new(k: usize) -> KNN {
    KNN {x_train: Array::zeros((0,0)), y_train: Array::zeros(0), k}
  }

  pub fn train(self, x_train: Array2<f32>, y_train: Array1<f32>) -> KNN {
    KNN {x_train, y_train, k: self.k}
  }

  pub fn predict(self, input: Array2<f32>, labels: Array1<f32>) {
    info!("Training dim {:?}", self.x_train.dim());
    info!("Test dimensions {:?}", input.len());
    let now = Instant::now();

    let first = input.dot(&self.x_train.t()) * -2.0;
    debug!("Computed -2xy, result dimesions is {:?} (time: {:?})", first.dim(), now.elapsed().as_millis());
    
    let second = self.x_train.map(|e| e * e).sum_axis(Axis(1));
    debug!("Computed x^2, result dimension is {:?} (time: {:?})", second.dim(), now.elapsed().as_millis());
    
    let third = input.map(|e| e * e).sum_axis(Axis(1)).insert_axis(Axis(1));
    debug!("Computed y^2, result dimension is {:?} (time: {:?})", third.dim(), now.elapsed().as_millis());
     
    let result = first + second + third; 
    debug!("Computed (x-y)^2 = -2xy + x^2 + y^2, result dimension is {:?} (time: {:?})", result.dim(), now.elapsed().as_millis());

    let predicted = self.predict_labels(result);

    let matching = predicted.iter().zip(&labels).filter(|&(a, b)| a == b).count();

    debug!("Sample labels: {:?}", labels.slice(s![0..20]));
    debug!("Sample predicted: {:?}", predicted.slice(s![0..20]));

    info!("Accuracy is {:?} of {:?} ({:?}%)", matching, labels.len(), matching as f32 * 100.0 / labels.len() as f32);
  }

  fn predict_labels(self, dists: Array2<f32>) -> Array1<f32> {
    let dim = dists.dim();    

    let mut y_pred: Vec<f32> = Vec::new();
 
    for i in 0..dim.0 {

      let row = dists.index_axis(Axis(0), i);

      let mut argsorted = row.iter().enumerate().collect::<Vec<_>>().to_vec();
      argsorted.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Equal));
      let sorted_indexes = Array::from(argsorted.iter().map(|(a,_)| a).collect::<Vec<&usize>>());

      let sorted_indexes_sliced = sorted_indexes.slice(s![..self.k]);

      let closest_labels = sorted_indexes_sliced.map(|e| self.y_train[**e]);

      let bincount = (0..10)
          .into_iter()
          .map(|e| count(&e, &closest_labels))
          .enumerate()
          .map(|(x, y)| (y, x))
          .max()
          .unwrap()
          .1;

      debug!("Count is {:?}", bincount);
      y_pred.push(bincount as f32);
      
    }
    return Array::from(y_pred);
  } 

}
