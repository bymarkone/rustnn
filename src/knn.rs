use ndarray::prelude::*;
use ndarray::SliceInfo;
use std::time::Instant;

pub struct KNN {
  x_train: Array2<f32>,
  y_train: Array1<f32>,
}

fn count(e: &usize, list: &Array1<f32>) -> usize {
  return 0;
}

fn max(list: Vec<usize>) -> f32 {
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
    println!("Training dim {:?}", self.x_train.dim());
    println!("Test dimensions {:?}", input.len());

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
    println!("Result {:?} {:?}", result.dim(), now.elapsed().as_millis());

    let predicted = self.predict_labels(result);
    let matching = predicted.iter().zip(&labels).filter(|&(a, b)| a == b).count();
    println!("Accuracy is {:?}%", matching);
    
  }

  fn predict_labels(self, dists: Array2<f32>) -> Array1<f32> {
    let dim = dists.dim();    

    let mut y_pred: Vec<f32> = Vec::new();
 
    for i in 0..dim.0 {
      let row = dists.index_axis(Axis(0), i);
      let mut argsorted = row.iter().enumerate().collect::<Vec<_>>().to_vec();
      argsorted.sort_by(|a,b| a.0.cmp(&b.0));
      let sorted_indexes = Array::from(argsorted.iter().map(|(a,b)| a).collect::<Vec<&usize>>());
      let sorted_indexes_sliced = sorted_indexes.slice(s![..1]);
      let closest_labels = sorted_indexes_sliced.map(|e| self.y_train[**e]);

      let bincount = vec![0,1,2].into_iter().map(|e| count(&e, &closest_labels)).collect::<Vec<usize>>();
      y_pred.push(max(bincount));
      
    }
    return Array::from(y_pred);
  } 

}
