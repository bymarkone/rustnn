extern crate ndarray;
extern crate blas_src;
extern crate accelerate_src;

use rustnn::mnist;
use rustnn::knn::KNN;

fn main() {
  let (x_train, y_train, x_test, y_test) = mnist::load(8000, 400);
  
  let knn = KNN::new();
  let knn = knn.train(x_train, y_train);

  knn.predict(x_test);
}
 
