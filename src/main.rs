#[allow(unused_imports)]
use blas_src;
#[allow(unused_imports)]
use accelerate_src;

use env_logger;

use rustnn::mnist;
use rustnn::knn::KNN;

fn main() {
  env_logger::init();
}

fn run_knn() {
  let (x_train, y_train, x_test, y_test) = mnist::load(20000, 200);
  let knn = KNN::new(2);
  let knn = knn.train(x_train, y_train);
  knn.predict(x_test, y_test);
}
 
fn run_softmax() {
  let (x_train, y_train, x_test, y_test) = mnist::load(20000, 200);
  

}
