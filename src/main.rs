use rustnn::mnist;
use rustnn::knn::KNN;

fn main() {
  let (x_train, y_train, x_test, y_test) = mnist::load();
  
  let knn = KNN::new();
  knn.train(x_train, y_train);
}
 
