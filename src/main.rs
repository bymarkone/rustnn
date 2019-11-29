use rustnn::mnist;
use rustnn::knn::KNN;

fn main() {
  let (x_train, y_train) = mnist::load();
  
  let knn = KNN::new();
  knn.train(x_train, y_train);
}
 
