use rustnn::knn::KNN;
use rustnn::knn::data;

fn main() {
  let (x_train, y_train) = data::load();
  
  let knn = knn::new();
  knn.train(x_train, y_train);
}
 
