use log::{info,debug};

pub struct SVM {
  // add parameters 
  // add W 
}

impl SVM {
  pub fn new() -> SVM {
    SVM {}
  }

  pub fn train(self, x_train: Array2<f32>, y_train: Array1<f32>) -> SVM {
    // generate W based on dimensions of the data 
    // W = np.random.randn(3073, 10) * 0.0001

    
  }

  pub fn predict(self, x_train: Array2<f32>, y_train: Array1<f32>) -> SVM {

  }

}
