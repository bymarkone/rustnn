pub struct KNN {
  train_input: Matrix<T>,
  train_labels: Vector<T>,
}

impl KNN {
  pub fn train(self, train_input, train_labels) -> KNN {
    KNN{train_input, train_labels}  
  }

  pub fn predict(self, input) {
    
    //M = np.dot(X, self.X_train.T)
    //te = np.square(X).sum(axis=1)
    //tr = np.square(self.X_train).sum(axis=1)
    //dists = np.sqrt(-2*M + np.matrix(tr) + np.matrix(te).T)
  }
}
